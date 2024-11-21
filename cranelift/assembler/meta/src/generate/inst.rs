use super::{fmtln, generate_derive, Formatter};
use crate::dsl;

impl dsl::Inst {
    /// `struct <inst> { <op>: Reg, <op>: Reg, ... }`
    pub fn generate_struct(&self, f: &mut Formatter) {
        let immediate_name = self.immediate_name();
        let struct_name = self.struct_name() + &immediate_name;
        let struct_fields = self.struct_fields();

        f.line(format!("/// `{self}`"), None);
        generate_derive(f);
        fmtln!(f, "pub struct {struct_name} {{ {struct_fields} }}");
    }

    /// `pub <op>: <type>, *`
    #[must_use]
    fn struct_fields(&self) -> String {
        self.format
            .locations()
            .filter_map(|l| {
                let ty = l.generate_type()?;
                Some(format!("pub {l}: {ty}"))
            })
            .collect::<Vec<String>>()
            .join(", ")
    }

    /// `<class name>_<format name>`
    #[must_use]
    fn struct_name(&self) -> String {
        format!("{}_{}", self.name.to_lowercase(), self.format.name.to_lowercase())
    }

    #[must_use]
    fn immediate_name(&self) -> String {
        self.format
            .operands_by_kind()
            .iter()
            .find(|op| match op {
                dsl::format::OperandKind::Imm(_) => true,
                _ => false,
            })
            .map_or_else(
                || String::new(),
                |op| match op {
                    dsl::format::OperandKind::Imm(dsl::Location::imm8) => "_ib".to_string(),
                    dsl::format::OperandKind::Imm(dsl::Location::imm16) => "_iw".to_string(),
                    dsl::format::OperandKind::Imm(dsl::Location::imm32) => "_id".to_string(),
                    _ => "".to_string(),
                },
            )
    }

    /// `<inst>_<immediate_name>(<inst>),`
    pub fn generate_enum_variant(&self, f: &mut Formatter) {
        let variant_name = self.struct_name();
        let immediate_name = self
            .format
            .operands_by_kind()
            .iter()
            .find(|op| match op {
                dsl::format::OperandKind::Imm(_) => true,
                _ => false,
            })
            .map_or_else(
                || String::new(),
                |op| match op {
                    dsl::format::OperandKind::Imm(dsl::Location::imm8) => "_ib".to_string(),
                    dsl::format::OperandKind::Imm(dsl::Location::imm16) => "_iw".to_string(),
                    dsl::format::OperandKind::Imm(dsl::Location::imm32) => "_id".to_string(),
                    _ => "".to_string(),
                },
            );

        let struct_name = self.struct_name() + &immediate_name;
        fmtln!(f, "{variant_name}{immediate_name}({struct_name}),");
    }

    // `Self::<inst>(i) => write!(f, "{}", i),`
    pub fn generate_variant_display(&self, f: &mut Formatter) {
        let variant_name = self.struct_name();
        let immediate_name = self.immediate_name();
        fmtln!(f, "Self::{variant_name}{immediate_name}(i) => write!(f, \"{{i}}\"),");
    }

    // `Self::<inst>(i) => i.encode(b),`
    pub fn generate_variant_encode(&self, f: &mut Formatter) {
        let variant_name = self.struct_name();
        let immediate_name = self.immediate_name();
        fmtln!(f, "Self::{variant_name}{immediate_name}(i) => i.encode(b),");
    }

    /// `impl <inst> { ... }`
    pub fn generate_struct_impl(&self, f: &mut Formatter) {
        let struct_name = self.struct_name();
        let immediate_name = self.immediate_name();
        fmtln!(f, "impl {struct_name}{immediate_name} {{");
        f.indent_push();
        self.generate_encode_function(f);
        f.empty_line();
        self.generate_regalloc_function(f);
        f.indent_pop();
        fmtln!(f, "}}");
    }

    /// `fn encode(&self, buf: &mut Vec<u8>) { ... }`
    pub fn generate_encode_function(&self, f: &mut Formatter) {
        fmtln!(f, "pub fn encode(&self, buf: &mut impl CodeSink) {{");
        f.indent_push();

        // Emit trap.
        if let Some(op) = self.format.uses_memory() {
            f.empty_line();
            f.comment("Emit trap.");
            fmtln!(f, "if let GprMem::Mem({op}) = &self.{op} {{");
            f.indent(|f| {
                fmtln!(f, "if let Some(trap_code) = {op}.trap_code() {{");
                f.indent(|f| {
                    fmtln!(f, "buf.add_trap(trap_code);");
                });
                fmtln!(f, "}}");
            });
            fmtln!(f, "}}");
        }

        match &self.encoding {
            dsl::Encoding::Rex(rex) => self.format.generate_rex_encoding(f, rex),
            dsl::Encoding::Vex(_) => todo!(),
        }

        f.indent_pop();
        fmtln!(f, "}}");
    }

    /// `fn regalloc(&self) -> String { ... }`
    pub fn generate_regalloc_function(&self, f: &mut Formatter) {
        use dsl::OperandKind::*;
        fmtln!(f, "pub fn regalloc(&mut self, visitor: &mut impl RegallocVisitor) {{");
        f.indent(|f| {
            for o in &self.format.operands {
                match o.location.kind() {
                    Imm(_) => {
                        // Immediates do not need register allocation.
                    }
                    FixedReg(_) => {
                        let call = o.mutability.generate_regalloc_call();
                        let Some(fixed) = o.location.generate_fixed_reg() else {
                            unreachable!()
                        };
                        fmtln!(f, "visitor.fixed_{call}({fixed}.enc());");
                    }
                    Reg(reg) => {
                        let call = o.mutability.generate_regalloc_call();
                        fmtln!(f, "self.{reg}.{call}(visitor);");
                    }
                    RegMem(rm) => {
                        let call = o.mutability.generate_regalloc_call();
                        fmtln!(f, "self.{rm}.{call}(visitor);");
                    }
                }
            }
        });
        fmtln!(f, "}}");
    }

    /// `impl Debug for <inst> { ... }`
    pub fn generate_display_impl(&self, f: &mut Formatter) {
        let immediate_name = self.immediate_name();
        let struct_name = self.struct_name() + &immediate_name;
        fmtln!(f, "impl std::fmt::Display for {struct_name} {{");
        f.indent_push();
        fmtln!(f, "fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {{");

        f.indent_push();
        // let mut i = 0;
        // for op in self.format.locations() {
        //     let extend = self.format.operands[i].extension;
        //     fmtln!(f, "// Operand: {op} Extend: {extend}");
        //let to_string = op.generate_to_string();
        //fmtln!(f, "//let {op} = {to_string};");
        //     i += 1;
        // }

        //let mut i = 0;
        //for op in &self.format.operands {
        //    let to_string = op.generate_to_string();
        // }
        let mut i = 0;
        for op in self.format.locations() {
            let extension = self.format.operands[i].extension;
            fmtln!(f, "// Iter: {i} Extend: {extension}");
            let to_string = op.generate_to_string(extension);
            fmtln!(f, "let {op} = {to_string};");
            i += 1;
        }
        let inst_name = &self.name;
        let ordered_ops = self.format.generate_att_style_operands();
        fmtln!(f, "write!(f, \"{inst_name} {ordered_ops}\")");
        f.indent_pop();
        fmtln!(f, "}}");

        f.indent_pop();
        fmtln!(f, "}}");
    }
}
