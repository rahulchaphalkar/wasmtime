use super::{fmtln, Formatter};
use crate::dsl::{self, LegacyPrefixes, Rex, Sse};
//use crate::dsl::encoding::{HasLegacyPrefix, HasOpcode};

impl dsl::Format {
    #[must_use]
    pub fn generate_att_style_operands(&self) -> String {
        let mut ordered_ops: Vec<_> = self
            .operands
            .iter()
            .map(|o| format!("{{{}}}", o.location))
            .collect();
        if ordered_ops.len() > 1 {
            let first = ordered_ops.remove(0);
            ordered_ops.push(first);
        }
        ordered_ops.join(", ")
    }

    pub fn generate_rex_encoding(&self, f: &mut Formatter, rex: &dsl::Rex) {
        self.generate_legacy_prefix(f, rex);
        self.generate_rex_prefix(f, rex);
        self.generate_opcode(f, rex);
        self.generate_modrm_byte(f, rex);
        self.generate_immediate(f);
    }

    pub fn generate_sse_encoding(&self, f: &mut Formatter, sse: &dsl::Sse) {
        self.generate_legacy_prefix(f, sse);
        self.generate_opcode(f, sse);
        //self.generate_modrm_byte(f, sse);
    }

    #[allow(clippy::unused_self)]
    fn generate_legacy_prefix<T: HasLegacyPrefix>(&self, f: &mut Formatter, encoding: &T) {
        let prefixes = encoding.get_legacy_prefix();
        if *prefixes != dsl::LegacyPrefixes::NoPrefix {
            f.empty_line();
            f.comment("Emit legacy prefixes.");
            match prefixes {
                dsl::LegacyPrefixes::NoPrefix => unreachable!(),
                dsl::LegacyPrefixes::_66 => fmtln!(f, "buf.put1(0x66);"),
                dsl::LegacyPrefixes::_F0 => fmtln!(f, "buf.put1(0xf0);"),
                dsl::LegacyPrefixes::_66F0 => fmtln!(f, "buf.put1(0x66); buf.put1(0xf0);"),
                dsl::LegacyPrefixes::_F2 => fmtln!(f, "buf.put1(0xf2);"),
                dsl::LegacyPrefixes::_F3 => fmtln!(f, "buf.put1(0xf3);"),
                dsl::LegacyPrefixes::_66F3 => fmtln!(f, "buf.put1(0x66); buf.put1(0xf3"),
            }
        }
    }

    #[allow(clippy::unused_self)]
    fn generate_opcode<T: HasOpcode>(&self, f: &mut Formatter, encoding: &T) {
        f.empty_line();
        f.comment("Emit opcode.");
        //fmtln!(f, "buf.put1(0x{:x});", rex.opcode);
        fmtln!(f, "buf.put1(0x{:x});", encoding.get_opcode());
    }

    fn generate_rex_prefix(&self, f: &mut Formatter, rex: &dsl::Rex) {
        use dsl::OperandKind::{FixedReg, Imm, Reg, RegMem};
        f.empty_line();
        f.comment("Emit REX prefix.");

        let rex_flags = if rex.w {
            "RexFlags::set_w()"
        } else {
            "RexFlags::clear_w()"
        };

        let find_8bit_registers =
            |l: &dsl::Location| l.bits() == 8 && matches!(l.kind(), Reg(_) | RegMem(_));
        if self.locations().any(find_8bit_registers) {
            fmtln!(f, "let mut rex = {rex_flags};");
            for op in self.locations().copied().filter(find_8bit_registers) {
                fmtln!(f, "self.{op}.always_emit_if_8bit_needed(&mut rex);");
            }
        } else {
            fmtln!(f, "let rex = {rex_flags};");
        }

        match self.operands_by_kind().as_slice() {
            [FixedReg(dst), Imm(_)] => {
                // TODO: don't emit REX here
                fmtln!(f, "let {dst} = {};", dst.generate_fixed_reg().unwrap());
                fmtln!(f, "let digit = 0x{:x};", rex.digit);
                fmtln!(f, "rex.emit_two_op(buf, digit, {dst}.enc());");
            }
            [RegMem(dst), Imm(_)] => {
                if rex.digit > 0 {
                    fmtln!(f, "let digit = 0x{:x};", rex.digit);
                    fmtln!(f, "match &self.{dst} {{");
                    f.indent(|f| {
                        fmtln!(
                            f,
                            "GprMem::Gpr({dst}) => rex.emit_two_op(buf, digit, {dst}.enc()),"
                        );
                        fmtln!(f, "GprMem::Mem({dst}) => {dst}.emit_rex_prefix(rex, digit, buf),");
                    });
                    fmtln!(f, "}}");
                } else {
                    todo!();
                }
            }
            [Reg(dst), RegMem(src)] => {
                fmtln!(f, "let {dst} = self.{dst}.enc();");
                fmtln!(f, "match &self.{src} {{");
                f.indent(|f| {
                    fmtln!(f, "GprMem::Gpr({src}) => rex.emit_two_op(buf, {dst}, {src}.enc()),");
                    fmtln!(f, "GprMem::Mem({src}) => {src}.emit_rex_prefix(rex, {dst}, buf),");
                });
                fmtln!(f, "}}");
            }
            [RegMem(dst), Reg(src)] => {
                fmtln!(f, "let {src} = self.{src}.enc();");
                fmtln!(f, "match &self.{dst} {{");
                f.indent(|f| {
                    fmtln!(f, "GprMem::Gpr({dst}) => rex.emit_two_op(buf, {src}, {dst}.enc()),");
                    fmtln!(f, "GprMem::Mem({dst}) => {dst}.emit_rex_prefix(rex, {src}, buf),");
                });
                fmtln!(f, "}}");
            }
            /*
            [XmmReg(dst), RegMem(src)] => {
                fmtln!(f, "let {dst} = self.{dst}.enc();");
                fmtln!(f, "match &self.{src} {{");
                f.indent(|f| {
                    fmtln!(f, "GprMem::Gpr({src}) => rex.emit_two_op(buf, {dst}, {src}.enc()),");
                    fmtln!(f, "GprMem::Mem({src}) => {src}.emit_rex_prefix(rex, {dst}, buf),");
                });
                fmtln!(f, "}}");
            }
            */

            unknown => todo!("unknown pattern: {unknown:?}"),
        }
    }

    fn generate_modrm_byte(&self, f: &mut Formatter, rex: &dsl::Rex) {
        use dsl::OperandKind::{FixedReg, Imm, Reg, RegMem};

        if let [FixedReg(_), Imm(_)] = self.operands_by_kind().as_slice() {
            // No need to emit a comment.
        } else {
            f.empty_line();
            f.comment("Emit ModR/M byte.");
        }

        match self.operands_by_kind().as_slice() {
            [FixedReg(_), Imm(_)] => {
                // No need to emit a ModRM byte: we know the register used.
            }
            [RegMem(dst), Imm(_)] => {
                debug_assert!(rex.digit > 0);
                fmtln!(f, "let digit = 0x{:x};", rex.digit);
                fmtln!(f, "match &self.{dst} {{");
                f.indent(|f| {
                    fmtln!(f, "GprMem::Gpr({dst}) => emit_modrm(buf, digit, {dst}.enc()),");
                    fmtln!(
                        f,
                        "GprMem::Mem({dst}) => emit_modrm_sib_disp(buf, off, digit, {dst}, 0, None),"
                    );
                });
                fmtln!(f, "}}");
            }
            [Reg(dst), RegMem(src)] => {
                fmtln!(f, "let {dst} = self.{dst}.enc();");
                fmtln!(f, "match &self.{src} {{");
                f.indent(|f| {
                    fmtln!(f, "GprMem::Gpr({src}) => emit_modrm(buf, {dst}, {src}.enc()),");
                    fmtln!(
                        f,
                        "GprMem::Mem({src}) => emit_modrm_sib_disp(buf, off, {dst}, {src}, 0, None),"
                    );
                });
                fmtln!(f, "}}");
            }
            [RegMem(dst), Reg(src)] => {
                fmtln!(f, "let {src} = self.{src}.enc();");
                fmtln!(f, "match &self.{dst} {{");
                f.indent(|f| {
                    fmtln!(f, "GprMem::Gpr({dst}) => emit_modrm(buf, {src}, {dst}.enc()),");
                    fmtln!(
                        f,
                        "GprMem::Mem({dst}) => emit_modrm_sib_disp(buf, off, {src}, {dst}, 0, None),"
                    );
                });
                fmtln!(f, "}}");
            }

            unknown => todo!("unknown pattern: {unknown:?}"),
        }
    }

    fn generate_immediate(&self, f: &mut Formatter) {
        use dsl::OperandKind::Imm;
        match self.operands_by_kind().as_slice() {
            [_, Imm(imm)] => {
                f.empty_line();
                f.comment("Emit immediate.");
                fmtln!(f, "let bytes = {};", imm.bytes());
                if imm.bits() == 32 {
                    fmtln!(f, "let value = self.{imm}.value();");
                } else {
                    fmtln!(f, "let value = u32::from(self.{imm}.value());");
                };
                fmtln!(f, "emit_simm(buf, bytes, value);");
            }
            unknown => {
                // Do nothing: no immediates expected.
                debug_assert!(!unknown.iter().any(|o| matches!(o, Imm(_))));
            }
        }
    }
}

pub trait HasLegacyPrefix {
    fn get_legacy_prefix(&self) -> &LegacyPrefixes;
}

impl HasLegacyPrefix for Rex {
    fn get_legacy_prefix(&self) -> &LegacyPrefixes {
        &self.prefixes
    }
}

impl HasLegacyPrefix for Sse {
    fn get_legacy_prefix(&self) -> &LegacyPrefixes {
        &self.prefixes
    }
}

pub trait HasOpcode {
    fn get_opcode(&self) -> u32;
}

impl HasOpcode for Rex {
    fn get_opcode(&self) -> u32 {
        self.opcode
    }
}

impl HasOpcode for Sse {
    fn get_opcode(&self) -> u32 {
        self.opcode
    }
}

pub trait HasModrmByte {
    fn generate_modrm_byte(&self, f: &mut Formatter, operands: &[dsl::Operand]);
}

impl HasModrmByte for Rex {
    fn generate_modrm_byte(&self, f: &mut Formatter, operands: &[dsl::Operand]) {
        use dsl::OperandKind::{FixedReg, Imm, Reg, RegMem};

        if let [dsl::Operand { kind: FixedReg(_), .. }, dsl::Operand { kind: Imm(_) }] = operands {
            // No need to emit a comment.
        } else {
            f.empty_line();
            f.comment("Emit ModR/M byte.");
        }

        match operands {
            [FixedReg(_), Imm(_)] => {
                // No need to emit a ModRM byte: we know the register used.
            }
            [RegMem(dst), Imm(_)] => {
                debug_assert!(self.digit > 0);
                fmtln!(f, "let digit = 0x{:x};", self.digit);
                fmtln!(f, "match &self.{dst} {{");
                f.indent(|f| {
                    fmtln!(f, "GprMem::Gpr({dst}) => emit_modrm(buf, digit, {dst}.enc()),");
                    fmtln!(
                        f,
                        "GprMem::Mem({dst}) => emit_modrm_sib_disp(buf, off, digit, {dst}, 0, None),"
                    );
                });
                fmtln!(f, "}}");
            }
            [Reg(dst), RegMem(src)] => {
                fmtln!(f, "let {dst} = self.{dst}.enc();");
                fmtln!(f, "match &self.{src} {{");
                f.indent(|f| {
                    fmtln!(f, "GprMem::Gpr({src}) => emit_modrm(buf, {dst}, {src}.enc()),");
                    fmtln!(
                        f,
                        "GprMem::Mem({src}) => emit_modrm_sib_disp(buf, off, {dst}, {src}, 0, None),"
                    );
                });
                fmtln!(f, "}}");
            }
            _ => unimplemented!("Unsupported operand combination for ModR/M byte generation"),
        }
    }
}
