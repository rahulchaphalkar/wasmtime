use crate::dsl;
use crate::dsl::format::Extension;
impl dsl::Location {
    /// `<operand type>`
    #[must_use]
    pub fn generate_type(&self) -> Option<&str> {
        use dsl::Location::*;
        match self {
            al | ax | eax | rax => None,
            imm8 => Some("Imm8"),
            imm16 => Some("Imm16"),
            imm32 => Some("Imm32"),
            r8 | r16 | r32 | r64 => Some("Gpr"),
            rm8 | rm16 | rm32 | rm64 => Some("GprMem"),
        }
    }

    /// `<operand>.to_string()`
    #[must_use]
    pub fn generate_to_string(&self, extension: Extension) -> String {
        use dsl::Location::*;
        match (self, extension) {
            (al, _) => "\"%al\"".into(),
            (ax, _) => "\"%ax\"".into(),
            (eax, _) => "\"%eax\"".into(),
            (rax, _) => "\"%rax\"".into(),
            //(imm8, _) | (imm16, _) => format!("self.{self}.to_string()"),
            (imm8, Extension::None) | (imm16, Extension::None) => {
                format!("self.{self}.to_string(Extension::None)")
            }
            (imm32, Extension::None) => format!("self.{self}.to_string(Extension::None)"),
            (imm8, Extension::SignExtend)
            | (imm16, Extension::SignExtend)
            | (imm32, Extension::SignExtend) => {
                format!("self.{self}.to_string(Extension::SignExtend)")
            }
            (imm8, Extension::ZeroExtend) | (imm16, Extension::ZeroExtend) => {
                format!("self.{self}.to_string(Extension::ZeroExtend)")
            }
            (imm32, Extension::ZeroExtend) => {
                format!("self.{self}.to_string(Extension::ZeroExtend)")
            }
            //imm32 => format!("self.{self}.to_string(Extension::SignExtend)"),
            //imm32 => format!("self.{self}.to_string(self.{self})"),
            (r8, _)
            | (r16, _)
            | (r32, _)
            | (r64, _)
            | (rm8, _)
            | (rm16, _)
            | (rm32, _)
            | (rm64, _) => match self.generate_size() {
                Some(size) => format!("self.{self}.to_string({size})"),
                None => unreachable!(),
            },
        }
    }

    /// `Size::<operand size>`
    #[must_use]
    pub fn generate_size(&self) -> Option<&str> {
        use dsl::Location::*;
        match self {
            al | ax | eax | rax | imm8 | imm16 | imm32 => None,
            r8 | rm8 => Some("Size::Byte"),
            r16 | rm16 => Some("Size::Word"),
            r32 | rm32 => Some("Size::Doubleword"),
            r64 | rm64 => Some("Size::Quadword"),
        }
    }

    /// `Gpr(regs::...)`
    #[must_use]
    pub fn generate_fixed_reg(&self) -> Option<&str> {
        use dsl::Location::*;
        match self {
            al | ax | eax | rax => Some("Gpr(reg::ENC_RAX)"),
            imm8 | imm16 | imm32 | r8 | r16 | r32 | r64 | rm8 | rm16 | rm32 | rm64 => None,
        }
    }
}

impl dsl::Mutability {
    #[must_use]
    pub fn generate_regalloc_call(&self) -> &str {
        match self {
            dsl::Mutability::Read => "read",
            dsl::Mutability::ReadWrite => "read_write",
        }
    }
}
