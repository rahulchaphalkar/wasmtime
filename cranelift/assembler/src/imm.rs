//! Immediate operands to instructions.

#![allow(clippy::module_name_repetitions)]

use arbitrary::Arbitrary;

#[derive(Arbitrary, Clone, Debug)]
pub struct Imm8(u8);

impl Imm8 {
    #[must_use]
    pub fn value(&self) -> u8 {
        self.0
    }

    pub fn to_string(&self, extend: Extension) -> String {
        if matches!(extend, Extension::SignExtend) {
            let extended = self.0 as i8 as i64; // Convert u8 to i8, then to i64 for sign extension
            if extended < 10 && extended >= 0 {
                format!("${:x}", extended)
            } else {
                format!("$0x{:x}", extended)
            }
        } else if matches!(extend, Extension::ZeroExtend) {
            let extended = self.0 as u8 as u64;
            if self.0 < 10 {
                format!("${:x}", extended)
            } else {
                format!("$0x{:x}", extended)
            }
        } else if self.0 < 10 {
            format!("${:x}", self.0)
        } else {
            format!("$0x{:x}", self.0)
        }
    }
}

impl std::fmt::Display for Imm8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 < 10 {
            write!(f, "${:x}", self.0)
        } else {
            write!(f, "$0x{:x}", self.0)
        }
    }
}

#[derive(Arbitrary, Clone, Debug)]
pub struct Imm16(u16);

impl Imm16 {
    #[must_use]
    pub fn value(&self) -> u16 {
        self.0
    }

    pub fn to_string(&self, extend: Extension) -> String {
        if matches!(extend, Extension::SignExtend) {
            let extended = self.0 as i16 as i64; // Convert u8 to i8, then to i64 for sign extension
            if extended < 10 && extended >= 0 {
                format!("${:x}", extended)
            } else {
                format!("$0x{:x}", extended)
            }
        } else if matches!(extend, Extension::ZeroExtend) {
            let extended = self.0 as u16 as u64;
            if self.0 < 10 {
                format!("${:x}", extended)
            } else {
                format!("$0x{:x}", extended)
            }
        } else if self.0 < 10 {
            format!("${:x}", self.0)
        } else {
            format!("$0x{:x}", self.0)
        }
    }
}

impl std::fmt::Display for Imm16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 < 10 {
            write!(f, "${:x}", self.0)
        } else {
            write!(f, "$0x{:x}", self.0)
        }
    }
}

#[derive(Arbitrary, Clone, Debug)]
pub struct Imm32(u32);

impl Imm32 {
    #[must_use]
    pub fn value(&self) -> u32 {
        self.0
    }
    pub fn to_string(&self, extend: Extension) -> String {
        if matches!(extend, Extension::SignExtend) {
            let extended = self.0 as i32 as i64; // Convert u8 to i8, then to i64 for sign extension
            if extended < 10 && extended >= 0 {
                format!("${:x}", extended)
            } else {
                format!("$0x{:x}", extended)
            }
        } else if matches!(extend, Extension::ZeroExtend) {
            let extended = self.0 as u32 as u64;
            if self.0 < 10 {
                format!("${:x}", extended)
            } else {
                format!("$0x{:x}", extended)
            }
        } else if self.0 < 10 {
            format!("${:x}", self.0)
        } else {
            format!("$0x{:x}", self.0)
        }
    }
}

pub enum Extension {
    None,
    SignExtend,
    ZeroExtend,
}

#[derive(Clone, Copy, Debug, Arbitrary)]
pub struct Simm32(pub(crate) i32);

impl std::fmt::LowerHex for Simm32 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.0 == 0 {
            return Ok(());
        }
        if self.0 < 0 {
            write!(f, "-")?;
        }
        if self.0 > 9 || self.0 < -9 {
            write!(f, "0x")?;
        }
        let abs = match self.0.checked_abs() {
            Some(i) => i,
            None => -2_147_483_648,
        };
        std::fmt::LowerHex::fmt(&abs, f)
    }
}
