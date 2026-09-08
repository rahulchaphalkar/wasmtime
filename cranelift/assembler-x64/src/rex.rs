//! Encoding logic for REX instructions.

use crate::api::CodeSink;

fn low8_will_sign_extend_to_32(xs: i32) -> bool {
    xs == ((xs << 24) >> 24)
}

/// Encode the ModR/M byte.
#[inline]
pub(crate) fn encode_modrm(m0d: u8, enc_reg_g: u8, rm_e: u8) -> u8 {
    debug_assert!(m0d < 4);
    debug_assert!(enc_reg_g < 8);
    debug_assert!(rm_e < 8);
    ((m0d & 3) << 6) | ((enc_reg_g & 7) << 3) | (rm_e & 7)
}

/// Encode the SIB byte (scale-index-base).
#[inline]
pub(crate) fn encode_sib(scale: u8, enc_index: u8, enc_base: u8) -> u8 {
    debug_assert!(scale < 4);
    debug_assert!(enc_index < 8);
    debug_assert!(enc_base < 8);
    ((scale & 3) << 6) | ((enc_index & 7) << 3) | (enc_base & 7)
}

/// Tests whether `enc` is `rsp`, `rbp`, `rsi`, or `rdi`. If 8-bit register
/// sizes are used then it means a REX prefix is required.
///
/// This function is used below in combination with `uses_8bit` booleans to
/// determine the `RexPrefix::must_emit` flag. Table 3-2 in volume 1 of the
/// Intel manual details how referencing `dil`, the low 8-bits of `rdi`,
/// requires the use of the REX prefix as without it, it would otherwise
/// reference the `AH` register.
///
/// This is used whenever a register is encoded with a `RexPrefix` and is also
/// only used if the register is referenced in its 8-bit form. That means for
/// example that when encoding addressing modes this function is not used.
/// Addressing modes use 64-bit versions of registers meaning that the 8-bit
/// special case does not apply.
const fn is_special_if_8bit(enc: u8) -> bool {
    enc >= 4 && enc <= 7
}

/// Opcode maps representable by the APX REX2 `M` bit.
#[derive(Clone, Copy)]
pub(crate) enum Rex2Map {
    Map0,
    #[allow(dead_code, reason = "used once a map-1 instruction opts into REX2")]
    Map1,
    Unsupported,
}

/// Construct and emit a REX or REX2 prefix.
///
/// For more details, see section 2.2.1, "REX Prefixes" in Intel's reference
/// manual.
#[derive(Clone, Copy)]
pub struct RexPrefix {
    rex: u8,
    rex2: u8,
    rex2_map: Rex2Map,
    needs_rex2: bool,
    must_emit: bool,
}

impl RexPrefix {
    /// Construct the [`RexPrefix`] for a unary instruction.
    ///
    /// Used with a single register operand:
    /// - `x` and `r` are unused.
    /// - `b` extends the `reg` register, allowing access to r8-r15, or the top
    ///   bit of the opcode digit.
    #[inline]
    #[must_use]
    pub const fn one_op(enc: u8, w_bit: bool, uses_8bit: bool) -> Self {
        assert!(enc < 32, "invalid register encoding");
        let must_emit = uses_8bit && is_special_if_8bit(enc);
        let w = if w_bit { 1 } else { 0 };
        let b3 = (enc >> 3) & 1;
        let b4 = (enc >> 4) & 1;
        Self {
            rex: 0x40 | (w << 3) | b3,
            rex2: (b4 << 4) | (w << 3) | b3,
            rex2_map: Rex2Map::Map0,
            needs_rex2: b4 != 0,
            must_emit,
        }
    }

    /// Construct the [`RexPrefix`] for a binary instruction.
    ///
    /// Used without a SIB byte or for register-to-register addressing:
    /// - `r` extends the `reg` operand, allowing access to r8-r15.
    /// - `x` is unused.
    /// - `b` extends the `r/m` operand, allowing access to r8-r15.
    #[inline]
    #[must_use]
    pub const fn two_op(enc_reg: u8, enc_rm: u8, w_bit: bool, uses_8bit: bool) -> Self {
        let mut ret = RexPrefix::mem_op(enc_reg, enc_rm, w_bit, uses_8bit);
        if uses_8bit && is_special_if_8bit(enc_rm) {
            ret.must_emit = true;
        }
        ret
    }

    /// Construct the [`RexPrefix`] for a binary instruction where one operand
    /// is a memory address.
    ///
    /// This is the same as [`RexPrefix::two_op`] except that `enc_rm` is
    /// guaranteed to address a 64-bit register. This has a slightly different
    /// meaning when `uses_8bit` is `true` to omit the REX prefix in more cases
    /// than `two_op` would emit.
    #[inline]
    #[must_use]
    pub const fn mem_op(enc_reg: u8, enc_rm: u8, w_bit: bool, uses_8bit: bool) -> Self {
        assert!(enc_reg < 32, "invalid register encoding");
        assert!(enc_rm < 32, "invalid register encoding");
        let must_emit = uses_8bit && is_special_if_8bit(enc_reg);
        let w = if w_bit { 1 } else { 0 };
        let r3 = (enc_reg >> 3) & 1;
        let r4 = (enc_reg >> 4) & 1;
        let b3 = (enc_rm >> 3) & 1;
        let b4 = (enc_rm >> 4) & 1;
        Self {
            rex: 0x40 | (w << 3) | (r3 << 2) | b3,
            rex2: (r4 << 6) | (b4 << 4) | (w << 3) | (r3 << 2) | b3,
            rex2_map: Rex2Map::Map0,
            needs_rex2: (r4 | b4) != 0,
            must_emit,
        }
    }

    /// Construct the [`RexPrefix`] for an instruction using an opcode digit.
    ///
    /// :
    /// - `r` extends the opcode digit.
    /// - `x` is unused.
    /// - `b` extends the `reg` operand, allowing access to r8-r15.
    #[inline]
    #[must_use]
    pub const fn with_digit(digit: u8, enc_reg: u8, w_bit: bool, uses_8bit: bool) -> Self {
        assert!(digit < 8, "invalid opcode digit");
        Self::two_op(digit, enc_reg, w_bit, uses_8bit)
    }

    /// Construct the [`RexPrefix`] for a ternary instruction, typically using a
    /// memory address.
    ///
    /// Used with a SIB byte:
    /// - `r` extends the `reg` operand, allowing access to r8-r15.
    /// - `x` extends the index register, allowing access to r8-r15.
    /// - `b` extends the base register, allowing access to r8-r15.
    #[inline]
    #[must_use]
    pub const fn three_op(
        enc_reg: u8,
        enc_index: u8,
        enc_base: u8,
        w_bit: bool,
        uses_8bit: bool,
    ) -> Self {
        assert!(enc_reg < 32, "invalid register encoding");
        assert!(enc_index < 32, "invalid register encoding");
        assert!(enc_base < 32, "invalid register encoding");
        let must_emit = uses_8bit && is_special_if_8bit(enc_reg);
        let w = if w_bit { 1 } else { 0 };
        let r3 = (enc_reg >> 3) & 1;
        let r4 = (enc_reg >> 4) & 1;
        let x3 = (enc_index >> 3) & 1;
        let x4 = (enc_index >> 4) & 1;
        let b3 = (enc_base >> 3) & 1;
        let b4 = (enc_base >> 4) & 1;
        Self {
            rex: 0x40 | (w << 3) | (r3 << 2) | (x3 << 1) | b3,
            rex2: (r4 << 6) | (x4 << 5) | (b4 << 4) | (w << 3) | (r3 << 2) | (x3 << 1) | b3,
            rex2_map: Rex2Map::Map0,
            needs_rex2: (r4 | x4 | b4) != 0,
            must_emit,
        }
    }

    #[must_use]
    pub(crate) const fn with_rex2_map(mut self, map: Rex2Map) -> Self {
        self.rex2_map = map;
        self
    }

    /// Possibly emit a REX or REX2 prefix.
    ///
    /// Returns `true` when REX2 was emitted. A map-1 instruction alone does not
    /// select REX2; the prefix is selected only when a register needs bit 4.
    #[inline]
    pub fn encode(&self, sink: &mut impl CodeSink) -> bool {
        if self.needs_rex2 {
            let m = match self.rex2_map {
                Rex2Map::Map0 => 0,
                Rex2Map::Map1 => 1,
                Rex2Map::Unsupported => panic!("REX2 cannot encode this opcode map"),
            };
            sink.put1(0xd5);
            sink.put1((m << 7) | self.rex2);
            true
        } else {
            if self.rex != 0x40 || self.must_emit {
                sink.put1(self.rex);
            }
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Rex2Map, RexPrefix};
    use alloc::vec::Vec;

    #[test]
    fn dynamically_select_rex_or_rex2() {
        let mut bytes = Vec::new();
        assert!(!RexPrefix::two_op(0, 1, false, false).encode(&mut bytes));
        assert_eq!(bytes, []);

        bytes.clear();
        assert!(!RexPrefix::two_op(8, 9, true, false).encode(&mut bytes));
        assert_eq!(bytes, [0x4d]);

        bytes.clear();
        assert!(RexPrefix::two_op(16, 17, true, false).encode(&mut bytes));
        assert_eq!(bytes, [0xd5, 0x58]);

        bytes.clear();
        let prefix = RexPrefix::two_op(16, 1, false, false).with_rex2_map(Rex2Map::Map1);
        assert!(prefix.encode(&mut bytes));
        assert_eq!(bytes, [0xd5, 0xc0]);

        bytes.clear();
        let prefix = RexPrefix::two_op(0, 1, false, false).with_rex2_map(Rex2Map::Map1);
        assert!(!prefix.encode(&mut bytes));
        assert_eq!(bytes, []);
    }

    #[test]
    fn encode_rex2_sib_extensions() {
        let mut bytes = Vec::new();
        assert!(RexPrefix::three_op(24, 18, 11, true, false).encode(&mut bytes));
        assert_eq!(bytes, [0xd5, 0x6d]);
    }

    #[test]
    #[should_panic(expected = "REX2 cannot encode this opcode map")]
    fn reject_rex2_for_unsupported_map() {
        let prefix = RexPrefix::one_op(16, false, false).with_rex2_map(Rex2Map::Unsupported);
        prefix.encode(&mut Vec::new());
    }
}

/// The displacement bytes used after the ModR/M and SIB bytes.
#[derive(Copy, Clone)]
pub enum Disp {
    None,
    Imm8(i8),
    Imm32(i32),
}

impl Disp {
    /// Classifies the 32-bit immediate `val` as how this can be encoded
    /// with ModRM/SIB bytes.
    ///
    /// For `evex_scaling` according to Section 2.7.5 of Intel's manual:
    ///
    /// > EVEX-encoded instructions always use a compressed displacement scheme
    /// > by multiplying disp8 in conjunction with a scaling factor N that is
    /// > determined based on the vector length, the value of EVEX.b bit
    /// > (embedded broadcast) and the input element size of the instruction
    ///
    /// The `evex_scaling` factor provided here is `Some(N)` for EVEX
    /// instructions.  This is taken into account where the `Imm` value
    /// contained is the raw byte offset.
    pub fn new(val: i32, evex_scaling: Option<i8>) -> Disp {
        if val == 0 {
            return Disp::None;
        }
        match evex_scaling {
            Some(scaling) => {
                if val % i32::from(scaling) == 0 {
                    let scaled = val / i32::from(scaling);
                    if low8_will_sign_extend_to_32(scaled) {
                        return Disp::Imm8(scaled as i8);
                    }
                }
                Disp::Imm32(val)
            }
            None => match i8::try_from(val) {
                Ok(val) => Disp::Imm8(val),
                Err(_) => Disp::Imm32(val),
            },
        }
    }

    /// Forces `Imm::None` to become `Imm::Imm8(0)`, used for special cases
    /// where some base registers require an immediate.
    pub fn force_immediate(&mut self) {
        if let Disp::None = self {
            *self = Disp::Imm8(0);
        }
    }

    /// Returns the two "mod" bits present at the upper bits of the mod/rm
    /// byte.
    pub fn m0d(self) -> u8 {
        match self {
            Disp::None => 0b00,
            Disp::Imm8(_) => 0b01,
            Disp::Imm32(_) => 0b10,
        }
    }

    /// Emit the truncated immediate into the code sink.
    pub fn emit(self, sink: &mut impl CodeSink) {
        match self {
            Disp::None => {}
            Disp::Imm8(n) => sink.put1(n as u8),
            Disp::Imm32(n) => sink.put4(n as u32),
        }
    }
}
