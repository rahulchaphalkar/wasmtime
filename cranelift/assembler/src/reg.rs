//! Pure register operands.

use crate::{alloc::RegallocVisitor, rex::RexFlags};
use arbitrary::Arbitrary;

pub const ENC_RAX: u8 = 0;
pub const ENC_RCX: u8 = 1;
pub const ENC_RDX: u8 = 2;
pub const ENC_RBX: u8 = 3;
pub const ENC_RSP: u8 = 4;
pub const ENC_RBP: u8 = 5;
pub const ENC_RSI: u8 = 6;
pub const ENC_RDI: u8 = 7;
pub const ENC_R8: u8 = 8;
pub const ENC_R9: u8 = 9;
pub const ENC_R10: u8 = 10;
pub const ENC_R11: u8 = 11;
pub const ENC_R12: u8 = 12;
pub const ENC_R13: u8 = 13;
pub const ENC_R14: u8 = 14;
pub const ENC_R15: u8 = 15;

#[derive(Clone, Debug)]
pub struct Gpr(pub(crate) u8);
impl Gpr {
    pub fn enc(&self) -> u8 {
        self.0
    }

    pub fn always_emit_if_8bit_needed(&self, rex: &mut RexFlags) {
        let enc_reg = self.enc();
        if (4..=7).contains(&enc_reg) {
            rex.always_emit();
        }
    }

    pub fn to_string(&self, size: Size) -> &str {
        use Size::{Byte, Doubleword, Quadword, Word};
        match self.enc() {
            ENC_RAX => match size {
                Byte => "%al",
                Word => "%ax",
                Doubleword => "%eax",
                Quadword => "%rax",
            },
            ENC_RBX => match size {
                Byte => "%bl",
                Word => "%bx",
                Doubleword => "%ebx",
                Quadword => "%rbx",
            },
            ENC_RCX => match size {
                Byte => "%cl",
                Word => "%cx",
                Doubleword => "%ecx",
                Quadword => "%rcx",
            },
            ENC_RDX => match size {
                Byte => "%dl",
                Word => "%dx",
                Doubleword => "%edx",
                Quadword => "%rdx",
            },
            ENC_RSI => match size {
                Byte => "%sil",
                Word => "%si",
                Doubleword => "%esi",
                Quadword => "%rsi",
            },
            ENC_RDI => match size {
                Byte => "%dil",
                Word => "%di",
                Doubleword => "%edi",
                Quadword => "%rdi",
            },
            ENC_RBP => match size {
                Byte => "%bpl",
                Word => "%bp",
                Doubleword => "%ebp",
                Quadword => "%rbp",
            },
            ENC_RSP => match size {
                Byte => "%spl",
                Word => "%sp",
                Doubleword => "%esp",
                Quadword => "%rsp",
            },
            ENC_R8 => match size {
                Byte => "%r8b",
                Word => "%r8w",
                Doubleword => "%r8d",
                Quadword => "%r8",
            },
            ENC_R9 => match size {
                Byte => "%r9b",
                Word => "%r9w",
                Doubleword => "%r9d",
                Quadword => "%r9",
            },
            ENC_R10 => match size {
                Byte => "%r10b",
                Word => "%r10w",
                Doubleword => "%r10d",
                Quadword => "%r10",
            },
            ENC_R11 => match size {
                Byte => "%r11b",
                Word => "%r11w",
                Doubleword => "%r11d",
                Quadword => "%r11",
            },
            ENC_R12 => match size {
                Byte => "%r12b",
                Word => "%r12w",
                Doubleword => "%r12d",
                Quadword => "%r12",
            },
            ENC_R13 => match size {
                Byte => "%r13b",
                Word => "%r13w",
                Doubleword => "%r13d",
                Quadword => "%r13",
            },
            ENC_R14 => match size {
                Byte => "%r14b",
                Word => "%r14w",
                Doubleword => "%r14d",
                Quadword => "%r14",
            },
            ENC_R15 => match size {
                Byte => "%r15b",
                Word => "%r15w",
                Doubleword => "%r15d",
                Quadword => "%r15",
            },
            _ => panic!("invalid reg: {}", self.0),
        }
    }

    pub fn read(&mut self, visitor: &mut impl RegallocVisitor) {
        // TODO: allow regalloc to replace a virtual register with a real one.
        visitor.read(self.0);
    }

    pub fn read_write(&mut self, visitor: &mut impl RegallocVisitor) {
        // TODO: allow regalloc to replace a virtual register with a real one.
        visitor.read_write(self.0);
    }
}

impl<'a> Arbitrary<'a> for Gpr {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self(u.int_in_range(0..=15)?))
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Size {
    Byte,
    Word,
    Doubleword,
    Quadword,
}

#[derive(Clone, Debug)]
pub struct Gpr2MinusRsp(Gpr);

impl Gpr2MinusRsp {
    pub fn enc(&self) -> u8 {
        self.0.enc()
    }
    pub fn to_string(&self, size: Size) -> &str {
        self.0.to_string(size)
    }
}

impl<'a> Arbitrary<'a> for Gpr2MinusRsp {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let gpr = u.choose(&[
            ENC_RAX, ENC_RCX, ENC_RDX, ENC_RBX, ENC_RBP, ENC_RSI, ENC_RDI, ENC_R8, ENC_R9, ENC_R10,
            ENC_R11, ENC_R12, ENC_R13, ENC_R14, ENC_R15,
        ])?;
        Ok(Self(Gpr(*gpr)))
    }
}
