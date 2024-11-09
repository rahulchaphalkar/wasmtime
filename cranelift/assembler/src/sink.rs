//! Contains the encoding machinery for the various x64 instruction formats.
use arbitrary::Arbitrary;
use std::{num::NonZeroU8, vec::Vec};

/// The encoding formats in this module all require a way of placing bytes into
/// a buffer.
pub trait CodeSink {
    /// Add 1 byte to the code section.
    fn put1(&mut self, _: u8);

    /// Add 2 bytes to the code section.
    fn put2(&mut self, _: u16);

    /// Add 4 bytes to the code section.
    fn put4(&mut self, _: u32);

    /// Add 8 bytes to the code section.
    fn put8(&mut self, _: u64);

    /// TODO: I would prefer to keep this away from this trait (but see how
    /// `Amode::RipRelative` is lowered).
    fn cur_offset(&self) -> u32;

    /// TODO: I would prefer to keep this away from this trait (but see how
    /// `Amode::RipRelative` is lowered).
    fn use_label_at_offset<LU>(&mut self, offset: u32, label: Label, kind: LU);

    /// TODO: I would prefer to keep this away from this trait (but see
    /// how we lower `Amode`).
    fn add_trap(&mut self, code: TrapCode);
}

#[derive(Debug, Clone, Arbitrary)]
pub struct Label(pub u32);

#[derive(Debug, Clone, Copy, Arbitrary)]
pub struct TrapCode(pub NonZeroU8);

/// Provide a convenient implementation for testing.
impl CodeSink for Vec<u8> {
    fn put1(&mut self, v: u8) {
        self.extend_from_slice(&[v]);
    }

    fn put2(&mut self, v: u16) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn put4(&mut self, v: u32) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn put8(&mut self, v: u64) {
        self.extend_from_slice(&v.to_le_bytes());
    }

    fn cur_offset(&self) -> u32 {
        self.len().try_into().unwrap()
    }

    fn use_label_at_offset<LU>(&mut self, _: u32, _: Label, _: LU) {}

    fn add_trap(&mut self, _: TrapCode) {}
}
