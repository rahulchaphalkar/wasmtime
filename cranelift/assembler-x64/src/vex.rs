//! Encoding logic for VEX instructions.

use crate::api::CodeSink;

    /// Set the length of the instruction.
    #[inline(always)]
    pub fn length(mut self, length: VexVectorLength) -> Self {
        self.length = length;
        self
    }

    /// Set the legacy prefix byte of the instruction: None | 66 | F2 | F3. VEX instructions
    /// pack these into the prefix, not as separate bytes.
    #[inline(always)]
    pub fn prefix(mut self, prefix: LegacyPrefixes) -> Self {
        debug_assert!(
            prefix == LegacyPrefixes::None
                || prefix == LegacyPrefixes::_66
                || prefix == LegacyPrefixes::_F2
                || prefix == LegacyPrefixes::_F3
        );

        self.prefix = prefix;
        self
    }

    /// Set the opcode map byte of the instruction: None | 0F | 0F38 | 0F3A. VEX instructions pack
    /// these into the prefix, not as separate bytes.
    #[inline(always)]
    pub fn map(mut self, map: OpcodeMap) -> Self {
        self.map = map;
        self
    }

    /// Set the W bit, denoted by `.W1` or `.W0` in the instruction string.
    /// Typically used to indicate an instruction using 64 bits of an operand (e.g.
    /// 64 bit lanes). EVEX packs this bit in the EVEX prefix; previous encodings used the REX
    /// prefix.
    #[inline(always)]
    pub fn w(mut self, w: bool) -> Self {
        self.w = w;
        self
    }

    /// Set the instruction opcode byte.
    #[inline(always)]
    pub fn opcode(mut self, opcode: u8) -> Self {
        self.opcode = opcode;
        self
    }

    /// Set the register to use for the `reg` bits; many instructions use this as the write operand.
    #[inline(always)]
    pub fn reg(mut self, reg: impl Into<Register>) -> Self {
        self.reg = reg.into().into();
        self
    }

    /// Some instructions use the ModRM.reg field as an opcode extension. This is usually denoted by
    /// a `/n` field in the manual.
    #[inline(always)]
    pub fn opcode_ext(mut self, n: u8) -> Self {
        self.reg = n;
        self
    }

    /// Set the register to use for the `rm` bits; many instructions use this
    /// as the "read from register/memory" operand. Setting this affects both
    /// the ModRM byte (`rm` section) and the VEX prefix (the extension bits
    /// for register encodings > 8).
    #[inline(always)]
    pub fn rm(mut self, reg: impl Into<RegisterOrAmode>) -> Self {
        self.rm = reg.into();
        self
    }

    /// Set the `vvvv` register; some instructions allow using this as a second, non-destructive
    /// source register in 3-operand instructions (e.g. 2 read, 1 write).
    #[allow(dead_code)]
    #[inline(always)]
    pub fn vvvv(mut self, reg: impl Into<Register>) -> Self {
        self.vvvv = Some(reg.into());
        self
    }

    /// Set the imm byte when used for a register. The reg bits are stored in `imm8[7:4]` with
    /// the lower bits unused. Overrides a previously set [Self::imm] field.
    #[inline(always)]
    pub fn imm_reg(mut self, reg: impl Into<Register>) -> Self {
        let reg: u8 = reg.into().into();
        self.imm = Some((reg & 0xf) << 4);
        self
    }

    /// Set the imm byte.
    /// Overrides a previously set [Self::imm_reg] field.
    #[inline(always)]
    pub fn imm(mut self, imm: u8) -> Self {
        self.imm = Some(imm);
        self
    }

    /// The R bit in encoded format (inverted).
    #[inline(always)]
    fn r_bit(&self) -> u8 {
        (!(self.reg >> 3)) & 1
    }

    /// The X bit in encoded format (inverted).
    #[inline(always)]
    fn x_bit(&self) -> u8 {
        let reg = match &self.rm {
            RegisterOrAmode::Register(_) => 0,
            RegisterOrAmode::Amode(Amode::ImmReg { .. }) => 0,
            RegisterOrAmode::Amode(Amode::ImmRegRegShift { index, .. }) => {
                index.to_real_reg().unwrap().hw_enc()
            }
            RegisterOrAmode::Amode(Amode::RipRelative { .. }) => 0,
        };

        !(reg >> 3) & 1
    }

    /// The B bit in encoded format (inverted).
    #[inline(always)]
    fn b_bit(&self) -> u8 {
        let reg = match &self.rm {
            RegisterOrAmode::Register(r) => (*r).into(),
            RegisterOrAmode::Amode(Amode::ImmReg { base, .. }) => {
                base.to_real_reg().unwrap().hw_enc()
            }
            RegisterOrAmode::Amode(Amode::ImmRegRegShift { base, .. }) => {
                base.to_real_reg().unwrap().hw_enc()
            }
            RegisterOrAmode::Amode(Amode::RipRelative { .. }) => 0,
        };

        !(reg >> 3) & 1
    }

    /// Is the 2 byte prefix available for this instruction?
    /// We essentially just check if we need any of the bits that are only available
    /// in the 3 byte instruction
    #[inline(always)]
    fn use_2byte_prefix(&self) -> bool {
        // These bits are only represented on the 3 byte prefix, so their presence
        // implies the use of the 3 byte prefix
        self.b_bit() == 1 && self.x_bit() == 1 &&
        // The presence of W1 in the opcode column implies the opcode must be encoded using the
        // 3-byte form of the VEX prefix.
        self.w == false &&
        // The presence of 0F3A and 0F38 in the opcode column implies that opcode can only be
        // encoded by the three-byte form of VEX
        !(self.map == OpcodeMap::_0F3A || self.map == OpcodeMap::_0F38)
    }

    /// The last byte of the 2byte and 3byte prefixes is mostly the same, share the common
    /// encoding logic here.
    #[inline(always)]
    fn prefix_last_byte(&self) -> u8 {
        let vvvv = self.vvvv.map(|r| r.into()).unwrap_or(0x00);

        let mut byte = 0x00;
        byte |= self.prefix.bits();
        byte |= self.length.bits() << 2;
        byte |= ((!vvvv) & 0xF) << 3;
        byte
    }

    /// Encode the 2 byte prefix
    #[inline(always)]
    fn encode_2byte_prefix<CS: ByteSink + ?Sized>(&self, sink: &mut CS) {
        //  2 bytes:
        //    +-----+ +-------------------+
        //    | C5h | | R | vvvv | L | pp |
        //    +-----+ +-------------------+

        let last_byte = self.prefix_last_byte() | (self.r_bit() << 7);

        sink.put1(0xC5);
        sink.put1(last_byte);
    }

    /// Encode the 3 byte prefix
    #[inline(always)]
    fn encode_3byte_prefix<CS: ByteSink + ?Sized>(&self, sink: &mut CS) {
        //  3 bytes:
        //    +-----+ +--------------+ +-------------------+
        //    | C4h | | RXB | m-mmmm | | W | vvvv | L | pp |
        //    +-----+ +--------------+ +-------------------+

        let mut second_byte = 0x00;
        second_byte |= self.map.bits(); // m-mmmm field
        second_byte |= self.b_bit() << 5;
        second_byte |= self.x_bit() << 6;
        second_byte |= self.r_bit() << 7;

        let w_bit = self.w as u8;
        let last_byte = self.prefix_last_byte() | (w_bit << 7);

        sink.put1(0xC4);
        sink.put1(second_byte);
        sink.put1(last_byte);
    }

    /// Emit the VEX-encoded instruction to the provided buffer.
    pub fn encode(&self, sink: &mut MachBuffer<Inst>) {
        if let RegisterOrAmode::Amode(amode) = &self.rm {
            if let Some(trap_code) = amode.get_flags().trap_code() {
                sink.add_trap(trap_code);
            }
        }

        // 2/3 byte prefix
        if self.use_2byte_prefix() {
            self.encode_2byte_prefix(sink);
        } else {
            self.encode_3byte_prefix(sink);
        }

        // 1 Byte Opcode
        sink.put1(self.opcode);

        match &self.rm {
            // Not all instructions use Reg as a reg, some use it as an extension
            // of the opcode.
            RegisterOrAmode::Register(reg) => {
                let rm: u8 = (*reg).into();
                sink.put1(rex::encode_modrm(3, self.reg & 7, rm & 7));
            }
            // For address-based modes reuse the logic from the `rex` module
            // for the modrm and trailing bytes since VEX uses the same
            // encoding.
            RegisterOrAmode::Amode(amode) => {
                let bytes_at_end = if self.imm.is_some() { 1 } else { 0 };
                rex::emit_modrm_sib_disp(sink, self.reg & 7, amode, bytes_at_end, None);
            }
        }

        // Optional 1 Byte imm
        if let Some(imm) = self.imm {
            sink.put1(imm);
        }
    }
