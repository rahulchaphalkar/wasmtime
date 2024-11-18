use crate::dsl::{fmt, inst, r, rex, rw, sx, Features::*, Inst, LegacyPrefixes::*, Location::*};

pub fn list() -> Vec<Inst> {
    vec![
        inst("andb", fmt("I", [rw(al), r(imm8)]), rex(0x24).ib(), None),
        inst("andw", fmt("I", [rw(ax), r(imm16)]), rex(0x25).prefix(_66).iw(), None),
        inst("andl", fmt("I", [rw(eax), r(imm32)]), rex(0x25).id(), None),
        // inst("andq", fmt("I_SX", [rw(rax), r(imm32)]), rex(0x25).w().id(), None), // TODO: need a way to sign-extend the imm32, e.g., idsx()?
        inst("andb", fmt("MI", [rw(rm8), r(imm8)]), rex(0x80).digit(4).ib(), None),
        // inst("andb", fmt("MI_SX", [rw(rm8), r(imm8)]), rex(0x80).w().digit(4).ib(), None),
        inst("andw", fmt("MI", [rw(rm16), r(imm16)]), rex(0x81).prefix(_66).digit(4).iw(), None),
        inst("andl", fmt("MI", [rw(rm32), r(imm32)]), rex(0x81).digit(4).id(), None),
        inst("andq", fmt("MI_SX", [rw(rm64), sx(imm32)]), rex(0x81).w().digit(4).id(), None),
        // inst("andbw", fmt("MI_SX", [rw(rm16), r(imm8)]), rex(0x83).digit(4).ib(), None),
        // inst("andbd", fmt("MI_SX", [rw(rm32), r(imm8)]), rex(0x83).digit(4).ib(), None),
        // inst("andbq", fmt("MI_SX", [rw(rm64), r(imm8)]), rex(0x83).w().digit(4).ib(), None),
        inst("andb", fmt("MR", [rw(rm8), r(r8)]), rex(0x20).r(), None),
        // inst("andb", fmt("MR_SX", [rw(rm8), r(r8)]), rex(0x20).w().r(), None),
        inst("andw", fmt("MR", [rw(rm16), r(r16)]), rex(0x21).prefix(_66).r(), None),
        inst("andl", fmt("MR", [rw(rm32), r(r32)]), rex(0x21).r(), None),
        inst("andq", fmt("MR", [rw(rm64), r(r64)]), rex(0x21).w().r(), None),
        inst("andb", fmt("RM", [rw(r8), r(rm8)]), rex(0x22).r(), None),
        // inst("andb", fmt("RM_SX", [rw(r8), r(rm8)]), rex(0x22).w().r(), None),
        inst("andw", fmt("RM", [rw(r16), r(rm16)]), rex(0x23).prefix(_66).r(), None),
        inst("andl", fmt("RM", [rw(r32), r(rm32)]), rex(0x23).r(), None),
        inst("andq", fmt("RM", [rw(r64), r(rm64)]), rex(0x23).w().r(), None),
    ]
}
