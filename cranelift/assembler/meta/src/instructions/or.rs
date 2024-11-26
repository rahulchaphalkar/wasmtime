use crate::dsl::{fmt, inst, r, rex, rw, sx, Features::*, Inst, LegacyPrefixes::*, Location::*};

pub fn list() -> Vec<Inst> {
    vec![
        inst("orb", fmt("I", [rw(al), r(imm8)]), rex(0x0C).ib(), None),
        inst("orw", fmt("I", [rw(ax), r(imm16)]), rex(0x0D).prefix(_66).iw(), None),
        inst("orl", fmt("I", [rw(eax), r(imm32)]), rex(0x0D).id(), None),
        inst("orq", fmt("I_SX", [rw(rax), sx(imm32)]), rex(0x0D).w().id(), None),
        inst("orb", fmt("MI", [rw(rm8), r(imm8)]), rex(0x80).digit(1).ib(), None),
    ]
}