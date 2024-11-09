//! Fuzzing utilities.

use crate::Inst;
use capstone::arch::{BuildsCapstone, BuildsCapstoneSyntax};

/// Generate a random assembly instruction and check its encoding and
/// pretty-printing against a known-good disassembler.
///
/// # Panics
///
/// This function panics to express failure as expected by the `arbitrary`
/// fuzzer infrastructure. It may fail during assembly, disassembly, or when
/// comparing the disassembled strings.
pub fn roundtrip(inst: &Inst) {
    // Check that we can actually assemble this instruction.
    let assembled = assemble(inst);
    let expected = disassemble(&assembled);

    // Check that our pretty-printed output matches the known-good output.
    let expected = expected.split_once(' ').unwrap().1;
    let actual = inst.to_string();
    if expected != actual {
        println!("> {inst}");
        println!("  debug: {inst:x?}");
        println!("  assembled: {}", pretty_print_hexadecimal(&assembled));
        assert_eq!(expected, &actual);
    }
}

/// Use this assembler to emit machine code into a byte buffer.
///
/// This will skip any traps or label registrations, but this is fine for the
/// single-instruction disassembly we're doing here.
fn assemble(insn: &Inst) -> Vec<u8> {
    let mut buffer = Vec::new();
    insn.encode(&mut buffer);
    buffer
}

/// Building a new `Capstone` each time is suboptimal (TODO).
fn disassemble(assembled: &[u8]) -> String {
    let cs = capstone::Capstone::new()
        .x86()
        .mode(capstone::arch::x86::ArchMode::Mode64)
        .syntax(capstone::arch::x86::ArchSyntax::Att)
        .detail(true)
        .build()
        .expect("failed to create Capstone object");
    let insns = cs
        .disasm_all(assembled, 0x0)
        .expect("failed to disassemble");
    assert_eq!(insns.len(), 1, "not a single instruction: {assembled:x?}");
    let insn = insns.first().expect("at least one instruction");
    assert_eq!(assembled.len(), insn.len());
    insn.to_string()
}

fn pretty_print_hexadecimal(hex: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(hex.len() * 2);
    for b in hex {
        write!(&mut s, "{b:02X}").unwrap();
    }
    s
}

#[cfg(test)]
mod test {
    use super::*;
    use arbtest::arbtest;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn smoke() {
        let count = AtomicUsize::new(0);
        arbtest(|u| {
            let inst: Inst = u.arbitrary()?;
            roundtrip(&inst);
            println!("#{}: {inst}", count.fetch_add(1, Ordering::SeqCst));
            Ok(())
        })
        .budget_ms(1_000);

        // This will run the `roundtrip` fuzzer for one second. To repeatably
        // test a single input, append `.seed(0x<failing seed>)`.
    }
}
