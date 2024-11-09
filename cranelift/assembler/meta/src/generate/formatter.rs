//! Borrowed extensively from `cranelift/codegen/meta/src/srcgen.rs`.

use std::fs;
use std::io::{self, Write};

static SHIFTWIDTH: usize = 4;

/// A macro that simplifies the usage of the Formatter by allowing format
/// strings.
macro_rules! fmtln {
    ($fmt:ident, $fmtstring:expr, $($fmtargs:expr),*) => {
        let loc = crate::generate::maybe_file_loc($fmtstring, file!(), line!());
        $fmt.line(format!($fmtstring, $($fmtargs),*), loc)
    };

    ($fmt:ident, $arg:expr) => {{
        let loc = crate::generate::maybe_file_loc($arg, file!(), line!());
        $fmt.line(format!($arg), loc)
    }};

    ($_:tt, $($args:expr),+) => {
        compile_error!("This macro requires at least two arguments: the Formatter instance and a format string.")
    };

    ($_:tt) => {
        compile_error!("This macro requires at least two arguments: the Formatter instance and a format string.")
    };
}
pub(crate) use fmtln;

use super::FileLocation;

#[derive(Default)]
pub struct Formatter {
    indent: usize,
    lines: Vec<String>,
}

impl Formatter {
    /// Source code formatter class. Used to collect source code to be written
    /// to a file, and keep track of indentation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Increase current indentation level by one.
    pub fn indent_push(&mut self) {
        self.indent += 1;
    }

    /// Decrease indentation by one level.
    pub fn indent_pop(&mut self) {
        assert!(self.indent > 0, "Already at top level indentation");
        self.indent -= 1;
    }

    pub fn indent<T, F: FnOnce(&mut Formatter) -> T>(&mut self, f: F) -> T {
        self.indent_push();
        let ret = f(self);
        self.indent_pop();
        ret
    }

    /// Get the current whitespace indentation in the form of a String.
    fn get_indent(&self) -> String {
        if self.indent == 0 {
            String::new()
        } else {
            format!("{:-1$}", " ", self.indent * SHIFTWIDTH)
        }
    }

    /// Add an indented line.
    pub fn line(&mut self, contents: impl AsRef<str>, location: Option<FileLocation>) {
        let indented_line = if let Some(location) = location {
            format!("{} {} // {location}\n", self.get_indent(), contents.as_ref())
        } else {
            format!("{}{}\n", self.get_indent(), contents.as_ref())
        };
        self.lines.push(indented_line);
    }

    /// Pushes an empty line.
    pub fn empty_line(&mut self) {
        self.lines.push("\n".to_string());
    }

    /// Add a comment line.
    pub fn comment(&mut self, s: impl AsRef<str>) {
        self.line(format!("// {}", s.as_ref()), None);
    }

    /// Write `self.lines` to a file.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> io::Result<()> {
        let mut f = fs::File::create(path)?;
        for l in self.lines.iter().map(String::as_bytes) {
            f.write_all(l)?;
        }
        Ok(())
    }
}

/// Compute the indentation of s, or None of an empty line.
fn _indent(s: &str) -> Option<usize> {
    if s.is_empty() {
        None
    } else {
        let t = s.trim_start();
        Some(s.len() - t.len())
    }
}
