//! The native TPTC container ("TPT Columnar v1").
//!
//! Layout (all integers little-endian):
//!
//! ```text
//! magic  : b"TPTC1\n"                       (6 bytes)
//! batch* : schema | num_rows | column data   (repeated until EOF)
//! ```
//!
//! Each batch embeds its own schema so files can be appended and streamed.
//! This format is original to this workspace and carries no third-party
//! intellectual property.

use std::io::{Read, Write};

use crate::datatypes::SchemaRef;
use crate::error::ColumnarError;
use crate::record_batch::RecordBatch;

/// Magic bytes at offset 0 of every TPTC file/stream.
pub const MAGIC: &[u8; 6] = b"TPTC1\n";

fn encode_batch(batch: &RecordBatch) -> Vec<u8> {
    let mut out = Vec::new();
    batch.encode_into(&mut out);
    out
}

/// Streaming writer producing TPTC bytes into any [`Write`] sink.
pub struct FileWriter<W: Write> {
    sink: W,
    finished: bool,
}

impl<W: Write> FileWriter<W> {
    /// Write the magic header and bind to a schema (recorded with the first
    /// batch, matching the self-describing layout).
    pub fn try_new(mut sink: W, _schema: &SchemaRef) -> Result<Self, ColumnarError> {
        sink.write_all(MAGIC)?;
        Ok(Self {
            sink,
            finished: false,
        })
    }

    /// Append one record batch.
    pub fn write(&mut self, batch: &RecordBatch) -> Result<(), ColumnarError> {
        let enc = encode_batch(batch);
        self.sink.write_all(&(enc.len() as u32).to_le_bytes())?;
        self.sink.write_all(&enc)?;
        Ok(())
    }

    /// Flush and finalize the stream.
    pub fn finish(mut self) -> Result<(), ColumnarError> {
        self.sink.flush()?;
        self.finished = true;
        Ok(())
    }
}

impl<W: Write> Drop for FileWriter<W> {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.sink.flush();
        }
    }
}

/// Streaming reader over TPTC bytes from any [`Read`] source. Yields one
/// [`RecordBatch`] per stored batch; iteration ends cleanly at EOF.
pub struct FileReader<R: Read> {
    src: R,
}

impl<R: Read> Iterator for FileReader<R> {
    type Item = Result<RecordBatch, ColumnarError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut len_buf = [0u8; 4];
        match self.src.read_exact(&mut len_buf) {
            Ok(()) => {}
            // Clean EOF between batches.
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return None,
            Err(e) => return Some(Err(e.into())),
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        if let Err(e) = self.src.read_exact(&mut buf) {
            return Some(Err(e.into()));
        }
        let mut pos = 0usize;
        match RecordBatch::decode(&buf, &mut pos) {
            Ok(b) => Some(Ok(b)),
            Err(e) => Some(Err(e)),
        }
    }
}

impl<R: Read> FileReader<R> {
    /// Verify the magic header and prepare to read batches.
    pub fn try_new(mut src: R) -> Result<Self, ColumnarError> {
        let mut magic = [0u8; 6];
        src.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(ColumnarError::ParseError(
                "not a TPTC container (bad magic)".into(),
            ));
        }
        Ok(Self { src })
    }
}
