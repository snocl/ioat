//! The `ioat` crate provides trait for atomic, random-access I/O access.

use std::fs::File;
use std::cmp;
use std::io::{Empty, Error, ErrorKind, Read, Repeat, Result, Seek, SeekFrom, Sink, Write};

/// The `ReadAt` trait allows for atomically reading bytes from a source at specific offsets.
///
/// As an example, this trait is implemented by `File`. Unlike `Read`,
/// however, it is not implemented for `&File`, since the implementation
/// is not thread-safe. If two threads where to call `read_at` at once,
/// a race condition could occur, where one thread seeks and the other
/// thread reads.
///
/// For this reason, `ReadAt` is not automatically implemented for all
/// `Read + Seek` types. If a `Read + Seek` type is guaranteed to not
/// seek in parallel with a call to `read_at`, it can be wrapped in
/// [`AssertThreadSafe`](struct.AssertThreadSafe.html).
pub trait ReadAt {
    /// Reads some bytes from `pos` bytes into the source.
    ///
    /// This method returns the number of bytes read or an error, if the
    /// bytes could not be read. If `Ok(n)` is returned, then it is
    /// guaranteed, that `0 <= n <= buf.len()`.
    ///
    /// # Errors
    ///
    /// This method can return any I/O error.
    fn read_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<usize>;

    /// Reads exactly `buf.len()` bytes from `pos` bytes into the source.
    ///
    /// This is usually handled by repeatedly calling `read_at` until
    /// the buffer is full. This method is an analogue to `read_exact`
    /// for `Read`.
    ///
    /// # Errors
    ///
    /// If `read_at()` returns an error, then this method immediately
    /// propagates that error by returning.
    ///
    /// If `read_at()` returns `Ok(0)` to indicate the end of the source
    /// has been reached before `buf` has been filled, then this method
    /// errs with an error of kind `UnexpectedEof`. Any bytes read up
    /// until this point are discarded.
    fn read_exact_at(&mut self, mut pos: u64, mut buf: &mut [u8]) -> Result<()> {
        while !buf.is_empty() {
            match self.read_at(pos, buf) {
                Ok(0) => break,
                Ok(n) => {
                    let tmp = buf;
                    buf = &mut tmp[n..];
                    pos += n as u64;
                }
                Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        if !buf.is_empty() {
            Err(Error::new(ErrorKind::UnexpectedEof, "failed to fill whole buffer"))
        } else {
            Ok(())
        }
    }
}

/// The `WriteAt` trait allows for atomically writing bytes to a sink at specific offsets.
///
/// As an example, this trait is implemented by `File`. Similarly to
/// [`ReadAt`](trait.ReadAt.html) it is not implemented for `&File`,
/// or other `Write + Seek` types in general, as this would not be
/// thread-safe.
///
/// If it can be guaranteed, that a `Write + Seek` value will not seek
/// in parallel to a call to `write_at`, it can be wrapped in
/// [`AssertThreadSafe`](struct.AssertThreadSafe.html) to implement this
/// trait.
pub trait WriteAt {
    /// Writes some bytes at `pos` bytes into `self`.
    ///
    /// This method returns the number of bytes written. If `Ok(n)` is
    /// returned, then it is guaranteed, that `0 <= n <= buf.len()`.
    ///
    /// # Errors
    ///
    /// This method can return any I/O error.
    fn write_at(&mut self, pos: u64, buf: &[u8]) -> Result<usize>;

    /// Flushes any pending writes to the underlying sink.
    ///
    /// # Errors
    ///
    /// This method can return any I/O error.
    fn flush(&mut self) -> Result<()>;

    /// Writes exactly `buf.len()` bytes at `pos` bytes into `self`.
    ///
    /// This is usually handled by repeatedly calling `write_at` until
    /// the entire buffer has been written. This method is an analogue
    /// to `write_all` for `Write`.
    ///
    /// # Errors
    ///
    /// If `write_at` returns an error, then this method immediately
    /// propagates it by returning.
    ///
    /// If `write_at` returns `Ok(0)`, indicating that no more bytes
    /// could be written, then this method returns an error of kind
    /// `WriteZero`.
    fn write_all_at(&mut self, mut pos: u64, mut buf: &[u8]) -> Result<()> {
        while !buf.is_empty() {
            match self.write_at(pos, buf) {
                Ok(0) => {
                    return Err(Error::new(ErrorKind::WriteZero, "failed to write whole buffer"));
                }
                Ok(n) => {
                    buf = &buf[n..];
                    pos += n as u64;
                }
                Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

impl<'a, R: ReadAt> ReadAt for &'a mut R {
    #[inline]
    fn read_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<usize> {
        (**self).read_at(pos, buf)
    }

    #[inline]
    fn read_exact_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<()> {
        (**self).read_exact_at(pos, buf)
    }
}

impl<'a, W: WriteAt> WriteAt for &'a mut W {
    #[inline]
    fn write_at(&mut self, pos: u64, buf: &[u8]) -> Result<usize> {
        (**self).write_at(pos, buf)
    }

    #[inline]
    fn write_all_at(&mut self, pos: u64, buf: &[u8]) -> Result<()> {
        (**self).write_all_at(pos, buf)
    }

    #[inline]
    fn flush(&mut self) -> Result<()> {
        (**self).flush()
    }
}

// For now this only supports `[u8]`, `Vec<u8>` and `Box<[u8]>`, which
// is the same as `Cursor` supports for `Write`. A wrapper struct with
// a blanket implementation would be possible for any `AsRef<[u8]>`, but
// that can also be handled downstream. A blanket directly on
// `AsRef<[u8]>` is not possible, since that would conflict with the
// concrete implementations.

impl<'a> ReadAt for &'a [u8] {
    fn read_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<usize> {
        if pos >= self.len() as u64 {
            return Ok(0);
        }
        let i = pos as usize;
        buf.copy_from_slice(&self[i..]);
        Ok(cmp::min(self.len() - i, buf.len()))
    }

    fn read_exact_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<()> {
        if try!(self.read_at(pos, buf)) < buf.len() {
            Err(Error::new(ErrorKind::UnexpectedEof, "failed to write whole buffer"))
        } else {
            Ok(())
        }
    }
}

impl ReadAt for Vec<u8> {
    #[inline]
    fn read_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<usize> {
        (&self[..]).read_at(pos, buf)
    }

    #[inline]
    fn read_exact_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<()> {
        (&self[..]).read_exact_at(pos, buf)
    }
}

impl ReadAt for Box<[u8]> {
    #[inline]
    fn read_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<usize> {
        (&self[..]).read_at(pos, buf)
    }

    #[inline]
    fn read_exact_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<()> {
        (&self[..]).read_exact_at(pos, buf)
    }
}

impl ReadAt for File {
    #[inline]
    fn read_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<usize> {
        AssertThreadSafe(self).read_at(pos, buf)
    }

    #[inline]
    fn read_exact_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<()> {
        AssertThreadSafe(self).read_exact_at(pos, buf)
    }
}

impl ReadAt for Empty {
    #[inline]
    fn read_at(&mut self, _pos: u64, _buf: &mut [u8]) -> Result<usize> {
        Ok(0)
    }
}

impl ReadAt for Repeat {
    #[inline]
    fn read_at(&mut self, _pos: u64, buf: &mut [u8]) -> Result<usize> {
        self.read(buf)
    }
}

impl WriteAt for [u8] {
    fn write_at(&mut self, pos: u64, buf: &[u8]) -> Result<usize> {
        if pos >= self.len() as u64 {
            return Ok(0);
        }
        let i = pos as usize;
        self[i..].copy_from_slice(buf);
        Ok(cmp::min(self.len() - i, buf.len()))
    }

    #[inline]
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn write_all_at(&mut self, pos: u64, buf: &[u8]) -> Result<()> {
        if try!(self.write_at(pos, buf)) < buf.len() {
            Err(Error::new(ErrorKind::UnexpectedEof, "failed to write whole buffer"))
        } else {
            Ok(())
        }
    }
}

impl WriteAt for Vec<u8> {
    fn write_at(&mut self, pos: u64, buf: &[u8]) -> Result<usize> {
        if pos >= usize::max_value() as u64 {
            return Ok(0);
        }
        let i = pos as usize;
        if i >= self.len() {
            let needed = self.len() - i;
            self.reserve(needed);
        }
        self[i..].copy_from_slice(buf);
        Ok(cmp::min(self.len() - i, buf.len()))
    }

    #[inline]
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn write_all_at(&mut self, pos: u64, buf: &[u8]) -> Result<()> {
        if try!(self.write_at(pos, buf)) < buf.len() {
            Err(Error::new(ErrorKind::UnexpectedEof, "failed to write whole buffer"))
        } else {
            Ok(())
        }
    }
}

impl WriteAt for Box<[u8]> {
    #[inline]
    fn write_at(&mut self, pos: u64, buf: &[u8]) -> Result<usize> {
        (&mut self[..]).write_at(pos, buf)
    }

    #[inline]
    fn write_all_at(&mut self, pos: u64, buf: &[u8]) -> Result<()> {
        (&mut self[..]).write_all_at(pos, buf)
    }

    #[inline]
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

impl WriteAt for File {
    #[inline]
    fn write_at(&mut self, pos: u64, buf: &[u8]) -> Result<usize> {
        AssertThreadSafe(self).write_at(pos, buf)
    }

    #[inline]
    fn flush(&mut self) -> Result<()> {
        AssertThreadSafe(self).flush()
    }

    #[inline]
    fn write_all_at(&mut self, pos: u64, buf: &[u8]) -> Result<()> {
        AssertThreadSafe(self).write_all_at(pos, buf)
    }
}

impl WriteAt for Sink {
    #[inline]
    fn write_at(&mut self, _pos: u64, buf: &[u8]) -> Result<usize> {
        Ok(buf.len())
    }

    #[inline]
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// A struct for wrapping thread-safe readers or writers.
///
/// Using this struct asserts, that the contained `Read + Seek` or
/// `Write + Seek` value can be used for [`ReadAt`](trait.ReadAt.html)
/// and [`WriteAt`](trait.WriteAt.html), without risking a seek from
/// another thread in parallel.
///
/// The traits are implemented by first seeking to the offset, and then
/// calling the appropriate method on the wrapped value.
#[derive(Clone, Debug)]
pub struct AssertThreadSafe<T>(pub T);

impl<T> ReadAt for AssertThreadSafe<T>
    where T: Read + Seek
{
    #[inline]
    fn read_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<usize> {
        try!(self.0.seek(SeekFrom::Start(pos)));
        self.0.read(buf)
    }

    #[inline]
    fn read_exact_at(&mut self, pos: u64, buf: &mut [u8]) -> Result<()> {
        try!(self.0.seek(SeekFrom::Start(pos)));
        self.0.read_exact(buf)
    }
}

impl<T> WriteAt for AssertThreadSafe<T>
    where T: Write + Seek
{
    #[inline]
    fn write_at(&mut self, pos: u64, buf: &[u8]) -> Result<usize> {
        try!(self.0.seek(SeekFrom::Start(pos)));
        self.0.write(buf)
    }

    #[inline]
    fn flush(&mut self) -> Result<()> {
        self.0.flush()
    }

    #[inline]
    fn write_all_at(&mut self, pos: u64, buf: &[u8]) -> Result<()> {
        try!(self.0.seek(SeekFrom::Start(pos)));
        self.0.write_all(buf)
    }
}
