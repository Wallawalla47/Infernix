"""Keep large offline transfers from retaining whole artifacts in Linux's page cache.

On non-POSIX platforms (Windows) the page-cache hints are unavailable, and the
absolute pread/pwrite primitives fall back to seek-based reads and writes. This
module is imported by every reader/writer/source before it touches a descriptor,
so the shims below are in place wherever they are needed.
"""

from __future__ import annotations

import os

IO_CHUNK_BYTES = 8 * 1024 * 1024
WRITEBACK_BYTES = 64 * 1024 * 1024
try:
    _PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")
except (AttributeError, ValueError):
    _PAGE_BYTES = 4096


def _install_pread_pwrite() -> None:
    if hasattr(os, "pread") and hasattr(os, "pwrite"):
        return

    # On Windows a bare os.lseek/os.read on a descriptor from os.open does not
    # honour the seek position (it returns a short read), so transfer through a
    # buffered file object on a duplicate descriptor instead. The duplicate is
    # owned and closed by the context manager; the original descriptor is left
    # untouched.
    def pread(fd: int, length: int, offset: int) -> bytes:
        dup = os.dup(fd)
        with os.fdopen(dup, "rb", buffering=IO_CHUNK_BYTES) as handle:
            handle.seek(offset)
            return handle.read(length)

    def pwrite(fd: int, data, offset: int) -> int:
        dup = os.dup(fd)
        with os.fdopen(dup, "wb", buffering=IO_CHUNK_BYTES) as handle:
            handle.seek(offset)
            view = memoryview(data)
            written = 0
            while written < len(view):
                written += handle.write(view[written:])
            return written

    if not hasattr(os, "pread"):
        os.pread = pread
    if not hasattr(os, "pwrite"):
        os.pwrite = pwrite


_install_pread_pwrite()


def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
    if not hasattr(os, "posix_fadvise"):
        return
    if count is None:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    elif count > 0:
        begin = offset // _PAGE_BYTES * _PAGE_BYTES
        end = (offset + count + _PAGE_BYTES - 1) // _PAGE_BYTES * _PAGE_BYTES
        os.posix_fadvise(fd, begin, end - begin, os.POSIX_FADV_DONTNEED)


class Writeback:
    """Bound dirty output across all open shards; release clean pages after writeback."""

    def __init__(self) -> None:
        self._bytes = 0
        self._fds: set[int] = set()

    def written(self, fd: int, count: int) -> None:
        self._fds.add(fd)
        self._bytes += count
        if self._bytes >= WRITEBACK_BYTES:
            self.flush()

    def flush(self) -> None:
        for fd in self._fds:
            (os.fdatasync if hasattr(os, "fdatasync") else os.fsync)(fd)
            discard_cached_pages(fd)
        self._fds.clear()
        self._bytes = 0
