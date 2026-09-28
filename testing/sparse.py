"""Sparse-by-construction file sizes for tests.

Tests need multi-GB model files without the bytes.  On ext4/xfs a
``truncate()`` produces a sparse file that costs no blocks; on NTFS it
allocates, and a hosted windows runner with ~30 GB free then dies with
ENOSPC mid-suite (the 2026-09-26 windows CI failures: the unit tests alone
materialize ~50 GB of "sparse" files).

``sized_file`` flags the file FILE_ATTRIBUTE_SPARSE_FILE *before* extending
it, so ``truncate`` only updates metadata on windows too.  The flagging is
best-effort: if the filesystem refuses, the caller still gets the sized file
and the test behaves as before.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

_FSCTL_SET_SPARSE = 0x000900C4
_FSCTL_SET_ZERO_DATA = 0x000980C8
_GENERIC_WRITE = 0x40000000
_OPEN_EXISTING = 3
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_INVALID_HANDLE = -1  # windows INVALID_HANDLE_VALUE, as an unsigned pointer


def _open_for_write(path: Path):
    """A kernel32 handle for ``path`` (``None`` on failure), windows only."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
    ]
    handle = kernel32.CreateFileW(
        str(path), _GENERIC_WRITE, _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        None, _OPEN_EXISTING, 0, None,
    )
    if not handle or handle == ctypes.c_void_p(_INVALID_HANDLE).value:
        return None, kernel32
    return handle, kernel32


def mark_sparse(path: Path) -> bool:
    """Best-effort NTFS sparse flag on ``path``.  No-op/True off windows."""
    if os.name != "nt":
        return True

    import ctypes
    from ctypes import wintypes

    handle, kernel32 = _open_for_write(path)
    if handle is None:
        print(
            f"testing.sparse: CreateFileW failed on {path!r}; the sized file "
            "will allocate real blocks",
            file=sys.stderr,
        )
        return False
    kernel32.DeviceIoControl.restype = wintypes.BOOL
    kernel32.DeviceIoControl.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
        ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    try:
        returned = wintypes.DWORD(0)
        ok = kernel32.DeviceIoControl(
            handle, _FSCTL_SET_SPARSE, None, 0, None, 0,
            ctypes.byref(returned), None,
        )
        if not ok:
            print(
                f"testing.sparse: FSCTL_SET_SPARSE failed on {path!r} "
                f"(winerror {ctypes.get_last_error()}); the sized file will "
                "allocate real blocks",
                file=sys.stderr,
            )
        return bool(ok)
    finally:
        kernel32.CloseHandle(handle)


def deallocate(path: Path, start: int, end: int) -> bool:
    """Free the clusters of a sparse file's ``[start, end)`` range (windows).

    ``truncate`` on NTFS reserves the extended range even when the file has
    the sparse attribute; ``FSCTL_SET_ZERO_DATA`` then marks it unallocated.
    Only the *extension* may be zeroed: the range before ``start`` is real
    content the caller wrote (a GGUF header), and zeroing it would wipe it.
    """
    if os.name != "nt" or end <= start:
        return True

    import ctypes
    from ctypes import wintypes

    class ZeroData(ctypes.Structure):
        _fields_ = [("FileOffset", ctypes.c_longlong),
                    ("BeyondFinalZero", ctypes.c_longlong)]

    handle, kernel32 = _open_for_write(path)
    if handle is None:
        return False
    kernel32.DeviceIoControl.restype = wintypes.BOOL
    kernel32.DeviceIoControl.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
        ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    info = ZeroData(start, end)
    try:
        returned = wintypes.DWORD(0)
        return bool(kernel32.DeviceIoControl(
            handle, _FSCTL_SET_ZERO_DATA,
            ctypes.byref(info), ctypes.sizeof(info),
            None, 0, ctypes.byref(returned), None,
        ))
    finally:
        kernel32.CloseHandle(handle)


def sized_file(path: Path, gib: float) -> Path:
    """Create ``path`` with size ``gib`` GiB, sparse where the fs supports it.

    Existing bytes (a GGUF header the caller just wrote) are preserved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    original = path.stat().st_size
    mark_sparse(path)
    size = int(gib * 1024 ** 3)
    if size > original:
        with path.open("r+b") as handle:
            handle.truncate(size)
        deallocate(path, original, size)
    return path
