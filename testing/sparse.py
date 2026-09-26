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
_GENERIC_WRITE = 0x40000000
_OPEN_EXISTING = 3
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_INVALID_HANDLE = -1  # windows INVALID_HANDLE_VALUE, as an unsigned pointer


def mark_sparse(path: Path) -> bool:
    """Best-effort NTFS sparse flag on ``path``.  No-op/True off windows."""
    if os.name != "nt":
        return True

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
    ]
    kernel32.DeviceIoControl.restype = wintypes.BOOL
    kernel32.DeviceIoControl.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
        ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    handle = kernel32.CreateFileW(
        str(path), _GENERIC_WRITE, _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        None, _OPEN_EXISTING, 0, None,
    )
    if not handle or handle == ctypes.c_void_p(_INVALID_HANDLE).value:
        print(
            f"testing.sparse: CreateFileW failed on {path!r}; the sized file "
            "will allocate real blocks",
            file=sys.stderr,
        )
        return False
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


def sized_file(path: Path, gib: float) -> Path:
    """Create ``path`` with size ``gib`` GiB, sparse where the fs supports it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    mark_sparse(path)
    with path.open("r+b") as handle:
        handle.truncate(int(gib * 1024 ** 3))
    return path
