"""The few operating-system facts the tool needs, for Linux and Windows alike: free RAM, whether the card is busy,
a free port, and starting/stopping an engine as its own process group."""

from __future__ import annotations

import ctypes
import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path

GIB = 1024 ** 3


def available_ram_bytes() -> int:
    """What the OS would hand a new program now (Linux MemAvailable; Windows ullAvailPhys)."""
    if sys.platform.startswith("linux"):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
        raise RuntimeError("/proc/meminfo has no MemAvailable")
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            raise RuntimeError("GlobalMemoryStatusEx failed")
        return int(status.ullAvailPhys)
    raise RuntimeError(f"free RAM is not implemented for {sys.platform}")


def free_disk_bytes(path: Path) -> int:
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def card_processes(device: int = 0) -> list[tuple[int, int]]:
    """(pid, MiB) of every compute process on the card, from nvidia-smi (on PATH on Linux and Windows)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        raise RuntimeError("nvidia-smi is not on PATH; it ships with the NVIDIA driver")
    uuid = subprocess.run([exe, "-i", str(device), "--query-gpu=uuid", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    out = subprocess.run([exe, "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30, check=True).stdout
    procs = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0] == uuid:
            procs.append((int(parts[1]), int(parts[2]) if parts[2].isdigit() else 0))
    return procs


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start(cmd: list[str], log_path: Path, env: dict | None = None) -> subprocess.Popen:
    """Start a process in its own group (so stop() reaches its children too), output to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    log.write("CMD " + subprocess.list2cmdline(cmd) + "\n")
    log.flush()
    kwargs: dict = {"stdout": log, "stderr": subprocess.STDOUT, "env": env}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    proc._ablate_log = log  # type: ignore[attr-defined]
    return proc


def stop(proc: subprocess.Popen | None, timeout: float = 60.0) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
            proc.kill()
            proc.wait(timeout=timeout)
    log = getattr(proc, "_ablate_log", None)
    if log is not None:
        log.close()
