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


# Another program holds the card when one of its compute processes holds this much of it, or (Windows' WDDM driver
# reports no memory per process) when this much of the card is in use: a desktop's shell, compositor and browsers are
# listed as compute processes there while holding a few hundred MiB.
BUSY_PROCESS_MIB = 1024
BUSY_CARD_MIB = 4096


def card_busy(device: int = 0, run=subprocess.run) -> str | None:
    """Why another program holds the card the tool's engines need whole (production still serving, say), or None.
    From nvidia-smi (on PATH on Linux and Windows); the engines' own --plan checks whether they fit what is free."""
    exe = shutil.which("nvidia-smi") or "nvidia-smi"

    def query(*args: str) -> str:
        try:
            return run([exe, *args], capture_output=True, text=True, timeout=30, check=True).stdout
        except FileNotFoundError:
            raise RuntimeError("nvidia-smi is not on PATH; it ships with the NVIDIA driver") from None

    card = [p.strip() for p in query("-i", str(device), "--query-gpu=uuid,memory.used",
                                     "--format=csv,noheader,nounits").strip().split(",")]
    uuid, used = card[0], int(card[1]) if len(card) > 1 and card[1].isdigit() else 0
    procs = []   # (pid, program, MiB or None where the driver does not say)
    for line in query("--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                      "--format=csv,noheader,nounits").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[0] == uuid:
            procs.append((int(parts[1]), Path(",".join(parts[2:-1])).name, int(parts[-1]) if parts[-1].isdigit() else None))
    holders = [p for p in procs if p[2] is not None and p[2] >= BUSY_PROCESS_MIB]
    if holders:
        return "held by " + ", ".join(f"{name} (process {pid}, {mib} MiB)" for pid, name, mib in holders)
    if used >= BUSY_CARD_MIB and not any(p[2] is not None for p in procs):
        names = ", ".join(f"{name} ({pid})" for pid, name, _ in procs) or "no process listed"
        return f"{used} MiB of the card in use (compute processes: {names})"
    return None


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
