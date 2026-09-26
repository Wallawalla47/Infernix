"""
A/B watchdog: waits for the production server (default 127.0.0.1:8080) to be closed,
then runs the A/B benchmark (ab_runner.py) and reports.

Launch it as a long-lived background process. It polls the port every 5 s; once the
port stops accepting connections for 4 consecutive checks (~20 s), it invokes
ab_runner.py (which serves on the same port for the benchmark) and waits for it to
finish. The watchdog fires once per production close window; it never starts the
production server itself — the user restarts it manually, after which the watchdog
re-arms. Same AB_HOST/AB_PORT environment overrides as ab_runner.py.
"""
import os
import socket
import subprocess
import sys
import time
import datetime

AB_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOY = r"E:\NInfer-Deploy-V3"
HOST = os.environ.get("AB_HOST", "127.0.0.1")
PORT = int(os.environ.get("AB_PORT", "8080"))
POLL_S = 5
CLOSED_CONFIRM = 4          # consecutive closed polls before triggering (~20 s; 2026-09-20
                            # logs showed a ~15 s transient :8080 close, so keep the margin)
RUNNER_TIMEOUT_S = 3 * 3600 # 3 h safety cap on the A/B run
LOG = os.path.join(AB_DIR, "watchdog.log")
DONE = os.path.join(AB_DIR, "AB_DONE.txt")
RUNNER = os.path.join(AB_DIR, "ab_runner.py")


def log(msg):
    line = "[%s] %s" % (datetime.datetime.now().isoformat(timespec="seconds"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def port_open(host, port, timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_manual_restart():
    """The runner does NOT restart production (the user does, manually). Wait for the user's
    restart before re-arming so a failed run doesn't loop re-fires while prod stays down."""
    log("waiting for the user to restart production :8080 before re-arming "
        "(the watchdog never starts the production server itself)")
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        if port_open(HOST, PORT):
            log("production :8080 is back (manual restart); re-arming")
            return True
        time.sleep(10)
    log("production did not return within 6h; stopping watchdog")
    return False


def main():
    if os.path.exists(DONE):
        log("AB_DONE.txt already present — A/B appears to have run; not re-triggering")
        return
    log("watchdog started; polling %s:%d every %ds" % (HOST, PORT, POLL_S))
    while True:
        closed_streak = 0
        while True:
            if port_open(HOST, PORT):
                closed_streak = 0
            else:
                closed_streak += 1
                if closed_streak >= CLOSED_CONFIRM:
                    log("port %d closed %dx in a row -> production stopped; launching A/B runner"
                        % (PORT, closed_streak))
                    break
                log("port %d closed (streak %d/%d)" % (PORT, closed_streak, CLOSED_CONFIRM))
            time.sleep(POLL_S)

        t0 = time.time()
        log("launching: %s %s" % (sys.executable, RUNNER))
        try:
            proc = subprocess.run([sys.executable, RUNNER], cwd=DEPLOY, timeout=RUNNER_TIMEOUT_S)
            log("runner exited code=%s after %.0fs" % (proc.returncode, time.time() - t0))
        except Exception as e:
            log("runner raised: %r after %.0fs" % (e, time.time() - t0))

        if os.path.exists(DONE):
            log("AB_DONE.txt present — A/B complete; stopping watchdog")
            return

        # Failure: production stays down until the user restarts it manually (never here).
        # Wait for that restart before re-arming to avoid re-firing into a down window.
        if not wait_for_manual_restart():
            return
        log("re-armed: waiting for the next :8080 close window")


if __name__ == "__main__":
    main()
