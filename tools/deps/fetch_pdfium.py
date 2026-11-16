"""Downloads PDFium for PDF documents (src/media/pdf): bblanchon/pdfium-binaries at the pinned release, the build
without V8 and XFA. Checks the archive against its pinned SHA-256, unpacks it and prints the tree to configure with
(PDFIUM_ROOT, cmake/Dependencies.cmake).

    py -3.11 -I tools/deps/fetch_pdfium.py [--platform win-x64|linux-x64] [--dest DIR] [--archive FILE]

--dest defaults to build-deps/pdfium/<release>/<platform> in the repository (which build_native.bat uses);
--archive unpacks an archive already downloaded instead of downloading it (it must still match the pin).
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

RELEASE = "chromium/8152"  # PDFium 159.0.8152.0, released 2026-11-10
ARCHIVES = {
    "win-x64": ("pdfium-win-x64.tgz", "cee2b0e9990bc67c0721774932961b9dbf2766046c10eec925acd4cfba6fdfe3"),
    "linux-x64": ("pdfium-linux-x64.tgz", "384ab5c51882408a143a21f9c14dcd2caf1f75280ae41c554d0489b46c3dba9f"),
}
URL = "https://github.com/bblanchon/pdfium-binaries/releases/download/{release}/{name}"
REPOSITORY = Path(__file__).resolve().parents[2]


def default_platform() -> str:
    if sys.platform == "win32":
        return "win-x64"
    if sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64"):
        return "linux-x64"
    raise SystemExit(f"no pinned PDFium build for {sys.platform} {platform.machine()}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not (member.isfile() or member.isdir()) or not target.is_relative_to(dest.resolve()):
                raise SystemExit(f"refusing archive member {member.name!r}")
        if hasattr(tarfile, "data_filter"):  # Python 3.11.4 and later
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", choices=sorted(ARCHIVES), default=None)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--archive", type=Path, default=None)
    args = parser.parse_args()
    target = args.platform or default_platform()
    name, pinned = ARCHIVES[target]
    dest = args.dest or REPOSITORY / "build-deps" / "pdfium" / RELEASE.replace("/", "-") / target
    if (dest / "PDFiumConfig.cmake").exists():
        print(dest)
        return 0
    with tempfile.TemporaryDirectory() as scratch:
        archive = args.archive
        if archive is None:
            archive = Path(scratch) / name
            url = URL.format(release=RELEASE.replace("/", "%2F"), name=name)
            print(f"downloading {url}", file=sys.stderr)
            with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as file:
                shutil.copyfileobj(response, file)
        actual = sha256(archive)
        if actual != pinned:
            raise SystemExit(f"{archive}: SHA-256 {actual} is not the pinned {pinned}")
        staging = Path(scratch) / "unpacked"
        staging.mkdir()
        unpack(archive, staging)
        if not (staging / "PDFiumConfig.cmake").exists():
            raise SystemExit(f"{archive} holds no PDFiumConfig.cmake")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging, dest)
    print(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
