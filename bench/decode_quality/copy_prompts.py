"""Copy-heavy decode workload for infernix_decode_quality_gen's `prompts` mode.

Each prompt is one source file of this repository at a pinned commit followed by the header of
the same file with one identifier renamed, so the continuation copies the file and departs from
the source where the renamed identifier occurs: n-gram copy proposals run long, then break and
rematch, as in an agent's rewrite of a file it has just read. Raw text, no chat template.

Usage: py -3.11 bench/decode_quality/copy_prompts.py <repo> <commit> <out.txt> [count] [min_chars]
       [max_chars]

The files are the commit's own C++, CUDA and Python sources (apps, bench, src, tests, tools) of
min_chars..max_chars characters (default 2800..4800, about 800..1400 tokens), in path order, every
n-th one so that `count` (default 24) are taken. Each file's renamed identifier is the most
frequent name of at least six characters, outside include/import lines and language words, that
occurs two to eight times in it (ties: the first in the file).
"""

import collections
import re
import subprocess
import sys

IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{5,}\b")
COMMENT = {".py": "#", ".cpp": "//", ".h": "//", ".cu": "//", ".cuh": "//"}
# The project's own sources only.
ROOTS = ("apps/", "bench/", "src/", "tests/", "tools/")
# Language words and library names a rename would not touch.
RESERVED = {
    "alignas", "alignof", "constexpr", "consteval", "constinit", "const_cast", "co_await", "co_return",
    "decltype", "default", "define", "delete", "double", "dynamic_cast", "explicit", "extern", "friend",
    "include", "inline", "int16_t", "int32_t", "int64_t", "mutable", "namespace", "noexcept", "nullptr",
    "operator", "pragma", "private", "protected", "public", "register", "reinterpret_cast", "return",
    "signed", "sizeof", "static", "static_assert", "static_cast", "struct", "switch", "template",
    "thread_local", "typedef", "typeid", "typename", "uint16_t", "uint32_t", "uint64_t", "unsigned",
    "virtual", "volatile", "size_t", "ptrdiff_t", "__device__", "__global__", "__host__",
    "__forceinline__", "__restrict__", "__shared__", "assert", "import", "lambda", "nonlocal",
    "global", "except", "finally", "isinstance", "print", "string", "vector", "optional",
}


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True).stdout


def renamed_identifier(text):
    # Names on include and import lines belong to other files.
    body = "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith(("#include", "import ", "from "))
    )
    counts = collections.Counter(IDENTIFIER.findall(body))
    first = {}
    for match in IDENTIFIER.finditer(body):
        first.setdefault(match.group(0), match.start())
    candidates = [name for name, n in counts.items() if 2 <= n <= 8 and name not in RESERVED]
    if not candidates:
        return None
    return min(candidates, key=lambda name: (-counts[name], first[name]))


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    repo, commit, out = sys.argv[1:4]
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    low = int(sys.argv[5]) if len(sys.argv) > 5 else 2800
    high = int(sys.argv[6]) if len(sys.argv) > 6 else 4800
    paths = sorted(
        path
        for path in git(repo, "ls-tree", "-r", "--name-only", commit).decode().splitlines()
        if path.startswith(ROOTS) and any(path.endswith(suffix) for suffix in COMMENT)
    )
    eligible = []
    for path in paths:
        try:
            text = git(repo, "show", f"{commit}:{path}").decode("utf-8")
        except UnicodeDecodeError:
            continue
        text = text.replace("\r\n", "\n")
        if low <= len(text) <= high and "<<<END>>>" not in text and renamed_identifier(text):
            eligible.append((path, text))
    if len(eligible) < count:
        sys.exit(f"only {len(eligible)} eligible files at {commit}")
    step = len(eligible) / count
    with open(out, "w", encoding="utf-8", newline="\n") as file:
        for i in range(count):
            path, text = eligible[int(i * step)]
            old = renamed_identifier(text)
            new = old + "_next"
            mark = COMMENT[path[path.rfind("."):]]
            file.write(f"{mark} ===== {path} =====\n{text.rstrip()}\n\n")
            file.write(f"{mark} ===== {path}, with `{old}` renamed to `{new}`, otherwise unchanged =====\n")
            file.write("\n<<<END>>>\n")
    print(f"{count} prompts from {len(eligible)} eligible files at {commit} -> {out}")


if __name__ == "__main__":
    main()
