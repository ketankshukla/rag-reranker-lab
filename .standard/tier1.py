# -*- coding: utf-8 -*-
"""The Tier 1 check launcher (tools/tier1_check.md §1).

The kit copies this file into a repository as `.standard/tier1.py`, and the project block's check command is

    uv run --isolated --no-project --no-config python -I -B .standard/tier1.py

It is a thin delegate: it holds no check logic, no tool version and no path of the standard. It reads
`standard_path` from the `yaml project` block of `docs/state/STATE.md` (default `docs/standard`) and hands its
arguments to `<standard_path>/tools/tier1_check.py`, which does every check.

uv runs it from a temporary environment built on the interpreter it chose, so an interpreter is named by its version
and base (`sys.base_prefix`), never by `sys.executable`, a path under uv's cache (tools/tier1_check.md §1).

It parses on **any** Python 3 so that an interpreter older than 3.9 reaches its own error message instead of a
SyntaxError: no f-strings, no annotations, no walrus, no `async`, no `match`; `%` formatting throughout. Files and
output are UTF-8, output with `backslashreplace`, because a Windows console's code page is often a legacy one such as 1252.

Exit codes: whatever the module exits with, or 1 for a failure of the launcher's own (an old interpreter, no
`STATE.md`, no module at the parsed path). The hook therefore fails closed.
"""

import io
import os
import re
import subprocess
import sys

# The module the launcher hands over to, relative to the standard, and where the project block lives.
MODULE = os.path.join("tools", "tier1_check.py")
STATE = os.path.join("docs", "state", "STATE.md")
DEFAULT_STANDARD_PATH = "docs/standard"

# The lowest interpreter the launcher and the module accept (§1). Kept as a tuple, not a string, so the comparison
# is numeric and 3.10 does not read as older than 3.9.
MINIMUM_PYTHON = (3, 9)

# The project block, and the one key the launcher reads out of it. This reader is deliberately **not** strict YAML:
# a kit on Windows writes `standard_path: "<drive>:\\t1\\m\\x"`, which pyyaml refuses as an unknown escape, and the
# launcher still has to find the checkout. It takes the text between the quotes exactly as written. Only an
# unindented key counts: an indented one is nested under another mapping or inside a block scalar, which a YAML
# reader would never take for the block's own key. The module's reader matches the same way.
BLOCK_RE = re.compile(r"```yaml project\s*\n(.*?)\n```", re.DOTALL)
KEY_RE = re.compile(r"^standard_path[ \t]*:[ \t]*(.*?)[ \t]*$")


def out(text):
    """Print to stdout as UTF-8, replacing what the console cannot show rather than raising."""
    data = (text + "\n").encode("utf-8", "backslashreplace")
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:  # a stdout replaced by a text-only object (a test harness, IDLE)
        sys.stdout.write(data.decode("utf-8", "backslashreplace"))
    else:
        stream.write(data)
        stream.flush()


def fail(text):
    """One-line refusal, and exit 1: the hook fails closed (§1)."""
    out("tier1: " + text)
    return 1


def read_text(path):
    """Read a file as UTF-8, tolerating a BOM; `io.open` so the call reads the same on every Python 3."""
    with io.open(path, "r", encoding="utf-8-sig") as handle:
        return handle.read()


def standard_path_of(text):
    """The `standard_path` value of the `yaml project` block, or None where there is no block.

    An empty string means the block is there without the key, which is the documented default (`docs/standard`).
    """
    found = BLOCK_RE.search(text)
    if found is None:
        return None
    for line in found.group(1).splitlines():
        matched = KEY_RE.match(line)
        if matched is None:
            continue
        value = matched.group(1).strip()
        quote = value[:1]
        if quote == '"' or quote == "'":
            closing = value.find(quote, 1)
            if closing != -1:
                return value[1:closing]
        comment = value.find(" #")
        if comment != -1:
            value = value[:comment].rstrip()
        return value
    return ""


def repository_of(script):
    """The repository root: the folder holding this launcher (`.standard/`), one level up."""
    return os.path.dirname(os.path.dirname(os.path.abspath(script)))


def main(argv):
    if sys.version_info[:2] < MINIMUM_PYTHON:
        running = "%d.%d.%d" % sys.version_info[:3]
        base = getattr(sys, "base_prefix", sys.prefix)
        return fail(
            "this check needs Python %d.%d or later, but the interpreter is running %s, on the base interpreter "
            "at %s. Recreate that environment at %d.%d or later, move it out of the way, or deactivate it."
            % (MINIMUM_PYTHON[0], MINIMUM_PYTHON[1], running, base, MINIMUM_PYTHON[0], MINIMUM_PYTHON[1])
        )

    repository = repository_of(__file__)
    state = os.path.join(repository, STATE)
    if not os.path.isfile(state):
        return fail("no %s under %s, so the standard cannot be found." % (STATE, repository))

    parsed = standard_path_of(read_text(state))
    if parsed is None:
        return fail("%s holds no ```yaml project``` block, so the standard cannot be found." % state)
    standard = os.path.normpath(os.path.join(repository, parsed.replace("\\", os.sep) or DEFAULT_STANDARD_PATH))

    module = os.path.join(standard, MODULE)
    if not os.path.isfile(module):
        return fail("no check module at %s. Check standard_path in %s, or restore the checkout." % (module, state))

    # -I keeps PYTHON* variables and the user site folder out, and the module's own folder off sys.path, so nothing
    # planted beside it can shadow what it imports; -X utf8 makes the module's own text UTF-8; -B writes no bytecode
    # into the standard. subprocess.call, never os.execv, which on Windows loses the exit code.
    command = [sys.executable, "-I", "-X", "utf8", "-B", module, "--repository", repository]
    return subprocess.call(command + list(argv))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
