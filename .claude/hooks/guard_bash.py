#!/usr/bin/env python3
"""PreToolUse hook for Bash: the owner's rules, enforced by Claude Code instead of left to the model.

1. On a cloud machine (CLAUDE_CODE_REMOTE=true) nothing that runs this project's code may run: Isaac Sim, the
   drone scripts, their tests.  The owner runs and renders only on their own PC.
2. Isaac Sim renders of the drone scene go through scripts/drone_scene/realism/run_probe.sh, which queues them on
   the one GPU and outlives the 10-minute command limit; running quadcopter_pbr.py directly is refused.
3. Every git push asks the owner first (also `git -C dir push`, which a permission rule would miss).

Contract (Claude Code hooks): the tool call arrives as JSON on stdin; exit 2 with a reason on stderr blocks it;
a JSON "ask" decision on stdout shows the owner the permission prompt.  Standard library only.
"""

import json
import os
import re
import shlex
import sys

WRAPPERS = {"nohup", "setsid", "env", "time", "nice", "ionice", "stdbuf", "command", "builtin", "exec", "sudo",
            "timeout", "xargs", "flock"}
SHELLS = {"bash", "sh", "zsh", "dash"}
PROJECT_RUNNERS = {"isaaclab.sh", "run_probe.sh", "run_tests.sh"}
# programs that only read, move or print files: mentioning a project script in their arguments does not run it
PASSIVE = {"git", "ls", "cat", "grep", "rg", "egrep", "fgrep", "head", "tail", "sed", "awk", "wc", "cut", "sort",
           "uniq", "diff", "stat", "file", "jq", "du", "df", "pwd", "cd", "echo", "printf", "test", "[", "true", "false",
           "mkdir", "cp", "mv", "rm", "touch", "ln", "chmod", "realpath", "dirname", "basename", "tee", "less", "more",
           "tr", "comm", "xxd", "md5sum", "sha1sum", "tar", "zip", "unzip", "date", "hostname", "whoami", "which"}


INTERPRETERS = r"python[0-9.]*|bash|sh|zsh|dash|node|perl|ruby|isaaclab\.sh"


def split_heredocs(cmd: str) -> tuple[str, list[str]]:
    """(the command without here-document bodies, the bodies that are fed to an interpreter).  A body is data (a
    commit message, a file being written) unless the command it feeds is an interpreter, which would run it."""
    lines, out, run_bodies, i = cmd.split("\n"), [], [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for _, tag in re.findall(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1", line):
            body = []
            while i < len(lines) and lines[i].strip() != tag:
                body.append(lines[i])
                i += 1
            i += 1                                          # the terminator line
            head = re.split(r"[;&|]", line.split("<<")[0])[-1].split()
            if head and re.fullmatch(INTERPRETERS, os.path.basename(head[0])):
                run_bodies.append("\n".join(body))
    return "\n".join(out), run_bodies


def segments(cmd: str) -> list[list[str]]:
    """The simple commands in a command line, as word lists (split on ; && || | & and newlines)."""
    out, cur = [], []
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=";&|()\n")
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    for tok in lex:
        if tok and set(tok) <= set(";&|()\n"):
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        out.append(cur)
    return out


def program(words: list[str]) -> list[str]:
    """The words from the program actually run: past VAR=value, wrappers (with their options) and `bash script`."""
    i = 0
    while i < len(words):
        w = words[i]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", w):
            i += 1
        elif os.path.basename(w) in WRAPPERS:
            i += 1
            while i < len(words) and (words[i].startswith("-") or re.fullmatch(r"[0-9.]+[smhd]?", words[i])):
                i += 1
            if os.path.basename(w) == "flock" and i < len(words):
                i += 1                                  # the lock file
        elif os.path.basename(w) in SHELLS and i + 1 < len(words) and not words[i + 1].startswith("-"):
            i += 1                                      # bash script.sh ...
        else:
            break
    return words[i:]


PROJECT_TEXT = r"isaaclab\.sh|quadcopter_pbr|run_probe\.sh|run_tests\.sh|drone_scene|heavy_machines|render_flight"


def inner_commands(words: list[str]) -> list[str]:
    """Command strings a word list runs in turn: `bash -c "..."`, `sh -c`, `eval ...`."""
    if not words:
        return []
    exe = os.path.basename(words[0])
    if exe in SHELLS and len(words) > 2 and words[1] in ("-c", "-lc", "-ic"):
        return [words[2]]
    if exe == "eval":
        return [" ".join(words[1:])]
    return []


def runs_project_code(words: list[str]) -> bool:
    if not words:
        return False
    exe = os.path.basename(words[0])
    if exe in PROJECT_RUNNERS:
        return True
    if re.fullmatch(r"python[0-9.]*|pytest|uv", exe):
        rest = " ".join(words[1:])
        return bool(re.search(r"drone_scene|heavy_machines|isaaclab|quadcopter|render_flight", rest))
    if exe == "find" and "-exec" in words or exe == "xargs":
        return bool(re.search(PROJECT_TEXT, " ".join(words[1:])))
    if exe not in PASSIVE and not inner_commands(words):
        return bool(re.search(PROJECT_TEXT, " ".join(words)))    # an unknown program handed a project script
    return False


def programs(cmd: str, depth: int = 0) -> list[list[str]]:
    """Every simple command the command line runs, including those inside bash -c / eval strings and here-documents
    fed to an interpreter."""
    cmd, bodies = split_heredocs(cmd)
    try:
        progs = [program(s) for s in segments(cmd)]
    except ValueError:                                  # unbalanced quotes: judge the raw text
        return [cmd.split()]
    out = list(progs)
    if depth < 3:
        for p in progs:
            for inner in inner_commands(p):
                out += programs(inner, depth + 1)
        for body in bodies:
            out += programs(body, depth + 1)
            if re.search(PROJECT_TEXT, body):           # python code that mentions the project: treat as running it
                out.append(["python", body])
    return out


def direct_isaac_render(words: list[str]) -> bool:
    return bool(words) and os.path.basename(words[0]) == "isaaclab.sh" and any(
        w.endswith("quadcopter_pbr.py") for w in words[1:])


def is_push(words: list[str]) -> bool:
    if not words or os.path.basename(words[0]) != "git":
        return False
    i = 1
    while i < len(words) and words[i].startswith("-"):
        i += 2 if words[i] in ("-C", "-c", "--git-dir", "--work-tree", "--namespace") else 1
    return i < len(words) and words[i] == "push"


def block(reason: str) -> None:
    print(reason, file=sys.stderr)
    sys.exit(2)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return
    if data.get("tool_name") != "Bash":
        return
    cmd = (data.get("tool_input") or {}).get("command") or ""
    progs = programs(cmd)
    remote = os.environ.get("CLAUDE_CODE_REMOTE") == "true"
    if remote and any(runs_project_code(p) for p in progs):
        block("Blocked: this is a cloud machine. The owner's rule is to run and render only on the owner's own PC, "
              "with its GPU. Edit and commit here if asked; do not run the project's code.")
    if any(direct_isaac_render(p) for p in progs):
        block("Blocked: start Isaac Sim renders of the drone scene with scripts/drone_scene/realism/run_probe.sh "
              "(start for probe stills, record for the whole flight, then wait). It queues renders on the one GPU "
              "and keeps running past the 10-minute command limit.")
    if any(is_push(p) for p in progs):
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "ask",
            "permissionDecisionReason": "git push: the owner approves every push (repo seyedalicheraghi/IsaacSim)."}}))


if __name__ == "__main__":
    main()
