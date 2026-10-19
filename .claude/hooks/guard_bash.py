#!/usr/bin/env python3
"""PreToolUse hook for Bash and Monitor: the owner's rules, enforced by Claude Code instead of left to the model.

1. On a cloud machine (CLAUDE_CODE_REMOTE=true) nothing that runs this project's code may run: Isaac Sim, the
   drone scripts, their tests, any interpreter started inside the repository.  The owner runs and renders only on
   their own PC.
2. Isaac Sim renders of the drone scene go through scripts/drone_scene/realism/run_probe.sh, which queues them on
   the one GPU and outlives the 10-minute command limit; starting quadcopter_pbr.py any other way is refused.
3. Every git push asks the owner first (also `git -C dir push`, which a permission rule would miss), and force
   pushes, mirror pushes and remote branch deletions are refused outright.

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
KEYWORDS = {"{", "}", "do", "then", "else", "elif", "!", "if", "while", "until", "time", "function", "coproc"}
SHELLS = {"bash", "sh", "zsh", "dash"}
PROJECT_RUNNERS = {"isaaclab.sh", "run_probe.sh", "run_tests.sh", "python.sh"}
INTERPRETER = re.compile(r"python[0-9.]*|python\.sh|pytest|uv|node|perl|ruby|make|cmake|ninja|isaaclab\.sh|isaacsim")
# programs that only read, move or print files: mentioning a project script in their arguments does not run it
PASSIVE = {"git", "ls", "cat", "grep", "rg", "egrep", "fgrep", "head", "tail", "sed", "awk", "wc", "cut", "sort",
           "uniq", "diff", "stat", "file", "jq", "du", "df", "pwd", "cd", "echo", "printf", "test", "[", "true", "false",
           "mkdir", "cp", "mv", "rm", "touch", "ln", "chmod", "realpath", "dirname", "basename", "tee", "less", "more",
           "tr", "comm", "xxd", "md5sum", "sha1sum", "sha256sum", "tar", "zip", "unzip", "date", "hostname", "whoami",
           "which", "tree", "readlink", "nl", "cmp", "column", "fold", "od", "strings", "gh", "curl", "wget",
           # shell keywords and builtins that run nothing by themselves
           "for", "done", "fi", "case", "esac", "in", "[[", "]]", "export", "local", "read", "set", "unset", "shift",
           "return", "exit", "break", "continue", "wait", "type", "declare", "trap", "umask", "pushd", "popd"}
REDIRECT = re.compile(r"^\d*(?:[<>]{1,2}|&>{1,2}|>&)$")          # a redirection operator; the next word is a file
PROJECT_TEXT = r"isaaclab\.sh|quadcopter_pbr|run_probe\.sh|run_tests\.sh|drone_scene|heavy_machines|render_flight"
LINT_MODULES = {"py_compile", "compileall", "pyflakes", "flake8", "ruff", "black", "pylint", "isort"}
HEREDOC = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def split_heredocs(cmd: str) -> tuple[str, list[str]]:
    """(the command without here-document bodies, the bodies fed to an interpreter).  A body is data (a commit
    message, a file being written) unless the command it feeds is an interpreter, which would run it.  A `<<` with no
    terminator line further down (a here-string, a quoted "<<") consumes nothing, so it can hide nothing."""
    lines, out, run_bodies, i = cmd.split("\n"), [], [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for m in HEREDOC.finditer(line):
            tag = m.group(2)
            end = next((j for j in range(i, len(lines)) if lines[j].strip() == tag), None)
            if end is None:
                continue
            body = lines[i:end]
            i = end + 1
            head = re.split(r"[;&|]", line[:m.start()])[-1].split()
            if head and INTERPRETER.fullmatch(os.path.basename(head[0])):
                run_bodies.append("\n".join(body))
    return "\n".join(out), run_bodies


def strip_comments(cmd: str) -> str:
    """Drop `# ...` up to the end of the line where `#` starts a word outside quotes (as bash does); keep the newline,
    which separates commands."""
    out, quote, prev = [], None, "\n"
    i = 0
    while i < len(cmd):
        c = cmd[i]
        if quote:
            if c == "\\" and quote == '"' and i + 1 < len(cmd):
                out.append(cmd[i:i + 2])
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c == "\\" and i + 1 < len(cmd):
            out.append(cmd[i:i + 2])
            prev = cmd[i + 1]
            i += 2
            continue
        elif c == "#" and (prev.isspace() or prev in ";&|("):
            while i < len(cmd) and cmd[i] != "\n":
                i += 1
            continue
        out.append(c)
        prev = c
        i += 1
    return "".join(out)


def segments(cmd: str) -> list[list[str]]:
    """The simple commands in a command line, as word lists (split on ; && || | & ( ) and newlines)."""
    out, cur = [], []
    lex = shlex.shlex(strip_comments(cmd), posix=True, punctuation_chars=";&|()\n")
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    lex.commenters = ""
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
    """The words from the program actually run: past VAR=value, keywords, wrappers (with their options) and
    `bash script`."""
    i = 0
    while i < len(words):
        w = words[i]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", w) or w in KEYWORDS:
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


def without_redirects(words: list[str]) -> list[str]:
    """The words minus redirections (`> file`, `2>/dev/null`, `<in`): a file written or read is not a program."""
    out, skip = [], False
    for w in words:
        if skip:
            skip = False
        elif REDIRECT.match(w):
            skip = True
        elif not re.match(r"^\d*(?:[<>]{1,2}|&>{1,2}|>&)\S", w):
            out.append(w)
    return out


def inner_commands(words: list[str]) -> list[str]:
    """Command strings a word list runs in turn: `bash -c "..."` (also -xc, -ec, -lc, ...), `eval ...`."""
    if not words:
        return []
    exe = os.path.basename(words[0])
    if exe in SHELLS:
        for k, w in enumerate(words[1:-1], start=1):
            if re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", w):
                return [words[k + 1]]
            if not w.startswith("-"):
                break
    if exe == "eval":
        return [" ".join(words[1:])]
    return []


def runs_project_code(words: list[str]) -> bool:
    words = without_redirects(words)
    if not words:
        return False
    exe = os.path.basename(words[0])
    rest = " ".join(words[1:])
    if exe in PROJECT_RUNNERS:
        return True
    if INTERPRETER.fullmatch(exe):
        return bool(re.search(r"drone_scene|heavy_machines|isaaclab|quadcopter|render_flight", rest))
    if exe == "find":
        return bool({"-exec", "-execdir", "-ok", "-okdir"} & set(words)) and bool(re.search(PROJECT_TEXT, rest))
    if exe == "xargs":
        return bool(re.search(PROJECT_TEXT, rest))
    if exe not in PASSIVE and exe not in SHELLS and exe != "eval":
        return bool(re.search(PROJECT_TEXT, " ".join(words)))    # an unknown program handed a project script
    return False


def runs_any_code(words: list[str]) -> bool:
    """An interpreter, a build tool, a shell running a script, or a ./script: anything that executes code."""
    words = without_redirects(words)
    if not words:
        return False
    exe = os.path.basename(words[0])
    return bool(INTERPRETER.fullmatch(exe) or exe.endswith(".sh") or words[0].startswith("./")
                or (exe in SHELLS and len(words) > 1))


def programs(cmd: str, depth: int = 0) -> list[list[str]]:
    """Every simple command the command line runs, including those inside bash -c / eval strings and here-documents
    fed to an interpreter."""
    cmd, bodies = split_heredocs(cmd)
    cmd = re.sub(r"\\\r?\n", "", cmd)                # line continuations: bash joins the lines
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
            if re.search(PROJECT_TEXT, body):           # code that mentions the project: treat it as running it
                out.append(["python", body])
    return out


def direct_isaac_render(words: list[str]) -> bool:
    """quadcopter_pbr.py started by isaaclab.sh, a python, python.sh or isaacsim; a syntax check or lint is fine."""
    words = without_redirects(words)
    if not words or not INTERPRETER.fullmatch(os.path.basename(words[0])):
        return False
    if any(w == "-m" and k + 1 < len(words) and words[k + 1] in LINT_MODULES for k, w in enumerate(words)):
        return False
    return any(w.endswith("quadcopter_pbr.py") for w in words[1:])


def push_args(words: list[str]):
    """The words after `push` when the word list is a git push (also `git -C dir push`), else None."""
    if not words or os.path.basename(words[0]) != "git":
        return None
    i = 1
    while i < len(words) and words[i].startswith("-"):
        i += 2 if words[i] in ("-C", "-c", "--git-dir", "--work-tree", "--namespace") else 1
    return words[i + 1:] if i < len(words) and words[i] == "push" else None


def destructive_push(args: list[str]) -> bool:
    for w in args:
        if w in ("-f", "--force", "--mirror", "--delete", "-d", "--prune", "--force-if-includes") \
                or w.startswith("--force-with-lease") or re.fullmatch(r"-[A-Za-z]*[fd][A-Za-z]*", w):
            return True
        if not w.startswith("-") and (w.startswith("+") or w.startswith(":")):
            return True                                 # +refspec forces, :branch deletes
    return False


def inside(path: str, root: str) -> bool:
    try:
        path, root = os.path.realpath(path), os.path.realpath(root)
    except (TypeError, ValueError):
        return False
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def block(reason: str) -> None:
    print(reason, file=sys.stderr)
    sys.exit(2)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return
    if data.get("tool_name") not in ("Bash", "Monitor"):
        return
    cmd = (data.get("tool_input") or {}).get("command") or ""
    if not cmd:
        return
    progs = programs(cmd)
    if os.environ.get("CLAUDE_CODE_REMOTE") == "true":
        project = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
        in_repo = inside(data.get("cwd") or os.getcwd(), project) or any(p and p[0] == "cd" for p in progs)
        if any(runs_project_code(p) for p in progs) or (in_repo and any(runs_any_code(p) for p in progs)):
            block("Blocked: this is a cloud machine. The owner's rule is to run and render only on the owner's own PC, "
                  "with its GPU. Edit and commit here if asked; do not run the project's code (use Read, Grep and Glob, "
                  "or jq, to look at files).")
    if any(direct_isaac_render(p) for p in progs):
        block("Blocked: start Isaac Sim renders of the drone scene with scripts/drone_scene/realism/run_probe.sh "
              "(start for probe stills, record for the whole flight, then wait). It queues renders on the one GPU "
              "and keeps running past the 10-minute command limit.")
    pushes = [a for a in (push_args(p) for p in progs) if a is not None]
    if any(destructive_push(a) for a in pushes):
        block("Blocked: force pushes, mirror pushes and remote branch deletions are not allowed in this repository. "
              "Ask the owner; they can do it themselves if they really want it.")
    if pushes:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "ask",
            "permissionDecisionReason": "git push: the owner approves every push (repo seyedalicheraghi/IsaacSim)."}}))


if __name__ == "__main__":
    main()
