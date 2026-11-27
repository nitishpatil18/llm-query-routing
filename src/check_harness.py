import os
import resource
import subprocess
import sys
import tempfile

from datasets import load_dataset


def _limits():
    for res, val in ((resource.RLIMIT_CPU, 10), (resource.RLIMIT_FSIZE, 10 * 1024 * 1024)):
        try:
            resource.setrlimit(res, (val, val))
        except (ValueError, OSError):
            pass


HARNESS = "import sys; exec(compile(open(sys.argv[1]).read(), 'prog', 'exec'), {'__name__': 'prog'})"


def run_tests(program):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, "-I", "-c", HARNESS, path], cwd=td,
                               env={"PATH": "/usr/bin:/bin"}, stdin=subprocess.DEVNULL,
                               capture_output=True, timeout=10, preexec_fn=_limits)
        except subprocess.TimeoutExpired:
            return "timeout"
    return "pass" if r.returncode == 0 else "fail"


ds = load_dataset("openai/openai_humaneval", split="test")
counts = {"pass": 0, "fail": 0, "timeout": 0}
bad = []
for r in ds:
    s = run_tests(r["prompt"] + r["canonical_solution"] + "\n\n" + r["test"] + f"\n\ncheck({r['entry_point']})\n")
    counts[s] += 1
    if s != "pass":
        bad.append(r["task_id"])
print("canonical solutions:", counts, "| not passing:", bad)
r = ds[0]
print("stub returning None (must NOT pass):", run_tests(r["prompt"] + "\n    return None\n\n" + r["test"] + f"\n\ncheck({r['entry_point']})\n"))
print("infinite loop (must NOT pass, takes ~10s):", run_tests("while True:\n    pass\n"))
