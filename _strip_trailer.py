import sys

lines = sys.stdin.read().splitlines()
kept = [l for l in lines if not l.startswith("Co-Authored-By: Claude")]
while kept and kept[-1] == "":
    kept.pop()
sys.stdout.write("\n".join(kept) + "\n")
