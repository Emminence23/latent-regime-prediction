"""Check that results/ matches the submitted report. Run after run_all.py."""
import json
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent / "results"
TOL = 5e-4

p1 = pd.read_csv(RES / "part1_iterations.csv")["test_mae"].tolist()
p2 = pd.read_csv(RES / "part2_components.csv")["test_mae"].tolist()
s = json.loads((RES / "summary.json").read_text())

expected = {
    "Part 1 (k=4, 5 iterations)": (p1[3], 9.860),
    "Part 2 (1 component)": (p2[0], 15.552),
    "Part 2 (16 components)": (p2[4], 9.559),
    "Best decision tree": (s["best_tree"]["mae"], 10.596),
}
ok = True
for name, (got, want) in expected.items():
    good = abs(got - want) < TOL
    ok &= good
    print(f"{'OK  ' if good else 'FAIL'} {name}: {got:.3f} (expected {want:.3f})")
raise SystemExit(0 if ok else 1)
