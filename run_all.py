"""Reproduce every number and figure in the report.  Usage: python run_all.py"""
import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from lrp import experiments as ex, plots
from lrp.data import load_data

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
RES, FIG = ROOT / "results", ROOT / "figures"


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    d = load_data()
    print(f"Train {len(d['train'])} rows | Test {len(d['test'])} rows")

    p1 = ex.part1_iterations(d)
    pd.DataFrame({"n_iter": ex.ITERS, "test_mae": p1}).to_csv(RES / "part1_iterations.csv", index=False)
    plots.plot_part1(ex.ITERS, p1, ex.KNEE_N_ITER, FIG / "part1_iterations.png")
    print("Part 1:", " ".join(f"{m:.3f}" for m in p1))

    p2, models = ex.part2_components(d)
    pd.DataFrame({"n_components": ex.COMPONENTS, "test_mae": p2}).to_csv(RES / "part2_components.csv", index=False)
    plots.plot_part2(ex.COMPONENTS, p2, FIG / "part2_components.png")
    print("Part 2:", " ".join(f"{m:.3f}" for m in p2))

    plots.plot_part3(ex.part3_assignments(d, models), FIG / "part3_assignments.png")
    print("Part 3: box plots saved")

    p4 = ex.part4_ablation(d)
    tab = pd.DataFrame(p4, index=ex.COMPONENTS).T
    tab.index.name = "excluded"
    tab.loc["(none)"] = p2
    tab.round(4).to_csv(RES / "part4_ablation.csv")
    plots.plot_part4(ex.COMPONENTS, p2, p4, FIG / "part4_ablation.png")
    print("Part 4:\n" + tab.round(2).to_string())

    best, trees, mixtures = ex.part5_tree_vs_mixture(d)
    pd.DataFrame(trees).to_csv(RES / "part5_tree.csv", index=False)
    pd.DataFrame(mixtures).to_csv(RES / "part5_mixture.csv", index=False)
    plots.plot_part5(trees, mixtures, FIG / "part5_speed_accuracy.png")
    print("Part 5: best tree", best)

    mix4 = p2[ex.COMPONENTS.index(4)]
    summary = {
        "mixture_k4_iter5_mae": mix4,
        "single_linear_mae": p2[0],
        "best_tree": best,
        "mae_reduction_vs_tree_pct": 100 * (1 - mix4 / best["mae"]),
        "latency_ratio_k4_vs_depth10_tree": (
            next(m["pred_time"] for m in mixtures if m["n_components"] == 4)
            / next(t["pred_time"] for t in trees if t["max_depth"] == 10)),
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Mixture (k=4) {mix4:.3f} vs best tree {best['mae']:.3f}: "
          f"{summary['mae_reduction_vs_tree_pct']:.1f}% lower MAE, "
          f"{summary['latency_ratio_k4_vs_depth10_tree']:.1f}x the prediction time")


if __name__ == "__main__":
    main()
