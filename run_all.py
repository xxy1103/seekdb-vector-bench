"""Master runner executing all 5 mechanism verification experiments and saving summary_metrics.json."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments import (  # noqa: E402
    exp1_curse_and_metrics,
    exp2_ann_and_quantization,
    exp3_filtered_ann_collapse,
    exp4_hybrid_rrf_vs_linear,
    exp5_slo_and_delta_snapshot_p99,
)


def main():
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Running Exp 1: Curse of Dimensionality & Metric Equivalence...")
    r1 = exp1_curse_and_metrics.run(results_dir)

    print("[2/5] Running Exp 2: ANN & Quantization Pareto + Live seekdb Engine...")
    r2 = exp2_ann_and_quantization.run(results_dir)

    print("[3/5] Running Exp 3: Filtered ANN Selectivity Collapse & ACORN 2-Hop...")
    r3 = exp3_filtered_ann_collapse.run(results_dir)

    print("[4/5] Running Exp 4: Hybrid Search RRF vs Linear Fusion...")
    r4 = exp4_hybrid_rrf_vs_linear.run(results_dir)

    print("[5/5] Running Exp 5: Recall-QPS-P99 SLO & Dual-Layer Delta+Snapshot P99...")
    r5 = exp5_slo_and_delta_snapshot_p99.run(results_dir)

    summary = {
        "exp1": r1,
        "exp2": r2,
        "exp3": r3,
        "exp4": r4,
        "exp5": r5,
    }
    with open(results_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done! All metrics saved to", results_dir / "summary_metrics.json")


if __name__ == "__main__":
    main()
