"""Tables and figures from trial results and the ODPB.

    uv run python scripts/report.py --trials outputs/synthetic/trial_results_SYNTHETIC.csv \
        --odpb outputs/synthetic/odpb_SYNTHETIC.csv --annotations a.csv b.csv --out outputs/report_SYNTHETIC

Inputs:
  --trials       operator sheet with a final ``outcome`` column (ok / wrong / fail / abstain)
  --odpb         CSV from scripts/build_odpb.py
  --annotations  two annotator sheets (video_code, outcome) -> Cohen's kappa
If any input is synthetic (file name contains SYNTHETIC or a ``synthetic`` column is 1), every
figure is stamped "SYNTHETIC" and the report says so at the top.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from saber.eval.io import fnum, is_synthetic, read_rows  # noqa: E402
from saber.eval.metrics import aurc, auroc, risk_coverage, trial_metrics  # noqa: E402
from saber.eval.odpb import GATE_SCORES  # noqa: E402
from saber.eval.stats import cluster_bootstrap, cohen_kappa, wilson  # noqa: E402


def stamp(fig, synthetic: bool) -> None:
    if synthetic:
        fig.text(
            0.5,
            0.5,
            "SYNTHETIC",
            fontsize=48,
            color="red",
            alpha=0.25,
            ha="center",
            va="center",
            rotation=30,
            transform=fig.transFigure,
        )
        fig.text(0.01, 0.01, "SYNTHETIC DATA - NOT A RESULT", fontsize=8, color="red")


def trial_section(rows: list[dict], synthetic: bool, out: Path) -> list[str]:
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in rows:
        if r.get("outcome"):
            cells[(r["system"], r["condition"])].append(r["outcome"])
    lines = [
        "## Trials",
        "",
        "| System | Condition | N | SR [95% Wilson] | WOR [95% Wilson] | FCR | AR | Risk |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (s, c), outs in sorted(cells.items()):
        m = trial_metrics(outs)
        sr_lo, sr_hi = wilson(m["N_ok"], m["N"])
        w_lo, w_hi = wilson(m["N_w"], m["N"])
        lines.append(
            f"| {s} | {c} | {m['N']} | {m['SR']:.2f} [{sr_lo:.2f}, {sr_hi:.2f}] | "
            f"{m['WOR']:.2f} [{w_lo:.2f}, {w_hi:.2f}] | {m['FCR']:.2f} | {m['AR']:.2f} | {m['risk']:.2f} |"
        )
    systems = sorted({s for s, _ in cells})
    conds = sorted({c for _, c in cells})
    fig, ax = plt.subplots(figsize=(max(6, len(conds) * 1.2), 4))
    width = 0.8 / max(len(systems), 1)
    for i, s in enumerate(systems):
        vals = [trial_metrics(cells[(s, c)])["WOR"] if (s, c) in cells else np.nan for c in conds]
        ax.bar(np.arange(len(conds)) + i * width, vals, width, label=s)
    ax.set_xticks(np.arange(len(conds)) + width * (len(systems) - 1) / 2, conds)
    ax.set_ylabel("wrong-object rate")
    ax.set_ylim(0, 1)
    ax.legend()
    stamp(fig, synthetic)
    fig.tight_layout()
    fig.savefig(out / "wor_by_condition.png", dpi=120)
    plt.close(fig)
    return lines + ["", "![WOR](wor_by_condition.png)", ""]


def odpb_section(rows: list[dict], synthetic: bool, out: Path, n_boot: int) -> list[str]:
    lines = [
        "## Offline decision-point bank",
        "",
        f"{len(rows)} decision points, {len({r['scene'] for r in rows})} scenes.",
        "",
        "| Gate | n | wrong rate | AUROC [95% scene bootstrap] | AURC |",
        "|---|---|---|---|---|",
    ]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    rng = np.random.default_rng(0)
    for gate, (col, sign, wcol, acol) in GATE_SCORES.items():
        sc = np.array([sign * fnum(r.get(col)) for r in rows])
        wr = np.array([fnum(r.get(wcol)) == 1 for r in rows])
        av = np.array([fnum(r.get(acol)) == 1 for r in rows]) & np.isfinite(sc)
        if av.sum() < 2:
            lines.append(f"| {gate} | 0 | - | - | - |")
            continue
        s, w = sc[av], wr[av]
        clusters = np.array([r["scene"] for r in rows])[av]
        est, lo, hi = cluster_bootstrap(
            lambda idx, s=s, w=w: auroc(s[idx], w[idx]), len(s), clusters, n_boot, rng=rng
        )
        lines.append(
            f"| {gate} | {len(s)} | {w.mean():.3f} | {est:.3f} [{lo:.3f}, {hi:.3f}] | {aurc(s, w):.3f} |"
        )
        _, cov, risk = risk_coverage(s, w)
        ax.plot(cov, risk, label=gate)
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (wrong among executed)")
    ax.legend(fontsize=7)
    stamp(fig, synthetic)
    fig.tight_layout()
    fig.savefig(out / "risk_coverage.png", dpi=120)
    plt.close(fig)
    for col in ("wrong_raw", "wrong_s1", "wrong_img"):
        by = defaultdict(list)
        for r in rows:
            by[r["condition"]].append(fnum(r.get(col)))
        lines.append(
            f"\n{col} by condition: " + ", ".join(f"{c}={np.nanmean(v):.3f}" for c, v in sorted(by.items()))
        )
    return lines + ["", "![risk-coverage](risk_coverage.png)", ""]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials")
    ap.add_argument("--odpb")
    ap.add_argument("--annotations", nargs=2)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="outputs/report")
    args = ap.parse_args(argv)
    synthetic = False
    loaded = {}
    for name in ("trials", "odpb"):
        p = getattr(args, name)
        if p:
            loaded[name] = read_rows(p)
            synthetic |= is_synthetic(p, loaded[name])
    if args.annotations:
        synthetic |= any(is_synthetic(p) for p in args.annotations)
    out = Path(args.out)
    if synthetic and "SYNTHETIC" not in out.name:
        out = out.with_name(out.name + "_SYNTHETIC")
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# SABER report", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DATA. These numbers are not results and must not be reported.**", ""]
    if "trials" in loaded:
        lines += trial_section(loaded["trials"], synthetic, out)
    if args.annotations:
        a, b = (read_rows(p) for p in args.annotations)
        bmap = {r["video_code"]: r["outcome"] for r in b}
        pairs = [(r["outcome"], bmap[r["video_code"]]) for r in a if r["video_code"] in bmap and r["outcome"]]
        k = cohen_kappa([x for x, _ in pairs], [y for _, y in pairs]) if pairs else float("nan")
        lines += ["## Annotation agreement", "", f"Cohen's kappa = {k:.3f} over {len(pairs)} videos", ""]
    if "odpb" in loaded:
        lines += odpb_section(loaded["odpb"], synthetic, out, args.n_boot)
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("report ->", out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
