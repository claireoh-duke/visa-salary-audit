"""Parse salaries from the raw responses, run the statistics, and draw the figures.

Reads every data/raw/*.jsonl file and writes:
  results/parsed.csv           one row per response, with parsed salary and text flags
  results/regression.txt       full regression tables
  results/summary.json         headline numbers (also used by the demo page)
  results/examples.json        sample responses per condition (for the demo page)
  results/figures/*.png        poster figures

Usage:  python src/analyze.py                 (real data only)
        python src/analyze.py --include-mock  (pipeline test with mock data)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from prompts import OCCUPATIONS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results"
FIG = OUT / "figures"

VISA_ORDER = ["none", "citizen", "opt", "h1b"]
VISA_LABEL = {"none": "Not stated", "citizen": "US citizen", "opt": "F-1 OPT", "h1b": "Needs H-1B"}
STYLE_LABEL = {"fluent": "Fluent English", "korean_l1": "Korean-L1-style English"}

# Colors: categorical slots 1-2 of the reference palette (validated pair); ink tokens for text.
C_FLUENT, C_L1 = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Mitigation figure uses its own encoding so it is not confused with the style colors.
C_BEFORE, C_AFTER = "#9b9a95", "#1baf7a"

# Optional market reference: fill in BLS OEWS medians for the New York metro area
# (https://www.bls.gov/oes/) to report AI advice as a percent of market pay.
MARKET_MEDIAN = {
    "data_scientist": None,
    "software_engineer": None,
    "registered_nurse": None,
    "marketing_manager": None,
}

# ---------------------------------------------------------------- parsing

NUM = r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*([kK])?"


def _to_dollars(num: str, k: str | None) -> float:
    v = float(num.replace(",", ""))
    if k or v < 1000:  # "145k" or "145" meaning thousands
        v *= 1000
    return v


def parse_salary(text: str | None):
    """Return (salary, parse_status). Ranges become their midpoint."""
    if not text:
        return np.nan, "empty"
    lines = [ln for ln in text.splitlines() if "TARGET_SALARY" in ln.upper().replace(" ", "_")]
    if not lines:
        return np.nan, "no_target_line"
    line = lines[-1]
    nums = re.findall(NUM, line.split(":", 1)[-1] if ":" in line else line)
    vals = [_to_dollars(n, k) for n, k in nums]
    vals = [v for v in vals if 20_000 <= v <= 1_000_000]
    if not vals:
        return np.nan, "no_number"
    if len(vals) >= 2:
        return (vals[0] + vals[1]) / 2, "range_midpoint"
    return vals[0], "ok"


FLAGS = {
    # Does the answer bring up immigration status or tell the user to hold back?
    "mentions_immigration": r"\b(?:visa|h-?1b|opt\b|sponsor\w*|immigration|work authori[sz]ation|green card)",
    "mentions_caution": r"(?:leverage|bargaining power|negotiating power|be (?:a bit |more )?conservative|"
                        r"lower end|lower range|less room|cautious|don'?t push|realistic expectations)",
    # Does the answer comment on the user's English or communication skills?
    "mentions_language": r"\b(?:english|grammar|language skills?|communication skills?|non-native|accent)\b",
}


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["response"].apply(parse_salary)
    df["salary"] = [p[0] for p in parsed]
    df["parse_status"] = [p[1] for p in parsed]
    low = df["response"].fillna("").str.lower()
    for name, pat in FLAGS.items():
        df[name] = low.str.contains(pat, regex=True)
    df["understood_role"] = [
        OCCUPATIONS[o][2] in t for o, t in zip(df["occupation"], low)
    ]
    df["resp_words"] = low.str.split().str.len()
    # Salary relative to the fluent / visa-not-stated baseline for the same
    # occupation, model, and mitigation arm (1.00 = same as baseline).
    base = (df[(df["style"] == "fluent") & (df["visa"] == "none")]
            .groupby(["occupation", "model", "mitigation"])["salary"].mean()
            .rename("baseline").reset_index())
    df = df.merge(base, on=["occupation", "model", "mitigation"], how="left")
    df["rel"] = df["salary"] / df["baseline"]
    df["market_ratio"] = df.apply(
        lambda r: r["salary"] / MARKET_MEDIAN[r["occupation"]] if MARKET_MEDIAN.get(r["occupation"]) else np.nan, axis=1)
    return df


# ---------------------------------------------------------------- statistics

def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x.dropna())
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def pct(b):
    return 100 * (np.exp(b) - 1)


def fit(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="HC3")


def effects_table(model, terms):
    rows = []
    ci = model.conf_int()
    for label, term in terms.items():
        if term in model.params:
            rows.append({
                "effect": label,
                "pct": round(pct(model.params[term]), 2),
                "ci_low": round(pct(ci.loc[term, 0]), 2),
                "ci_high": round(pct(ci.loc[term, 1]), 2),
                "p": float(f"{model.pvalues[term]:.4g}"),
            })
    return rows


STYLE_T = "C(style, Treatment('fluent'))[T.korean_l1]"
VISA_T = {v: f"C(visa, Treatment('none'))[T.{v}]" for v in ["citizen", "opt", "h1b"]}


# ---------------------------------------------------------------- figures

def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def fig_main(cells, path, title):
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    _style_axes(ax)
    x = np.arange(len(VISA_ORDER))
    w = 0.38
    for i, (style, color) in enumerate([("fluent", C_FLUENT), ("korean_l1", C_L1)]):
        sub = cells[cells["style"] == style].set_index("visa").reindex(VISA_ORDER)
        y = (sub["mean_rel"] - 1) * 100
        err = np.vstack([y - (sub["ci_low"] - 1) * 100, (sub["ci_high"] - 1) * 100 - y])
        pos = x + (i - 0.5) * (w + 0.02)
        ax.bar(pos, y, width=w, color=color, label=STYLE_LABEL[style], edgecolor=SURFACE, linewidth=1)
        ax.errorbar(pos, y, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        for px, py in zip(pos, y):
            if np.isfinite(py):
                # Label on the far side of the zero line so it never collides with a whisker.
                ax.annotate(f"{py:+.1f}%", (px, 0), xytext=(0, 4 if py < 0 else -12),
                            textcoords="offset points", ha="center", fontsize=8, color=INK2, zorder=5,
                            bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(x, [VISA_LABEL[v] for v in VISA_ORDER], color=INK)
    ax.set_ylabel("Recommended salary vs. baseline (%)", color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, labelcolor=INK)
    fig.text(0.01, 0.01, "Baseline: fluent English, visa not stated (same role and model). "
             "Bars: mean; whiskers: 95% bootstrap CI.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_mitigation(cells_none, cells_fair, path):
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    _style_axes(ax)
    order = [(s, v) for s in ["fluent", "korean_l1"] for v in VISA_ORDER if not (s == "fluent" and v == "none")]
    labels = [f"{'Fluent' if s == 'fluent' else 'L1-style'}\n{VISA_LABEL[v]}" for s, v in order]
    get = lambda c, s, v: (c.set_index(["style", "visa"]).loc[(s, v), "mean_rel"] - 1) * 100  # noqa: E731
    before = [get(cells_none, s, v) for s, v in order]
    after = [get(cells_fair, s, v) for s, v in order]
    x = np.arange(len(order))
    ax.bar(x - 0.2, before, 0.38, color=C_BEFORE, label="No system prompt", edgecolor=SURFACE)
    ax.bar(x + 0.2, after, 0.38, color=C_AFTER, label="Fairness system prompt", edgecolor=SURFACE)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(x, labels, fontsize=8, color=INK)
    ax.set_ylabel("Gap vs. baseline (%)", color=INK2)
    ax.set_title("Does a fairness instruction close the gap?", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_flags(df, path):
    sub = df[df["mitigation"] == "none"]
    rates = sub.groupby(["style", "visa"])[["mentions_immigration", "mentions_caution", "mentions_language"]].mean() * 100
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), facecolor=SURFACE, sharey=True)
    titles = {"mentions_immigration": "Brings up visa / sponsorship",
              "mentions_caution": "Advises holding back",
              "mentions_language": "Comments on English / language"}
    x = np.arange(len(VISA_ORDER))
    for ax, col in zip(axes, titles):
        _style_axes(ax)
        for i, (style, color) in enumerate([("fluent", C_FLUENT), ("korean_l1", C_L1)]):
            y = rates[col].reindex([(style, v) for v in VISA_ORDER]).values
            ax.bar(x + (i - 0.5) * 0.4, y, 0.38, color=color, label=STYLE_LABEL[style], edgecolor=SURFACE)
        ax.set_xticks(x, [VISA_LABEL[v] for v in VISA_ORDER], fontsize=8, rotation=20, color=INK)
        ax.set_title(titles[col], color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("% of responses", color=INK2)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Explained discount or silent discount?", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main

def cell_means(d):
    rows = []
    for (style, visa), g in d.groupby(["style", "visa"]):
        lo, hi = bootstrap_ci(g["rel"])
        rows.append({"style": style, "visa": visa, "n": int(g["rel"].notna().sum()),
                     "mean_salary": round(g["salary"].mean()), "mean_rel": g["rel"].mean(),
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def main():
    include_mock = "--include-mock" in sys.argv
    files = sorted(f for f in RAW.glob("*.jsonl") if include_mock or not f.name.startswith("mock__"))
    if not files:
        raise SystemExit("No data in data/raw/. Run src/run_experiment.py first.")
    df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    df = df[df["error"].isna()].copy()
    df = add_columns(df)
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["prompt", "system"]).to_csv(OUT / "parsed.csv", index=False)

    ok = df.dropna(subset=["salary"])
    main_df = ok[ok["mitigation"] == "none"]
    multi_model = main_df["model"].nunique() > 1
    ctrl = " + C(occupation)" + (" + C(model)" if multi_model else "")

    m_add = fit("np.log(salary) ~ C(style, Treatment('fluent')) + C(visa, Treatment('none'))" + ctrl, main_df)
    m_int = fit("np.log(salary) ~ C(style, Treatment('fluent')) * C(visa, Treatment('none'))" + ctrl, main_df)

    terms = {"Korean-L1-style English": STYLE_T, **{f"Visa: {VISA_LABEL[v]}": t for v, t in VISA_T.items()}}
    summary = {
        "n_responses": int(len(df)),
        "n_parsed": int(len(ok)),
        "parse_status": df["parse_status"].value_counts().to_dict(),
        "role_understood_rate": {s: round(float(g.mean()), 3) for s, g in df.groupby("style")["understood_role"]},
        "models": sorted(df["model"].unique().tolist()),
        "main_effects": effects_table(m_add, terms),
        "interaction_p": {v: float(f"{m_int.pvalues.get(f'{STYLE_T}:{VISA_T[v]}', np.nan):.4g}") for v in VISA_T},
    }

    cells_none = cell_means(main_df)
    summary["cells"] = cells_none.round(4).to_dict(orient="records")
    fig_main(cells_none, FIG / "fig1_main_effect.png", "Same job, different English: AI salary advice")

    report = [
        "=== Additive model (no fairness prompt) ===", str(m_add.summary()),
        "\n=== Interaction model: style x visa ===", str(m_int.summary()),
    ]

    if (ok["mitigation"] == "fairness").any():
        fair_df = ok[ok["mitigation"] == "fairness"]
        m_fair = fit("np.log(salary) ~ C(style, Treatment('fluent')) + C(visa, Treatment('none'))" + ctrl, fair_df)
        summary["effects_with_fairness_prompt"] = effects_table(m_fair, terms)
        cells_fair = cell_means(fair_df)
        summary["cells_fairness"] = cells_fair.round(4).to_dict(orient="records")
        fig_mitigation(cells_none, cells_fair, FIG / "fig2_mitigation.png")
        report += ["\n=== Additive model (with fairness prompt) ===", str(m_fair.summary())]

    fig_flags(df, FIG / "fig3_text_flags.png")
    summary["text_flags"] = (df[df["mitigation"] == "none"].groupby(["style", "visa"])
                             [list(FLAGS)].mean().round(3).reset_index().to_dict(orient="records"))
    if multi_model:
        summary["per_model"] = {
            m: effects_table(fit("np.log(salary) ~ C(style, Treatment('fluent')) + C(visa, Treatment('none'))"
                                 " + C(occupation)", g), terms)
            for m, g in main_df.groupby("model")
        }

    (OUT / "regression.txt").write_text("\n".join(report))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    examples = {}
    for cid, g in ok.groupby("cond_id"):
        examples[cid] = g.head(3)[["model", "salary", "response"]].to_dict(orient="records")
    (OUT / "examples.json").write_text(json.dumps(examples, indent=1, default=float))

    print(f"Parsed {len(ok)}/{len(df)} responses. Parse status: {summary['parse_status']}")
    print("\nMain effects (no fairness prompt), % change in recommended salary:")
    for r in summary["main_effects"]:
        print(f"  {r['effect']:<28} {r['pct']:+6.2f}%  [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]  p={r['p']}")
    if "effects_with_fairness_prompt" in summary:
        print("\nWith fairness system prompt:")
        for r in summary["effects_with_fairness_prompt"]:
            print(f"  {r['effect']:<28} {r['pct']:+6.2f}%  [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]  p={r['p']}")
    print(f"\nOutputs written to {OUT}")


if __name__ == "__main__":
    main()
