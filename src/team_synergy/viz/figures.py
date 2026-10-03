"""Matplotlib figures for spec section 8.

Each figure function takes the relevant AnalysisResult(s) and returns a matplotlib Figure.
save_all() writes PNG at 150 dpi to a directory, skipping figures whose required data is empty.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from team_synergy.analysis.result import AnalysisResult

log = logging.getLogger("team_synergy")

# Matplotlib style
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = False
plt.rcParams["figure.facecolor"] = "white"


def fig2(res: AnalysisResult) -> plt.Figure | None:
    """Kernel densities of team residuals (WAR vs WAR-), annotate sd ratio + CI."""
    if "fig2_density" not in res.tables or res.tables["fig2_density"].empty:
        return None

    density = res.tables["fig2_density"]
    ratio_info = res.tables.get("fig2_ratio", pd.DataFrame())

    fig, ax = plt.subplots(figsize=(8, 5))
    for spec in density["spec"].unique():
        d = density[density["spec"] == spec]
        ax.plot(d["x"], d["density"], label=spec, lw=2)

    ax.set_xlabel("Residual (wins)")
    ax.set_ylabel("Density")
    ax.set_title("Fig. 2: Residual distributions (WAR vs WAR-)")
    ax.legend()

    if not ratio_info.empty:
        r = ratio_info.iloc[0]
        ratio = r["ratio"]
        ci_low, ci_high = r["ci_low"], r["ci_high"]
        ax.text(0.98, 0.97, f"Ratio σ(WAR−)/σ(WAR) = {ratio:.3f}\nCI [{ci_low:.3f}, {ci_high:.3f}]",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    return fig


def fig3(res: AnalysisResult) -> plt.Figure | None:
    """Per-franchise boxplot of 0-100 culture ranks across windows, sorted by median."""
    if "fig3_summary" not in res.tables or res.tables["fig3_summary"].empty:
        return None

    scores = res.tables["fig3_scores"]
    summary = res.tables["fig3_summary"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get primary source for sorting
    try:
        primary = summary[summary["primary"]]
    except (KeyError, TypeError):
        primary = summary
    if primary.empty:
        primary = summary

    # Sort by median
    sorted_franch = primary.sort_values("median", ascending=False)["franch_id"].tolist()

    # Prepare data for boxplot
    data_by_franch = []
    for franch in sorted_franch:
        mask = scores["franch_id"] == franch
        data_by_franch.append(scores[mask]["score"].values)

    bp = ax.boxplot(data_by_franch, patch_artist=True)
    ax.set_xticklabels(sorted_franch)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")

    # Add median dots
    for i, franch in enumerate(sorted_franch):
        med = summary[summary["franch_id"] == franch]["median"].values
        if len(med) > 0:
            ax.plot(i + 1, med[0], "o", color="red", markersize=6)

    ax.set_ylabel("Culture rank (0-100)")
    ax.set_title("Fig. 3: Franchise culture ranking (λ rank) across windows")
    ax.set_xlabel("Franchise")
    fig.autofmt_xdate(rotation=45, ha="right")
    fig.tight_layout()

    return fig


def fig4(res: AnalysisResult) -> plt.Figure | None:
    """Team residual vs tcWAR scatter with labels on flagged extreme points."""
    if "fig4" not in res.tables or res.tables["fig4"].empty:
        return None

    data = res.tables["fig4"]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot all points
    ax.scatter(data["tc_war"], data["eps"], alpha=0.5, s=30)

    # Label extreme points
    labeled = data[data["label"].notna()]
    for _, row in labeled.iterrows():
        ax.annotate(row["label"], (row["tc_war"], row["eps"]),
                    fontsize=8, ha="right", xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("tcWAR (wins)")
    ax.set_ylabel("Team residual (wins)")
    ax.set_title("Fig. 4: Team synergy vs residuals")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()

    return fig


def fig5(res: AnalysisResult) -> plt.Figure | None:
    """Horizontal bars of cumulative team synergy wins-above-expected, sorted."""
    if "fig5_bars" not in res.tables or res.tables["fig5_bars"].empty:
        return None

    data = res.tables["fig5_bars"].sort_values("synergy_wins", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(data["franch_id"], data["synergy_wins"], color="steelblue")
    ax.set_xlabel("Cumulative synergy wins")
    ax.set_title("Fig. 5: Team synergy wins-above-expected (by season sum)")
    fig.tight_layout()

    return fig


def fig6(res: AnalysisResult) -> plt.Figure | None:
    """Two panels: WAR- vs WAR and WAR+ vs WAR, 45° line, vertical lines at WAR=1,4."""
    if "fig6" not in res.tables or res.tables["fig6"].empty:
        return None

    data = res.tables["fig6"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (y_col, title) in zip(axes, [("war_minus", "WAR−"), ("war_plus", "WAR+")]):
        ax.scatter(data["war"], data[y_col], alpha=0.3, s=20)

        # 45° line
        lims = [min(data["war"].min(), data[y_col].min()), max(data["war"].max(), data[y_col].max())]
        ax.plot(lims, lims, "k--", alpha=0.3, lw=1)

        # Vertical lines at WAR = 1, 4
        ax.axvline(1, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(4, color="gray", linestyle=":", alpha=0.5)

        ax.set_xlabel("WAR (wins)")
        ax.set_ylabel(f"{title} (wins)")
        ax.set_title(f"Fig. 6: {title} vs WAR")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    fig.tight_layout()
    return fig


def fig7(res: AnalysisResult) -> plt.Figure | None:
    """pcWAR vs WAR with vertical lines at WAR=1,4."""
    if "fig7" not in res.tables or res.tables["fig7"].empty:
        return None

    data = res.tables["fig7"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data["war"], data["pc_war"], alpha=0.3, s=20)

    # Vertical lines at WAR = 1, 4
    ax.axvline(1, color="gray", linestyle=":", alpha=0.5, label="WAR = 1")
    ax.axvline(4, color="gray", linestyle=":", alpha=0.5, label="WAR = 4")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("WAR (wins)")
    ax.set_ylabel("pcWAR (wins)")
    ax.set_title("Fig. 7: Player complementarity (pcWAR) vs WAR")
    ax.legend(loc="upper left")
    fig.tight_layout()

    return fig


MAX_BARS = 30  # bars per ranking panel; the full quartile stays in the CSV table


def _rank_panels(data: pd.DataFrame, value: str, parts: list[tuple[str, str, str]] | None,
                 title: str, xlabel: str, max_bars: int = MAX_BARS) -> plt.Figure:
    """Side-by-side top/bottom ranking panels (paper Figs. 8 and 11 layout).

    ``parts`` = [(column, label, color), ...] are stacked as diverging bars: positive parts
    stack right of 0 and negative parts left of 0, so a bar never starts from the wrong side.
    The paper shows the full top/bottom quartile; we draw at most ``max_bars`` per side."""
    sides = [("top", "Top"), ("bottom", "Bottom")]
    n_max = max(min(max_bars, int((data["side"] == s).sum())) for s, _ in sides)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.0, 0.22 * n_max + 1.5)))
    for ax, (side, lab) in zip(axes, sides):
        d = data[data["side"] == side].sort_values(value, ascending=(side == "bottom"))
        n_all = len(d)
        d = d.head(max_bars).iloc[::-1]  # best at the top of the panel
        y = np.arange(len(d))
        if parts:
            pos_left = np.zeros(len(d))
            neg_left = np.zeros(len(d))
            for col, plab, color in parts:
                v = d[col].to_numpy()
                left = np.where(v >= 0, pos_left, neg_left)
                ax.barh(y, v, left=left, color=color, label=plab, height=0.8)
                pos_left += np.clip(v, 0, None)
                neg_left += np.clip(v, None, 0)
            ax.scatter(d[value], y, color="black", s=8, zorder=3, label="total")
        else:
            ax.barh(y, d[value], color=np.where(d[value] >= 0, "steelblue", "coral"), height=0.8)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(d["name"], fontsize=7)
        shown = f"{len(d)} of {n_all}" if n_all > len(d) else f"{n_all}"
        ax.set_title(f"{lab} 25% ({shown} shown)", fontsize=10)
        ax.set_xlabel(xlabel)
    if parts:
        axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def fig8(res: AnalysisResult) -> plt.Figure | None:
    """Fig. 8: career-average pcWAR rankings, split into character and team components."""
    if "fig8" not in res.tables or res.tables["fig8"].empty:
        return None
    return _rank_panels(res.tables["fig8"], "pc_war",
                        [("pc_war_char", "character (player)", "steelblue"),
                         ("pc_war_team", "team player x culture", "coral")],
                        "Fig. 8: Career-average pcWAR, active players", "Career-average pcWAR (wins)")


def fig9(res: AnalysisResult) -> plt.Figure | None:
    """Career complementarity wins vs average lagged WAR, with percentile lines."""
    if "fig9" not in res.tables or res.tables["fig9"].empty:
        return None

    data = res.tables["fig9"]
    lines = res.tables.get("fig9_lines", pd.DataFrame())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data["mean_prev_war"], data["career_resid"], alpha=0.4, s=30)

    # Add percentile lines
    for _, row in lines.iterrows():
        # fig9_lines are percentiles of career wins-above-expected: horizontal lines only.
        ax.axhline(row["value"], color="gray", linestyle="--", alpha=0.6)
    for w in (1, 4):  # FanGraphs Scrub/Role and Role/Star thresholds
        ax.axvline(w, color="gray", linestyle=":", alpha=0.6)

    ax.set_xlabel("Career-average lagged WAR")
    ax.set_ylabel("Career complementarity (sum pcWAR residuals)")
    ax.set_title("Fig. 9: Player complementarity persistence")
    ax.axhline(0, color="black", linestyle="-", alpha=0.2, lw=0.5)
    ax.axvline(0, color="black", linestyle="-", alpha=0.2, lw=0.5)
    fig.tight_layout()

    return fig


def fig10(res: AnalysisResult) -> plt.Figure | None:
    """Small multiples of age profile estimate with CI, ages 20-40."""
    if "fig10" not in res.tables or res.tables["fig10"].empty:
        return None

    data = res.tables["fig10"]
    positions = sorted(data["pos"].unique())

    # Create grid
    n_pos = len(positions)
    n_cols = 4
    n_rows = (n_pos + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 2.5))
    axes = axes.flatten()

    for i, pos in enumerate(positions):
        ax = axes[i]
        d = data[data["pos"] == pos].sort_values("age")

        if len(d) > 0:
            ax.plot(d["age"], d["estimate"], "o-", color="steelblue", lw=2)
            ax.fill_between(d["age"], d["ci_low"], d["ci_high"], alpha=0.2, color="steelblue")
            ax.set_title(pos)
            ax.set_xlabel("Age")
            ax.set_ylabel("Effect (wins)")
            ax.set_xlim(20, 40)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Fig. 10: Age-productivity profile by position (marginal effects + 95% CI)")
    fig.tight_layout()
    return fig


def fig11(res: AnalysisResult) -> plt.Figure | None:
    """Fig. 11: career-average Intangibles (xi) rankings, active players."""
    if "fig11" not in res.tables or res.tables["fig11"].empty:
        return None
    return _rank_panels(res.tables["fig11"], "xi", None,
                        "Fig. 11: Career-average Intangibles, active players", "Career-average xi (wins)")


def fig12(res: AnalysisResult) -> plt.Figure | None:
    """One player's pcWAR and ξ vs WAR, points labelled by season (+ team)."""
    if "fig12" not in res.tables or res.tables["fig12"].empty:
        return None

    data = res.tables["fig12"]
    player = res.tables.get("player", pd.DataFrame())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y_col, ylabel in zip(axes, ["pc_war", "xi"], ["pcWAR (wins)", "Intangibles ξ (wins)"]):
        ax.scatter(data["war"], data[y_col], s=50, alpha=0.6)

        for _, row in data.iterrows():
            label = str(int(row["season"]))
            if pd.notna(row.get("franch_id")):
                label += f" {row['franch_id']}"
            ax.annotate(label, (row["war"], row[y_col]), fontsize=8,
                        xytext=(3, 3), textcoords="offset points")

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("WAR (wins)")
        ax.set_ylabel(ylabel)
        if len(player) > 0:
            ax.set_title(f"Fig. 12: {player.iloc[0]['name']}")
        else:
            ax.set_title("Fig. 12: Player career trajectory")

    fig.tight_layout()
    return fig


def save_all(results: dict[str, AnalysisResult], fig_dir: Path | str, dpi: int = 150) -> list[str]:
    """Save all figures to PNG at given dpi.

    Args:
        results: dict mapping module name to AnalysisResult.
        fig_dir: output directory.
        dpi: dots per inch (default 150).

    Returns:
        list of paths saved (relative to fig_dir).
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Map figure name to (result module, figure function)
    figures = [
        ("fig2", "table1", fig2),
        ("fig3", "org_culture", fig3),
        ("fig4", "tc_persistence", fig4),
        ("fig5", "tc_persistence", fig5),
        ("fig6", "player_eval", fig6),
        ("fig7", "player_eval", fig7),
        ("fig8", "player_eval", fig8),
        ("fig9", "pc_persistence", fig9),
        ("fig10", "profiles", fig10),
        ("fig11", "intangibles", fig11),
        ("fig12", "ross", fig12),
    ]

    saved = []
    for fig_name, mod_name, fig_func in figures:
        res = results.get(mod_name)
        if res is None:
            log.warning(f"{fig_name}: result for {mod_name} not found")
            continue

        fig = fig_func(res)
        if fig is None:
            log.debug(f"{fig_name}: skipped (empty tables)")
            continue

        path = fig_dir / f"{fig_name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        log.info(f"saved {path}")
        saved.append(fig_name + ".png")

    return saved
