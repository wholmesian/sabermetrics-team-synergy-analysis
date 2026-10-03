"""Write reproduction report in Korean (REPORT.md in outputs/<war_source>/)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from team_synergy.analysis.inputs import AnalysisInputs
from team_synergy.analysis.result import AnalysisResult

log = logging.getLogger("team_synergy")


def _markdown_table(df: pd.DataFrame, max_rows: int = 15, round_digits: int = 3) -> str:
    """Simple Markdown table from DataFrame, rounding floats."""
    df = df.copy()

    # Round float columns
    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        df[col] = df[col].apply(lambda x: round(x, round_digits) if pd.notna(x) else x)

    # Truncate rows
    omitted = ""
    if len(df) > max_rows:
        omitted = f"\n\n*({len(df) - max_rows}개 행 생략됨)*"
        df = df.head(max_rows)

    # Build table
    lines = []
    lines.append("| " + " | ".join(str(c) for c in df.columns) + " |")
    lines.append("|" + "|".join(["---"] * len(df.columns)) + "|")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")

    return "\n".join(lines) + omitted


def _checks_table(checks: list[dict]) -> str:
    """Render checks as a Korean table: 항목 | 값 | 논문 기대 결과 | 판정."""
    lines = []
    lines.append("| 항목 | 값 | 논문 기대 결과 | 판정 |")
    lines.append("|---|---|---|---|")

    for check in checks:
        item = check.get("item", "")
        value = check.get("value", "")
        expected = check.get("expected", "")
        passed = check.get("passed")

        if passed is True:
            verdict = "✅"
        elif passed is False:
            verdict = "❌"
        else:
            verdict = "실데이터 필요"

        lines.append(f"| {item} | {value} | {expected} | {verdict} |")

    return "\n".join(lines)


def write_report(results: dict[str, AnalysisResult], inputs: AnalysisInputs, out_path: Path | str,
                 fig_dir_rel: str = "figures") -> None:
    """Write REPORT.md in Korean.

    Args:
        results: dict mapping module name to AnalysisResult.
        inputs: AnalysisInputs with cfg and war_source.
        out_path: path to REPORT.md.
        fig_dir_rel: relative path to figures directory from report.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = inputs.cfg
    war_source = inputs.war_source
    seed = cfg.get("seed", 0)

    # Count checks
    check_counts = {"passed": 0, "failed": 0, "unverified": 0}
    for res in results.values():
        for check in res.checks:
            p = check.get("passed")
            if p is True:
                check_counts["passed"] += 1
            elif p is False:
                check_counts["failed"] += 1
            else:
                check_counts["unverified"] += 1

    # Build sections
    sections = []

    # Header
    sections.append(f"# 재현 보고서: Brave et al. (2019) 팀 시너지 모형\n")
    sections.append(f"**데이터:** {war_source}\n")
    sections.append(f"**생성 날짜:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    if war_source == "synthetic":
        sections.append("**참고:** 합성 데이터는 MLB에 보정되지 않았다. 논문 수치와의 비교는 코드가 끝까지 돌아가는지와 부호·방향의 구조적 점검일 뿐이며, ❌는 합성 데이터 생성 과정의 한계일 수 있다. 선수 이름이 필요한 점검은 '실데이터 필요'로 표시한다.\n")
    sections.append("")

    # Sections per paper item (spec 8 order): (title, module, figure stem or None,
    # [(table, selector, max_rows)]). Selectors pick a readable summary; raw plotting
    # data (KDE grids, 570-point scatters) stays in the CSVs under tables/.
    def _extremes(df):
        return df[df["label"].notna() & (df["label"].astype(str) != "")] if "label" in df else df

    def _sides(n):
        def f(df):
            if "side" not in df:
                return df.head(2 * n)
            return pd.concat([df[df["side"] == "top"].head(n), df[df["side"] == "bottom"].head(n)])
        return f

    def _career_tails(df):
        d = df.sort_values("career_resid")
        return pd.concat([d.tail(5).iloc[::-1], d.head(5)])

    ident = lambda df: df
    paper_sections = [
        ("표 1: 팀 승수 회귀", "table1", None, [("table1", ident, 5)]),
        ("그림 2: 팀 생산성 잔차 분포", "table1", "fig2", [("fig2_ratio", ident, 5)]),
        ("그림 3: 조직문화 순위", "org_culture", "fig3", [("fig3_summary", ident, 30)]),
        ("그림 4: 팀 잔차 vs tcWAR", "tc_persistence", "fig4", [("fig4", _extremes, 20)]),
        ("시너지 지속성과 그림 5", "tc_persistence", "fig5", [("persistence", ident, 5), ("fig5_bars", ident, 30)]),
        ("표 3: PECOTA 대비 예측", "pecota", None, [("table3", ident, 20), ("oos_summary", ident, 5)]),
        ("그림 6: WAR− · WAR+ vs WAR", "player_eval", "fig6", [("tier_summary", ident, 5)]),
        ("그림 7: pcWAR vs WAR", "player_eval", "fig7", []),
        ("그림 8: 커리어 평균 pcWAR 순위", "player_eval", "fig8", [("fig8", _sides(10), 20)]),
        ("표 4: pcWAR 지속성 (식 16)", "pc_persistence", None, [("table4", ident, 20), ("table4_fit", ident, 5)]),
        ("그림 9: 커리어 보완성 초과승", "pc_persistence", "fig9", [("fig9", _career_tails, 10)]),
        ("그림 10: 나이-포지션 프로파일", "profiles", "fig10", [("slopes", lambda d: d.sort_values("slope_rank"), 12)]),
        ("그림 11: Intangibles 순위", "intangibles", "fig11", [("fig11", _sides(10), 20)]),
        ("그림 12: 단일 선수 (David Ross)", "ross", "fig12", [("player", ident, 3), ("fig12", ident, 25)]),
        ("표 5: 연봉 회귀 (식 17)", "salary", None, [("table5", lambda d: d[d["term"].astype(str).str.contains("Cum", case=False)] if "term" in d else d, 20), ("table5_fit", ident, 5)]),
    ]

    shown_checks = set()
    for section_title, mod_name, fig_stem, table_specs in paper_sections:
        res = results.get(mod_name)
        if res is None:
            continue
        sections.append(f"## {section_title}\n")
        if fig_stem and (out_path.parent / fig_dir_rel / f"{fig_stem}.png").exists():
            sections.append(f"![{section_title}]({fig_dir_rel}/{fig_stem}.png)\n")
        for table_name, sel, max_rows in table_specs:
            t = res.tables.get(table_name)
            if t is None or t.empty:
                continue
            try:
                t = sel(t)
            except (KeyError, AttributeError):
                pass  # selector does not fit this table version: show it unfiltered
            sections.append(f"`{mod_name}__{table_name}.csv`\n")
            sections.append(_markdown_table(t, max_rows=max_rows))
            sections.append("")
        # Checks are listed once, under the module's first section.
        if res.checks and mod_name not in shown_checks:
            shown_checks.add(mod_name)
            sections.append("**검증 (논문 기대 결과 대비)**\n")
            sections.append(_checks_table(res.checks))
            sections.append("")

    # Summary
    sections.append("## 검증 요약\n")
    total = sum(check_counts.values())
    sections.append(f"- ✅ 통과: {check_counts['passed']} / {total}")
    sections.append(f"- ❌ 실패: {check_counts['failed']} / {total}")
    sections.append(f"- 실데이터 필요: {check_counts['unverified']} / {total}")

    # Configuration
    sections.append("\n## 설정\n")
    sections.append(f"- war_source: {war_source}")
    sections.append(f"- seed: {seed}")
    sections.append(f"- n_boot: {cfg.get('n_boot', 'n/a')}")
    sections.append(f"- culture_lambda_source: {cfg.get('culture_lambda_source', 'prob')}")
    sections.append(f"- sign_convention: {cfg.get('sign_convention', 'consistent')}")
    sections.append(f"- factor_method: {cfg.get('factor_method', 'als')}")

    # Write file
    content = "\n".join(sections)
    out_path.write_text(content, encoding="utf-8")
    log.info(f"report written to {out_path}")
