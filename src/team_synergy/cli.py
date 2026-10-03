"""Spec section 9: command-line interface (``synergy build | estimate | analyze``)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

log = logging.getLogger("team_synergy")


def _parse_sets(items) -> dict:
    """Turn ``['a.b=1', 'c=x']`` into a nested override dict (values via yaml.safe_load)."""
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, val = item.split("=", 1)
        d = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return out


def _resolve(cfg: dict, key: str) -> Path:
    from team_synergy.config import _find_repo_root
    return (_find_repo_root() / cfg["paths"][key]).resolve()


def _load_cfg(args, **extra) -> dict:
    from team_synergy.config import load_config
    overrides = _parse_sets(getattr(args, "set", None))
    cfg = load_config(args.config, overrides=overrides)
    cfg.update({k: v for k, v in extra.items() if v is not None})
    return cfg


def _cmd_build(args) -> int:
    from team_synergy.build.panel import build_from_raw, check_panel, save_panel, save_side_tables
    from team_synergy.io import MissingRawDataError

    cfg = _load_cfg(args, war_source=args.war)
    args.war = cfg["war_source"]
    try:
        panel, side = build_from_raw(cfg, args.war, return_side_tables=True)
    except MissingRawDataError as e:
        # Spec section 0: no workaround for missing raw data; report what is expected.
        print(str(e))
        return 2
    report = check_panel(panel, cfg, strict=True)
    path = save_panel(panel, _resolve(cfg, "processed") / f"panel_{args.war}.parquet")
    save_side_tables(side, _resolve(cfg, "processed"))
    log.info("panel: %d stints, %d team-seasons -> %s", len(panel), report["n_team_seasons"], path)
    return 0


def _culture_lines(res: dict, cfg: dict) -> list[str]:
    lw = res["lambda_windows"]
    last = lw[lw["window_end"] == lw["window_end"].max()].sort_values("rank", ascending=False)
    k = min(5, len(last))
    fmt = lambda df: ", ".join(f"{r.franch_id} ({r.rank:.0f})" for r in df.itertuples())
    return [f"culture rank (source={cfg.get('culture_lambda_source', 'prob')}) top {k}: {fmt(last.head(k))}",
            f"culture rank bottom {k}: {fmt(last.tail(k)[::-1])}"]


def _summary(res: dict, cfg: dict) -> str:
    s = res["window_summary"].iloc[-1]
    ok1, ok2 = s["ratio"] < 1, s["corr_tc_eps"] > 0
    lines = [
        f"window [..{int(s['window_end'])}]: n_stints={int(s['n_stints'])}, "
        f"sign_convention={cfg.get('sign_convention')}, factor_method={cfg.get('factor_method')}",
        f"alpha={s['alpha']:.2f} beta={s['beta']:.3f} rho={s['rho']:.3f} "
        f"(SE {s['rho_se']:.3f}, at_bound={bool(s['at_bound'])})",
        f"factor: iter={int(s['factor_iter'])} converged={bool(s['factor_converged'])} "
        f"explained_share={s['explained_share']:.4f} (als iter={int(s['als_iter'])}, "
        f"prob iter={int(s['prob_iter'])} converged={bool(s['prob_converged'])})",
        f"diagnostic 1 sd(eps WAR-)/sd(eps WAR) = {s['ratio']:.3f} (expect < 1): {'PASS' if ok1 else 'FAIL'}",
        f"diagnostic 2 corr(tcWAR, eps) = {s['corr_tc_eps']:.3f} (expect > 0): {'PASS' if ok2 else 'FAIL'}",
    ]
    return "\n".join(lines + _culture_lines(res, cfg))


def _cmd_estimate(args) -> int:
    from team_synergy.model.recursive import run_recursive, save_outputs

    cfg = _load_cfg(args, war_source=args.war, seed=args.seed)
    args.war = cfg["war_source"]
    s0, s1 = (args.seasons if args.seasons else (cfg["seasons"]["start"], cfg["seasons"]["end"]))
    if args.synthetic:
        from team_synergy.synthetic import make_synthetic_panel
        stints, _ = make_synthetic_panel(n_teams=args.synthetic_teams, n_seasons=s1 - s0 + 1,
                                         roster=args.synthetic_roster, seed=cfg["seed"])
        stints["season"] = stints["season"] - stints["season"].min() + s0  # label like real windows
        out_dir = Path(args.out_dir) if args.out_dir else _resolve(cfg, "processed") / "synthetic"
    else:
        import pandas as pd
        path = Path(args.panel) if args.panel else _resolve(cfg, "processed") / f"panel_{args.war}.parquet"
        if not path.exists():
            print(f"panel not found: {path}\nrun `synergy build --war {args.war}` first.")
            return 2
        stints = pd.read_parquet(path)
        stints = stints[(stints["season"] >= s0) & (stints["season"] <= s1)]
        out_dir = Path(args.out_dir) if args.out_dir else _resolve(cfg, "processed") / args.war
    last = int(stints["season"].max())
    res = run_recursive(stints, cfg, min_end=last if args.full_only else None, n_jobs=args.n_jobs)
    out = save_outputs(res, out_dir, cfg, "synthetic" if args.synthetic else args.war)
    log.info("outputs written to %s", out)
    print(_summary(res, cfg))
    return 0


def _cmd_analyze(args) -> int:
    import time
    from team_synergy.analysis.inputs import load_inputs, synthetic_inputs
    from team_synergy.analysis.runner import run_all, save_results, MODULES
    from team_synergy.viz.figures import save_all as save_figures
    from team_synergy.analysis.report import write_report

    cfg = _load_cfg(args)

    # Load inputs
    if args.synthetic:
        log.info(f"generating synthetic inputs (n_seasons={args.synthetic_seasons}, ~1.5 min at 19)...")
        t0 = time.time()
        inputs = synthetic_inputs(seed=args.seed, n_seasons=args.synthetic_seasons, cfg=cfg,
                                  n_teams=args.synthetic_teams, roster=args.synthetic_roster)
        log.info(f"synthetic inputs ready in {time.time() - t0:.1f}s")
        war_source = "synthetic"
        out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/synthetic")
    else:
        war_source = args.war if args.war else cfg.get("war_source")
        processed_dir = Path(args.processed_dir) if args.processed_dir else _resolve(cfg, "processed") / war_source
        if not (processed_dir / "metrics_full.parquet").exists():
            print(f"inputs not found in {processed_dir}")
            print(f"run `synergy build --war {war_source}` then `synergy estimate --war {war_source}` first.")
            return 2
        try:
            inputs = load_inputs(processed_dir, war_source, cfg=cfg)
        except FileNotFoundError as e:
            print(str(e))
            return 2
        out_dir = Path(args.out_dir) if args.out_dir else _resolve(cfg, "processed") / war_source

    # Run analyses
    only = args.only.split(",") if args.only else None
    t0 = time.time()
    inputs.cfg["n_boot"] = args.n_boot  # recorded in REPORT.md settings
    results = run_all(inputs, n_boot=args.n_boot, seed=args.seed, only=only)
    log.info(f"all analyses completed in {time.time() - t0:.1f}s")

    # Save results
    save_results(results, out_dir)

    # Generate figures
    fig_dir = out_dir / "figures"
    fig_paths = save_figures(results, fig_dir)
    log.info(f"saved {len(fig_paths)} figures")

    # Write report
    report_path = out_dir / "REPORT.md"
    write_report(results, inputs, report_path, fig_dir_rel="figures")

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

    print(f"\nResults written to {out_dir}")
    print(f"Report: {report_path}")
    print(f"Checks: {check_counts['passed']}✅ {check_counts['failed']}❌ {check_counts['unverified']} 실데이터 필요")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="synergy", description="Brave et al. (2019) team synergy model.")
    sub = p.add_subparsers(dest="command")

    def common(sp, war=True):
        if war:
            sp.add_argument("--war", choices=["fwar", "bwar"], default=None,
                            help="WAR source (default: war_source in the config)")
        sp.add_argument("--config", default="config/default.yaml")
        sp.add_argument("--set", nargs="+", action="append", default=None, metavar="KEY=VALUE",
                        help="override dotted config keys, e.g. --set sign_convention=paper_literal")

    b = sub.add_parser("build", help="Build panel from raw data (spec section 1).")
    common(b)
    e = sub.add_parser("estimate", help="Recursive estimation (spec sections 3-7).")
    common(e)
    e.add_argument("--synthetic", action="store_true")
    e.add_argument("--seed", type=int, default=None)
    e.add_argument("--seasons", nargs=2, type=int, metavar=("A", "B"))
    e.add_argument("--full-only", action="store_true", help="only the full-sample window")
    e.add_argument("--out-dir", default=None)
    e.add_argument("--panel", default=None, help="panel parquet (default data/processed/panel_<war>.parquet)")
    e.add_argument("--n-jobs", type=int, default=1)
    e.add_argument("--synthetic-teams", type=int, default=30)
    e.add_argument("--synthetic-roster", type=int, default=45)
    a = sub.add_parser("analyze", help="Tables and figures (spec section 8).")
    common(a, war=False)
    a.add_argument("--war", choices=["fwar", "bwar"], default=None,
                   help="WAR source (default: war_source in config; ignored with --synthetic)")
    a.add_argument("--synthetic", action="store_true", help="use synthetic data")
    a.add_argument("--seed", type=int, default=0, help="random seed")
    a.add_argument("--n-boot", type=int, default=500, help="bootstrap replicates")
    a.add_argument("--only", default=None, help="comma-separated module names (e.g. table1,org_culture)")
    a.add_argument("--processed-dir", default=None, help="processed data directory")
    a.add_argument("--out-dir", default=None, help="output directory")
    a.add_argument("--synthetic-seasons", type=int, default=19)
    a.add_argument("--synthetic-teams", type=int, default=30)
    a.add_argument("--synthetic-roster", type=int, default=45)
    return p


def main(argv=None) -> int:
    """CLI entry point; returns the exit code (0 ok, 1 not implemented/error, 2 missing data)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "set", None):
        args.set = [x for grp in args.set for x in grp]
    if args.command is None:
        parser.print_help()
        return 0
    return {"build": _cmd_build, "estimate": _cmd_estimate, "analyze": _cmd_analyze}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
