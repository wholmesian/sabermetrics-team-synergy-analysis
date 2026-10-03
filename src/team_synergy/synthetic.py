"""Synthetic panel with known truth for validating the model code (spec section 9, step 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .model.network import build_blocks

ETA = {"bat": 0.57, "pit": 0.43}


def make_synthetic_panel(
    n_teams: int = 30,
    n_seasons: int = 10,
    roster: int = 45,
    rho: float = 0.3,
    trade_frac: float = 0.12,
    noise_sd: float = 0.3,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Generate a stint panel with known rho, lambda, c, tp (eqs. 1, 4, 7, 8; spec 9.1).

    Rosters: players persist; each season ~20% of each roster is released, vacancies
    are filled half by movers from other teams, half by new players, so repeated
    teammate history exists. ~trade_frac of player-seasons are split into two stints
    on different teams (identifies lambda later); kappa of a split is divided u:(1-u).
    weight = eta_g * tau, tau = kappa share within team-season and group
    (eta = 0.57 bat / 0.43 pit), so weights sum to 1 per team-season.

    Truth: lambda_n (sum 0, sd 1), per player-season c ~ N(0,.3), tp ~ N(0,1) shared by
    that player's stints; g_r = c + tp*lambda_n(r); e ~ N(0, noise_sd);
    y_true = Phi (g + e) per true block (history="past"), Phi = (I - rho A)^-1.

    Construction of war / team_wins (deviation from a naive T ~ N(35,10) draw, which
    would make OLS beta_hat attenuated through errors-in-variables): draw the team
    total Sw_nt ~ N(35,10) independent of y, set X_nt = Sw_nt - sum_r y_r,
    war_r = y_r + weight_r * X_nt (so sum war = Sw), team_wins = alpha + beta * X_nt
    with alpha=48, beta=1, no extra noise. Then eps = W - alpha - beta*Sw = -beta*sum y
    is independent of sum WAR, OLS recovers alpha, beta, and
    war - weight*(W-alpha)/beta = y_true exactly at the true (alpha, beta).
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    seasons = list(range(2000, 2000 + n_seasons))
    group_of: dict[int, str] = {}
    next_id = 0

    def new_player() -> int:
        nonlocal next_id
        group_of[next_id] = "bat" if rng.random() < 0.55 else "pit"
        next_id += 1
        return next_id - 1

    rosters = {t: [new_player() for _ in range(roster)] for t in teams}
    recs = []  # (player, season, team, kappa)
    for si, s in enumerate(seasons):
        if si > 0:
            released, kept = [], {}
            for t in teams:
                r = np.array(rosters[t])
                out = rng.random(len(r)) < 0.20
                kept[t] = list(r[~out])
                released += [(p, t) for p in r[out]]
            rng.shuffle(released)
            for t in teams:
                while len(kept[t]) < roster:
                    pick = None
                    if rng.random() < 0.5:
                        for j, (p, ot) in enumerate(released):
                            if ot != t:
                                pick = released.pop(j)[0]
                                break
                    kept[t].append(pick if pick is not None else new_player())
            rosters = kept
        for t in teams:
            for p in rosters[t]:
                k = rng.gamma(2.0, 1.0)
                if rng.random() < trade_frac:
                    ot = teams[rng.choice([i for i in range(n_teams) if teams[i] != t])]
                    u = rng.uniform(0.2, 0.8)
                    recs.append((p, s, t, k * u))
                    recs.append((p, s, ot, k * (1 - u)))
                else:
                    recs.append((p, s, t, k))
    df = pd.DataFrame(recs, columns=["player_id", "season", "franch_id", "kappa"])
    df["player_id"] = "p" + df["player_id"].astype(str).str.zfill(5)
    df["group"] = df["player_id"].str[1:].astype(int).map(group_of)
    df["tau"] = df["kappa"] / df.groupby(["franch_id", "season", "group"])["kappa"].transform("sum")
    df["eta"] = df["group"].map(ETA)
    df["weight"] = df["eta"] * df["tau"]
    slot_ps = df[["player_id", "season"]].drop_duplicates()
    slot_ps["lineup_slot_mean"] = rng.uniform(1, 9, len(slot_ps))
    df = df.merge(slot_ps, on=["player_id", "season"], how="left")
    df.loc[df["group"] == "pit", "lineup_slot_mean"] = np.nan
    df = df.sort_values(["franch_id", "season", "player_id"]).reset_index(drop=True)

    # truth factors
    lam = rng.normal(size=n_teams)
    lam = (lam - lam.mean()) / lam.std()
    lam = pd.Series(lam, index=teams, name="lam")
    fac = df[["player_id", "season"]].drop_duplicates().reset_index(drop=True)
    fac["c"] = rng.normal(0, 0.3, len(fac))
    fac["tp"] = rng.normal(0, 1.0, len(fac))
    f = df[["player_id", "season"]].merge(fac, on=["player_id", "season"], how="left")
    g = f["c"].to_numpy() + f["tp"].to_numpy() * df["franch_id"].map(lam).to_numpy()
    u = g + rng.normal(0, noise_sd, len(df))

    blocks = build_blocks(df, history="past")
    y = np.empty(len(df))
    for b in blocks:
        y[b.idx] = np.linalg.solve(np.eye(len(b.idx)) - rho * b.A, u[b.idx])

    alpha, beta = 48.0, 1.0
    key = [df["franch_id"], df["season"]]
    sy = pd.Series(y).groupby(key).transform("sum").to_numpy()
    sw_tab = df.groupby(["franch_id", "season"]).size().rename("n").to_frame()
    sw_tab["Sw"] = rng.normal(35, 10, len(sw_tab))
    Sw = df.join(sw_tab["Sw"], on=["franch_id", "season"])["Sw"].to_numpy()
    X = Sw - sy
    df["war"] = y + df["weight"].to_numpy() * X
    df["team_wins"] = alpha + beta * X

    # True out-of-own-row effect of each stint's shock on teammates (used by the salary
    # generator): colsum(Phi)_i - 1. Simplification: Phi_ii ~ 1 (own feedback loops ignored).
    colsum = np.empty(len(df))
    for b in blocks:
        colsum[b.idx] = np.linalg.solve((np.eye(len(b.idx)) - rho * b.A).T, np.ones(len(b.idx)))
    truth = dict(rho=rho, alpha=alpha, beta=beta, lam=lam, factors=fac, y_true=y,
                 u=u, pc_out_true=u * (colsum - 1.0))
    df, truth["player_info"] = _add_covariates(df, n_teams, seed)
    cols = schema.PANEL_COLUMNS
    return df[cols].copy(), truth


BAT_POS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "UT"]
BAT_POS_P = np.array([.10, .09, .09, .09, .09, .09, .09, .09, .07, .20])


def _add_covariates(df: pd.DataFrame, n_teams: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach schema.PANEL_COLUMNS covariates (spec section 1) from a SEPARATE rng stream.

    Uses ``default_rng([seed, 1])`` so the model columns drawn from ``default_rng(seed)``
    are unchanged. Covariates are plausible, not calibrated: games per season by position,
    pa/douts/pouts proportional to games, stable position per player (10% of player-seasons
    draw another position of the same group), debut age 21-27 (players on the initial
    rosters have already played 0-7 seasons before the first panel season), mlb_exp and
    team_exp = cumulative games before the season (in-sample, left-censored like the real
    build), one manager per franchise with a change probability 0.15 per season, first half
    of the franchises in the AL. Returns (df with covariates, player_info table).
    """
    rng = np.random.default_rng([seed, 1])
    df = df.copy()
    ps = df[["player_id", "season", "group"]].drop_duplicates(["player_id", "season"]).copy()
    pl = ps.groupby("player_id").agg(group=("group", "first"), first=("season", "min")).reset_index()
    s0 = int(df["season"].min())
    n = len(pl)
    debut_age = rng.integers(21, 28, n)
    prior = np.where(pl["first"].to_numpy() == s0, rng.integers(0, 8, n), 0)
    pl["birth_year"] = pl["first"] - prior - debut_age
    pl["debut_season"] = pl["first"] - prior
    pl["base_pos"] = np.where(pl["group"] == "bat", rng.choice(BAT_POS, n, p=BAT_POS_P),
                              np.where(rng.random(n) < 0.4, "SP", "RP"))
    ps = ps.merge(pl[["player_id", "base_pos", "birth_year"]], on="player_id")
    m = len(ps)
    alt = np.where(ps["group"] == "bat", rng.choice(BAT_POS, m, p=BAT_POS_P),
                   np.where(rng.random(m) < 0.4, "SP", "RP"))
    ps["pos"] = np.where(rng.random(m) < 0.10, alt, ps["base_pos"])
    ps["games"] = np.select(
        [ps["pos"] == "SP", ps["pos"] == "RP"],
        [np.clip(rng.normal(30, 3, m), 5, 38), np.clip(rng.normal(55, 12, m), 5, 85)],
        np.clip(rng.normal(115, 35, m), 10, 162)).round()
    ps = ps.drop(columns=["group", "base_pos"])
    df = df.merge(ps, on=["player_id", "season"], how="left")
    share = df["kappa"] / df.groupby(["player_id", "season"])["kappa"].transform("sum")
    df["g_stint"] = (df["games"] * share).clip(lower=1.0)
    df["age"] = df["season"] - df["birth_year"]
    bat = df["group"] == "bat"
    df["role"] = np.where(bat, None, df["pos"])
    df["pa"] = np.where(bat, (df["g_stint"] * rng.uniform(3.0, 4.2, len(df))).round(), 0.0)
    df["douts"] = np.where(bat, (df["g_stint"] * rng.uniform(15, 25, len(df))).round(), 0.0)
    df["pouts"] = np.where(bat, 0.0, (df["g_stint"] * np.where(df["pos"] == "SP", 15, 3.5)).round())
    df["l"] = np.where(bat, rng.uniform(0.5, 1.5, len(df)), np.nan)
    df["d"] = np.where(bat, rng.uniform(0.6, 1.4, len(df)), 1.0)
    # lineup_slot_mean (model-stream column) is kept untouched.
    # experience: cumulative games before the season (player) / before the season on the franchise
    gp = df.groupby(["player_id", "season"])["g_stint"].transform("sum")
    pse = df[["player_id", "season"]].assign(g=gp).drop_duplicates(["player_id", "season"])
    pse = pse.sort_values(["player_id", "season"])
    pse["mlb_exp"] = pse.groupby("player_id")["g"].cumsum() - pse["g"]
    df = df.merge(pse[["player_id", "season", "mlb_exp"]], on=["player_id", "season"], how="left")
    df = df.sort_values(["player_id", "franch_id", "season"])
    df["team_exp"] = df.groupby(["player_id", "franch_id"])["g_stint"].cumsum() - df["g_stint"]
    df = df.sort_values(["franch_id", "season", "player_id"]).reset_index(drop=True)
    # managers and leagues
    teams = sorted(df["franch_id"].unique())
    seasons = sorted(df["season"].unique())
    mrows = []
    for t in teams:
        k = 0
        for s in seasons:
            if s != seasons[0] and rng.random() < 0.15:
                k += 1
            mrows.append((t, s, f"m_{t}_{k}"))
    mg = pd.DataFrame(mrows, columns=["franch_id", "season", "manager_id"])
    df = df.merge(mg, on=["franch_id", "season"], how="left")
    df["lg"] = df["franch_id"].map({t: ("AL" if i < len(teams) // 2 else "NL") for i, t in enumerate(teams)})
    info = pl[["player_id", "birth_year", "debut_season"]].copy()
    return df, info


def _salary_class(n_prior: np.ndarray) -> np.ndarray:
    """Eq. (17) service class from in-sample prior seasons: 0 pre-arb (0-2), 1 arb (3-5), 2 FA (6+)."""
    return np.where(n_prior <= 2, 0, np.where(n_prior <= 5, 1, 2))


def make_synthetic_extras(df: pd.DataFrame, truth: dict, seed: int = 0, salary_coverage: float = 0.85) -> dict:
    """Side tables for the paper-reproduction analyses (spec sections 1, 8).

    Returns dict with
    - ``players``: player_id, name ("Player 0123"), birth_year, debut_season;
    - ``salaries``: player_id, season, salary (dollars; ``salary_coverage`` of player-seasons).
      ln S = theta_c + pi_c*CumWAR_{t-1} (+ 0.015 CumWAR^2 for CumWAR>0: increasing returns)
      + psi*Cum true pc contribution_{t-1} (FA class only, psi = 0.20) + 0.05*(age-27)
      - 0.002*(age-27)^2 + player FE N(0, 0.3^2) + N(0, 0.3^2), class c from in-sample prior
      seasons (eq. 17; mu_c teamExp and the position slopes are omitted: simplification);
    - ``pecota``: franch_id, season, pecota_w for seasons >= first + 10 (paper: 2008-2016 vs 1998)
      = 0.5*W_{t-1} + 0.5*(alpha + Sw_t) + N(0, 3^2), Sw_t = roster WAR (so tcWAR, which is
      W - alpha - Sw, carries information beyond it).
    Separate rng stream ``default_rng([seed, 2])``; synthetic data only (not calibrated).
    """
    rng = np.random.default_rng([seed, 2])
    players = truth["player_info"].copy()
    ids = players["player_id"].str[1:].astype(int)
    players.insert(1, "name", ["Player " + str(i).zfill(4) for i in ids])
    players = players[["player_id", "name", "birth_year", "debut_season"]].reset_index(drop=True)

    d = df.assign(pc_true=truth["pc_out_true"])
    ps = d.groupby(["player_id", "season"], as_index=False).agg(
        war=("war", "sum"), pc=("pc_true", "sum"), age=("age", "first"), mlb_exp=("mlb_exp", "first"))
    ps = ps.sort_values(["player_id", "season"]).reset_index(drop=True)
    g = ps.groupby("player_id")
    ps["n_prior"] = g.cumcount()
    ps["cum_war"] = g["war"].cumsum() - ps["war"]
    ps["cum_pc"] = g["pc"].cumsum() - ps["pc"]
    cls = _salary_class(ps["n_prior"].to_numpy())
    theta = np.array([13.0, 14.0, 14.6])[cls]
    pi = np.array([0.10, 0.20, 0.30])[cls]
    cw = ps["cum_war"].to_numpy()
    psi = np.where(cls == 2, 0.20, 0.0)
    fe = pd.Series(rng.normal(0, 0.3, len(players)), index=players["player_id"])
    a = ps["age"].to_numpy() - 27.0
    lns = (theta + pi * cw + 0.015 * np.where(cw > 0, cw, 0.0) ** 2 + psi * ps["cum_pc"].to_numpy()
           + 0.05 * a - 0.002 * a ** 2 + ps["player_id"].map(fe).to_numpy() + rng.normal(0, 0.3, len(ps)))
    ps["salary"] = np.exp(lns)
    keep = rng.random(len(ps)) < salary_coverage
    salaries = ps.loc[keep, ["player_id", "season", "salary"]].reset_index(drop=True)

    tm = d.groupby(["franch_id", "season"], as_index=False).agg(Sw=("war", "sum"), W=("team_wins", "first"))
    tm = tm.sort_values(["franch_id", "season"])
    tm["W_lag"] = tm.groupby("franch_id")["W"].shift(1)
    first = int(tm["season"].min())
    tm = tm[tm["season"] >= first + 10].copy()
    tm["pecota_w"] = (0.5 * tm["W_lag"] + 0.5 * (truth["alpha"] + tm["Sw"])
                      + rng.normal(0, 3.0, len(tm)))
    pecota = tm[["franch_id", "season", "pecota_w"]].reset_index(drop=True)
    return dict(players=players, salaries=salaries, pecota=pecota)
