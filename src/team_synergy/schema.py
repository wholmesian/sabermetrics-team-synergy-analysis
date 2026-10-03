"""Shared column contracts between io/, build/ and model/ (spec §1, §9).

The stint table is the unit of analysis: one row per (player, season, franchise).
`model/*` only requires MODEL_COLUMNS, so the synthetic generator and the real
panel builder both emit (at least) those columns.
"""

# Minimal columns every model step needs.
MODEL_COLUMNS = [
    "player_id",   # Lahman playerID (str)
    "season",      # int
    "franch_id",   # canonical 30-franchise code, see FRANCHISE_MAP
    "group",       # "bat" (position player) | "pit" (pitcher)
    "war",         # stint WAR (fWAR allocated by kappa, or bWAR as-is)
    "kappa",       # play intensity, spec §2 / paper κ_it
    "weight",      # appearance weight eta * tau, paper eq. (7)
    "team_wins",   # W_nt of the stint's team-season
]

# Full panel produced by build/panel.py (spec §1).
PANEL_COLUMNS = MODEL_COLUMNS + [
    "role",              # "SP" | "RP" for pitchers, None for position players
    "pa", "douts", "pouts",
    "l", "d",            # lineup / defensive weights, paper eqs. (5)-(6)
    "tau", "eta",
    "pos",               # primary position: C,1B,2B,3B,SS,LF,CF,RF,DH,UT,SP,RP
    "lineup_slot_mean",  # start-weighted mean batting slot (NaN if no starts)
    "age", "mlb_exp", "team_exp",
    "manager_id", "lg",
]

# Lahman franchID -> canonical code (spec §1).
FRANCHISE_MAP = {"MON": "WSN", "FLA": "MIA", "ANA": "LAA", "TBD": "TBR"}

FIELD_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
