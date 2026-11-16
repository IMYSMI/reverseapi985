"""
API RP 985 Refractory Thermal Calculator & Reverse Evolutionary Designer
Run locally with: streamlit run app.py
"""
from math import exp, log, pi, sqrt
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. CORE API RP 985 CALCULATION ENGINE (unchanged)
# ==========================================
SIGMA = 0.1714e-8
G = 32.174 * 3600**2

Z = [(500, 23.60), (1000, 14.85), (1500, 14.13), (2000, 15.61)]
IC = [(500, 3.32), (1000, 3.02), (1500, 3.24), (2000, 3.60)]
IB = [(500, 0.57), (1000, 0.80), (1500, 1.13), (2000, 1.60)]

LININGS = {
    1: ("Lining 1 · 1 layer (Castable X)", [("Castable X", 6.0, [(0, 4.0)])]),
    2: ("Lining 2 · 1 layer (Castable Y)", [("Castable Y", 6.0, [(500, 0.25), (1000, 0.50), (1500, 1.0), (2000, 2.0)])]),
    3: ("Lining 3 · 2 layers (Castable Z + Insulating Cast)", [("Castable Z", 6.0, Z), ("Insulating Cast", 4.0, IC)]),
    4: ("Lining 4 · 3 layers (Castable Z + Insul. Cast + Board)", [("Castable Z", 6.0, Z), ("Insulating Cast", 4.0, IC), ("Insulation board", 1.5, IB)]),
    5: ("Lining 5 · 3 layers (extended to 2500 °F)", [
        ("Castable Z ext", 6.0, [(500, 23.60), (1000, 14.85), (2000, 15.61), (2500, 19.0)]),
        ("Insulating Cast ext", 4.0, [(0, 4.05), (1000, 3.02), (2000, 3.60), (2500, 4.0)]),
        ("Insulation board ext", 1.5, [(0, 0.36), (500, 0.57), (1000, 0.80), (2000, 1.60)]),
    ]),
}

# Upper temperature limits (°F) of the experimental k-data per layer (API RP 985 Section 8.3)
LAYER_MAX_TEMP_F = {
    1: [2500.0],
    2: [2000.0],
    3: [2000.0, 2000.0],
    4: [2000.0, 2000.0, 2000.0],
    5: [2500.0, 2500.0, 2000.0],
}

# (geom, d_dict, lining_id, thick_list, eps, t0, wind, t1, expected_ts, expected_q)
PRESETS = {
    "Custom input": None,
    "Case 1: Horiz. Cylinder | Lining 1 | 0 mph": ("hcyl", {"D": 8.0}, 1, [6.0], 0.8, 100.0, 0.0, 2000.0, 413, 991),
    "Case 2: Horiz. Cylinder | Lining 1 | 10 mph": ("hcyl", {"D": 8.0}, 1, [6.0], 0.8, 100.0, 10.0, 2000.0, 385, 1008),
    "Case 3: Horiz. Cylinder | Lining 1 | -5 °F": ("hcyl", {"D": 8.0}, 1, [6.0], 0.8, -5.0, 10.0, 2000.0, 330, 1042),
    "Case 4: Horiz. Cylinder | Lining 2 (variable k)": ("hcyl", {"D": 8.0}, 2, [6.0], 0.8, -5.0, 10.0, 2000.0, 78, 208),
    "Case 5: Vert. Cylinder | Lining 2 | 0 mph": ("vcyl", {"D": 6.0, "h": 30.0}, 2, [6.0], 0.8, 100.0, 0.0, 2000.0, 198, 201),
    "Case 6: Vert. Cylinder | Lining 3": ("vcyl", {"D": 6.0, "h": 30.0}, 3, [6.0, 4.0], 0.8, 100.0, 0.0, 2000.0, 388, 873),
    "Case 7: Vert. Cylinder | Lining 4": ("vcyl", {"D": 6.0, "h": 30.0}, 4, [6.0, 4.0, 1.5], 0.8, 100.0, 0.0, 2000.0, 272, 419),
    "Case 8: Vert. Cylinder | Lining 5 | 2500 °F": ("vcyl", {"D": 30.0, "h": 8.0}, 5, [6.0, 4.0, 1.5], 0.8, -5.0, 10.0, 2500.0, 235, 649),
    "Case 9: Vertical Wall (30x6 ft) | Lining 4": ("wall", {"h": 30.0, "l": 6.0}, 4, [6.0, 4.0, 1.5], 0.8, 100.0, 0.0, 2000.0, 283, 457),
    "Case 10: Vertical Wall (6x30 ft) | Lining 4": ("wall", {"h": 6.0, "l": 30.0}, 4, [6.0, 4.0, 1.5], 0.8, 100.0, 0.0, 2000.0, 281, 458),
    "Case 11: Vertical Wall | Lining 4 | -5 °F, 10 mph": ("wall", {"h": 6.0, "l": 30.0}, 4, [6.0, 4.0, 1.5], 0.8, -5.0, 10.0, 2000.0, 161, 475),
    "Case 12: Vertical Wall | Lining 5 | ε 0.15": ("wall", {"h": 6.0, "l": 30.0}, 5, [6.0, 4.0, 1.5], 0.15, -5.0, 10.0, 2500.0, 302, 651),
    "Case 13: Roof (6x30 ft) | Lining 4 | 0 mph": ("roof", {"l": 6.0, "w": 30.0}, 4, [6.0, 4.0, 1.5], 0.8, 100.0, 0.0, 2000.0, 267, 460),
    "Case 14: Roof (30x6 ft) | Lining 4 | 0 mph": ("roof", {"l": 30.0, "w": 6.0}, 4, [6.0, 4.0, 1.5], 0.8, 100.0, 0.0, 2000.0, 267, 460),
    "Case 15: Roof | Lining 4 | -5 °F, 10 mph": ("roof", {"l": 30.0, "w": 6.0}, 4, [6.0, 4.0, 1.5], 0.8, -5.0, 10.0, 2000.0, 158, 476),
    "Case 16: Roof | Lining 5 | 2500 °F, ε 0.15": ("roof", {"l": 30.0, "w": 6.0}, 5, [6.0, 4.0, 1.5], 0.15, -5.0, 10.0, 2500.0, 285, 654),
    "Case 17: Floor (6x30 ft) | Lining 4 | 0 mph": ("floor", {"l": 6.0, "w": 30.0}, 4, [6.0, 4.0, 1.5], 0.8, 100.0, 0.0, 2000.0, 317, 452),
    "Case 18: Floor (30x6 ft) | Lining 4 | 0 mph": ("floor", {"l": 30.0, "w": 6.0}, 4, [6.0, 4.0, 1.5], 0.8, 100.0, 0.0, 2000.0, 317, 452),
    "Case 19: Floor | Lining 4 | -5 °F, 10 mph": ("floor", {"l": 30.0, "w": 6.0}, 4, [6.0, 4.0, 1.5], 0.8, -5.0, 10.0, 2000.0, 167, 475),
    "Case 20: Floor | Lining 5 | 2500 °F, ε 0.15": ("floor", {"l": 30.0, "w": 6.0}, 5, [6.0, 4.0, 1.5], 0.15, -5.0, 10.0, 2500.0, 332, 647),
}


def k_poly(points):
    t, k = zip(*points)
    return np.poly1d(np.polyfit(t, k, min(3, len(points) - 1)))


def k_eff(poly, t1, t2):
    if abs(t2 - t1) < 1e-9:
        return max(float(poly(t1)), 1e-4)
    P = poly.integ()
    return max(float((P(t2) - P(t1)) / (t2 - t1)), 1e-4)


def air(tf):
    tk = max((tf - 32.0) / 1.8 + 273.15, 50.0)
    kf = 6.325e-6 * sqrt(tk) / (1.0 + 245.4 * (10.0 ** (-12.0 / tk)) / tk) * 241.77
    mu = 145.8 * tk * sqrt(tk) / (tk + 110.4) * 241.9e-7
    pr = max(0.7189 - tf * (1.6349e-4 - tf * (1.8106e-7 - 5.6617e-11 * tf)), 0.1)
    cp = max(0.24008 - tf * (1.2477e-6 - tf * (4.0489e-8 - 1.6088e-11 * tf)), 0.1)
    rho = 22.0493 / tk
    return kf, mu / rho, pr, cp, rho, 1.0 / (1.8 * tk), mu


def h_out(ts, t0, eps, wind, geom, d):
    Ts, To = max(ts + 459.67, 1.0), max(t0 + 459.67, 1.0)
    hr = SIGMA * eps * (Ts**3 + Ts**2 * To + Ts * To**2 + To**3)
    kf, nu, pr, cp, rho, beta, _ = air(0.5 * (ts + t0))
    dT = abs(Ts - To)
    V = max(wind, 0.0) * 5280.0
    Ra = lambda L: G * beta * rho * cp * dT * (L**3) / (nu * kf)

    if geom in ("wall", "vcyl"):
        Ln = d["h"]
        Nun = (0.825 + 0.387 * Ra(Ln)**(1/6) / (1 + (0.492 / pr)**(9/16))**(8/27))**2
    elif geom == "hcyl":
        Ln = d["D"]
        Nun = (0.60 + 0.387 * Ra(Ln)**(1/6) / (1 + (0.559 / pr)**(9/16))**(8/27))**2
    elif geom == "sphere":
        Ln = d["D"]
        Nun = 2.0 + 0.589 * Ra(Ln)**0.25 / (1 + (0.469 / pr)**(9/16))**(4/9)
    else:
        Ln = d["l"] * d["w"] / (2.0 * (d["l"] + d["w"]))
        r = Ra(Ln)
        heat_flow_up = (geom == "roof" and ts >= t0) or (geom == "floor" and ts < t0)
        Nun = (0.54 * r**0.25 if r < 1e7 else 0.15 * r**(1/3)) if heat_flow_up else 0.27 * r**0.25

    if geom in ("wall", "roof", "floor"):
        Lf = d["l"]
        Re = V * Lf / nu
        Nuf = (0.6744 * sqrt(Re) * pr**(1/3) / (1 + (0.0468 / pr)**(2/3))**0.25 if Re < 5e5
               else max((0.037 * Re**0.8 - 871.0) * pr**(1/3), 0.0))
    elif geom in ("hcyl", "vcyl"):
        Lf = d["D"]
        Re = V * Lf / nu
        Nuf = 0.3 + 0.62 * sqrt(Re) * pr**(1/3) / (1 + (0.4 / pr)**(2/3))**0.25 * (1 + (Re / 282000.0)**(5/8))**0.8
    else:
        Lf = d["D"]
        _, nu_inf, pr_inf, _, _, _, mu_inf = air(t0)
        _, _, _, _, _, _, mu_s = air(ts)
        Re_inf = V * Lf / nu_inf
        Nuf = 2.0 + (0.4 * sqrt(Re_inf) + 0.060 * Re_inf**(2/3)) * pr_inf**0.4 * (mu_inf / mu_s)**0.25

    if geom == "hcyl":
        Nu = 0.3 + ((Nuf - 0.3)**4 + (Nun - 0.3)**4)**0.25
        hc = Nu * kf / Lf
    elif geom == "sphere":
        Nu = 2.0 + (max(Nuf - 2.0, 0.0)**4 + max(Nun - 2.0, 0.0)**4)**0.25
        hc = Nu * kf / Lf
    else:
        j = 3.5 if geom in ("roof", "floor") else 3.0
        hc = ((Nuf * kf / Lf)**j + (Nun * kf / Ln)**j)**(1.0 / j)

    return hr + hc, hr, hc


# Precompute polynomials once for speed during evolutionary optimization
PRECOMPUTED_POLYS = {
    lid: [k_poly(p) for _, _, p in LININGS[lid][1]] for lid in LININGS
}


def solve_api985(geom, d, lining_id, thick, eps, t0, wind, t1):
    polys = PRECOMPUTED_POLYS[lining_id]
    n = len(polys)

    if geom in ("hcyl", "vcyl", "sphere") and sum(thick) >= d["D"] * 6.0:
        raise ValueError(f"Total refractory thickness ({sum(thick):.2f} in.) exceeds vessel inner radius ({d['D']*6.0:.2f} in.).")

    x = np.concatenate(([0.0], np.cumsum(thick, dtype=float)))
    if geom in ("hcyl", "vcyl", "sphere"):
        x = d["D"] * 6.0 - x[-1] + x

    ts_init = t0 + 0.1 * (t1 - t0)
    t = np.concatenate(([t1], t1 + (ts_init - t1) * (x[1:] - x[0]) / (x[-1] - x[0])))

    alpha, prev_err = 1.0, float("inf")
    for it in range(500):
        ke = [k_eff(polys[m], t[m + 1], t[m]) for m in range(n)]
        if geom in ("hcyl", "vcyl"):
            R = [x[-1] * log(x[m + 1] / x[m]) / ke[m] for m in range(n)]
        elif geom == "sphere":
            R = [(x[-1]**2) * (x[m + 1] - x[m]) / (ke[m] * x[m] * x[m + 1]) for m in range(n)]
        else:
            R = [(x[m + 1] - x[m]) / ke[m] for m in range(n)]

        h_tot, hr, hc = h_out(t[-1], t0, eps, wind, geom, d)
        Ro = 1.0 / h_tot
        q = (t1 - t0) / (sum(R) + Ro)

        new = [t1]
        for m in range(n):
            new.append(new[-1] - q * R[m])
        new = np.array(new, dtype=float)

        err = float(np.max(np.abs(new - t)))
        if err < 0.1:
            t = new
            ke = [k_eff(polys[m], t[m + 1], t[m]) for m in range(n)]
            return t, q, ke, R, h_tot, hr, hc, it + 1

        if err > prev_err and it > 3:
            alpha = max(0.25, alpha * 0.7)
        prev_err = err
        t = (1.0 - alpha) * t + alpha * new

    raise RuntimeError("Solver did not converge within 500 iterations.")


# ==========================================
# 2. EVOLUTIONARY OPTIMIZER (Differential Evolution - DE/rand/1/bin)
# ==========================================
def run_evolutionary_design(
    geom, d, eps, t0, wind, t1, target_ts,
    allowed_linings, max_total_thick, layer_bounds, fixed_layers,
    step_inc=0.0, pop_size=24, generations=35, enforce_temp_limits=True, seed=42
):
    """
    Runs Differential Evolution across each candidate lining system for 3 objectives:
      - 'thermal': minimize heat flux q subject to Ts <= target_ts and sum(thick) <= max_total_thick
      - 'thickness': minimize total thickness subject to Ts <= target_ts and sum(thick) <= max_total_thick
      - 'balanced': minimize weighted normalized (q, total_thick) subject to constraints
    Returns:
      categories_dict, all_ranked_solutions_list
    """
    rng = np.random.default_rng(seed)
    vessel_rad = d["D"] * 6.0 - 0.25 if geom in ("hcyl", "vcyl", "sphere") else 999.0
    eff_max_total = min(max_total_thick, vessel_rad)

    archive = {}  # key: (lining_id, rounded_thick_tuple) -> solution dict

    def snap_and_clamp(vec, lid):
        out = []
        for i, val in enumerate(vec):
            if fixed_layers.get((lid, i)) is not None:
                out.append(float(fixed_layers[(lid, i)]))
            else:
                lo, hi = layer_bounds.get((lid, i), (0.5, eff_max_total))
                v = float(np.clip(val, lo, hi))
                if step_inc > 0:
                    v = round(v / step_inc) * step_inc
                    v = float(np.clip(v, lo, hi))
                out.append(v)
        return out

    def evaluate_candidate(vec, lid):
        th = snap_and_clamp(vec, lid)
        tot_th = sum(th)
        key = (lid, tuple(round(x, 4) for x in th))
        if key in archive:
            return archive[key]

        if tot_th > vessel_rad or tot_th <= 0:
            res = {
                "valid": False, "feasible": False, "lining_id": lid,
                "lining_name": LININGS[lid][0], "thick": th, "total_thick": tot_th,
                "ts": 9999.0, "q": 999999.0, "temp_violation": 9999.0
            }
            archive[key] = res
            return res

        try:
            t_prof, q_flux, ke_list, R_list, h_tot, hr, hc, iters = solve_api985(
                geom, d, lid, th, eps, t0, wind, t1
            )
            ts = float(t_prof[-1])
            # Check API RP 985 Section 8.3 layer temperature data range
            max_temps = LAYER_MAX_TEMP_F[lid]
            temp_viol = 0.0
            if enforce_temp_limits:
                for m_idx, max_tm in enumerate(max_temps):
                    layer_hot = float(max(t_prof[m_idx], t_prof[m_idx + 1]))
                    if layer_hot > max_tm + 1.0:
                        temp_viol += (layer_hot - max_tm)

            ts_viol = max(0.0, ts - target_ts)
            th_viol = max(0.0, tot_th - eff_max_total)
            feasible = (ts_viol == 0.0) and (th_viol == 0.0) and (temp_viol == 0.0)

            res = {
                "valid": True,
                "feasible": feasible,
                "lining_id": lid,
                "lining_name": LININGS[lid][0],
                "thick": th,
                "total_thick": tot_th,
                "ts": ts,
                "q": float(q_flux),
                "t_prof": t_prof,
                "ke_list": ke_list,
                "R_list": R_list,
                "h_tot": h_tot,
                "hr": hr,
                "hc": hc,
                "ts_viol": ts_viol,
                "th_viol": th_viol,
                "temp_viol": temp_viol,
            }
        except Exception:
            res = {
                "valid": False, "feasible": False, "lining_id": lid,
                "lining_name": LININGS[lid][0], "thick": th, "total_thick": tot_th,
                "ts": 9999.0, "q": 999999.0, "ts_viol": 9999.0, "th_viol": 9999.0, "temp_viol": 9999.0
            }

        archive[key] = res
        return res

    def fitness(sol, mode):
        if not sol["valid"]:
            return 1e12
        # Heavy constraint-violation penalty (Deb's feasibility rules)
        penalty = (
            sol["ts_viol"] * 5000.0
            + sol["th_viol"] * 8000.0
            + sol["temp_viol"] * 10000.0
        )
        if penalty > 0:
            return 1e7 + penalty + sol["q"] + sol["total_thick"] * 50.0

        if mode == "thermal":
            return sol["q"] + 0.01 * sol["total_thick"]
        elif mode == "thickness":
            return sol["total_thick"] * 1000.0 + 0.005 * sol["q"]
        else:  # balanced
            q_norm = sol["q"] / 600.0
            th_norm = sol["total_thick"] / max(eff_max_total, 1.0)
            return 0.5 * q_norm + 0.5 * th_norm

    # Run DE for each allowed lining and each mode so the archive is richly populated
    F_mut, CR = 0.65, 0.85
    for lid in allowed_linings:
        n_layers = len(LININGS[lid][1])
        free_indices = [i for i in range(n_layers) if fixed_layers.get((lid, i)) is None]

        # If all layers are locked, just evaluate once
        if not free_indices:
            evaluate_candidate([fixed_layers[(lid, i)] for i in range(n_layers)], lid)
            continue

        for mode in ("thickness", "thermal", "balanced"):
            # Initialize population within bounds (biasing total thickness < eff_max_total)
            pop = []
            for p_idx in range(pop_size):
                ind = []
                scale = rng.uniform(0.25, 0.95)
                for i in range(n_layers):
                    if fixed_layers.get((lid, i)) is not None:
                        ind.append(float(fixed_layers[(lid, i)]))
                    else:
                        lo, hi = layer_bounds.get((lid, i), (0.5, eff_max_total))
                        hi_eff = min(hi, eff_max_total)
                        if p_idx == 0:
                            # Seed with standard API default thickness
                            ind.append(float(np.clip(LININGS[lid][1][i][1], lo, hi_eff)))
                        else:
                            ind.append(float(rng.uniform(lo, max(lo + 0.1, hi_eff * scale))))
                # Project if sum exceeds eff_max_total
                tot = sum(ind)
                if tot > eff_max_total:
                    fixed_sum = sum(ind[i] for i in range(n_layers) if i not in free_indices)
                    rem = max(0.1, eff_max_total - fixed_sum)
                    free_sum = sum(ind[i] for i in free_indices)
                    if free_sum > 0:
                        for i in free_indices:
                            lo, hi = layer_bounds.get((lid, i), (0.5, eff_max_total))
                            ind[i] = float(np.clip(ind[i] * (rem / free_sum), lo, hi))
                pop.append(np.array(snap_and_clamp(ind, lid), dtype=float))

            pop_evals = [evaluate_candidate(ind, lid) for ind in pop]
            pop_fits = [fitness(ev, mode) for ev in pop_evals]

            for _ in range(generations):
                for i in range(pop_size):
                    idxs = [idx for idx in range(pop_size) if idx != i]
                    a, b, c = pop[rng.choice(idxs, 3, replace=False)]
                    mutant = a + F_mut * (b - c)
                    j_rand = rng.choice(free_indices)
                    trial = np.copy(pop[i])
                    for j in free_indices:
                        if rng.random() < CR or j == j_rand:
                            trial[j] = mutant[j]

                    # Repair total thickness if trial exceeds eff_max_total
                    trial = np.array(snap_and_clamp(trial, lid), dtype=float)
                    if np.sum(trial) > eff_max_total:
                        fixed_sum = sum(trial[k] for k in range(n_layers) if k not in free_indices)
                        rem = max(0.1, eff_max_total - fixed_sum)
                        free_sum = sum(trial[k] for k in free_indices)
                        if free_sum > 0:
                            for k in free_indices:
                                lo, hi = layer_bounds.get((lid, k), (0.5, eff_max_total))
                                trial[k] = float(np.clip(trial[k] * (rem / free_sum), lo, hi))
                        trial = np.array(snap_and_clamp(trial, lid), dtype=float)

                    trial_eval = evaluate_candidate(trial, lid)
                    trial_fit = fitness(trial_eval, mode)
                    if trial_fit <= pop_fits[i]:
                        pop[i] = trial
                        pop_evals[i] = trial_eval
                        pop_fits[i] = trial_fit

    # Filter valid evaluations
    valid_sols = [s for s in archive.values() if s["valid"]]
    feasible_sols = [s for s in valid_sols if s["feasible"]]

    pool = feasible_sols if feasible_sols else valid_sols
    if not pool:
        return None, []

    # Compute normalized Utopia distance for balanced score across the pool
    q_vals = np.array([s["q"] for s in pool])
    th_vals = np.array([s["total_thick"] for s in pool])
    q_min, q_max = float(np.min(q_vals)), float(np.max(q_vals))
    th_min, th_max = float(np.min(th_vals)), float(np.max(th_vals))

    for s in pool:
        qn = (s["q"] - q_min) / max(q_max - q_min, 1e-6)
        tn = (s["total_thick"] - th_min) / max(th_max - th_min, 1e-6)
        penalty_term = (s["ts_viol"] + s["th_viol"] + s["temp_viol"]) * 10.0
        # Euclidean distance to Utopia point (0, 0)
        s["balanced_score"] = sqrt(0.5 * qn**2 + 0.5 * tn**2) + penalty_term

    # Select the 3 Category Winners
    best_thermal = min(pool, key=lambda s: (not s["feasible"], s["temp_viol"] + s["ts_viol"] + s["th_viol"], s["q"], s["total_thick"]))
    best_thick = min(pool, key=lambda s: (not s["feasible"], s["temp_viol"] + s["ts_viol"] + s["th_viol"], s["total_thick"], s["q"]))
    best_balanced = min(pool, key=lambda s: (not s["feasible"], s["balanced_score"]))

    categories = {
        "thermal": best_thermal,
        "thickness": best_thick,
        "balanced": best_balanced,
    }

    # Build diverse ranked solutions list (best per lining per objective + top Pareto candidates)
    ranked_pool = sorted(pool, key=lambda s: (not s["feasible"], s["balanced_score"]))
    # Deduplicate near-identical thicknesses per lining (within 0.15 in.) so table shows distinct options
    distinct_ranked = []
    for cand in ranked_pool:
        is_dup = False
        for existing in distinct_ranked:
            if existing["lining_id"] == cand["lining_id"]:
                max_diff = max(abs(a - b) for a, b in zip(existing["thick"], cand["thick"]))
                if max_diff < 0.15:
                    is_dup = True
                    break
        if not is_dup:
            distinct_ranked.append(cand)
        if len(distinct_ranked) >= 25:
            break

    # Ensure the 3 category winners are always present in distinct_ranked
    for win in (best_thermal, best_thick, best_balanced):
        if not any(w["lining_id"] == win["lining_id"] and w["thick"] == win["thick"] for w in distinct_ranked):
            distinct_ranked.insert(0, win)

    for idx, sol in enumerate(distinct_ranked):
        sol["rank"] = idx + 1

    return categories, distinct_ranked


# ==========================================
# 3. UNIT CONVERSION LAYER (UI only - engine works in US units)
# ==========================================
FT_M = 0.3048            # 1 ft  = 0.3048 m
IN_MM = 25.4             # 1 in  = 25.4 mm
MPH_MS = 0.44704         # 1 mph = 0.44704 m/s
BTU_HFT2_TO_WM2 = 3.154591      # Btu/(h·ft²)        -> W/m²
BTU_HFT_TO_WM = 0.961519        # Btu/(h·ft)         -> W/m
H_TO_SI = 5.678263              # Btu/(h·ft²·°F)     -> W/(m²·K)
K_TO_SI = 0.1442279             # Btu·in/(h·ft²·°F)  -> W/(m·K)
R_TO_SI = 0.1761102             # h·ft²·°F/Btu      -> m²·K/W

US_UNIT = {"len": "ft", "thk": "in.", "temp": "°F", "wind": "mph"}
SI_UNIT = {"len": "m", "thk": "mm", "temp": "°C", "wind": "m/s"}

_TO_SI = {
    "len": lambda v: v * FT_M,
    "thk": lambda v: v * IN_MM,
    "temp": lambda v: (v - 32.0) / 1.8,
    "wind": lambda v: v * MPH_MS,
}
_TO_US = {
    "len": lambda v: v / FT_M,
    "thk": lambda v: v / IN_MM,
    "temp": lambda v: v * 1.8 + 32.0,
    "wind": lambda v: v / MPH_MS,
}


def to_si(kind, v):
    return None if v is None else _TO_SI[kind](v)


def to_us(kind, v):
    return None if v is None else _TO_US[kind](v)


def convert_inputs_on_toggle():
    """When the unit switch flips, convert the numbers already typed so nothing is lost."""
    si_now = st.session_state.get("si_units", False)
    for key, kind in st.session_state.get("_unit_reg", {}).items():
        if key in st.session_state:
            v = st.session_state[key]
            st.session_state[key] = to_si(kind, v) if si_now else to_us(kind, v)


# ==========================================
# 4. SHARED UI RENDERING HELPERS
# ==========================================
PALETTE = ["#e4572e", "#f3a712", "#5b9bd5", "#7a8b99"]


def get_unit_converters(si_flag):
    if si_flag:
        return {
            "U_T": "°C", "U_DT": "K", "U_LEN": "mm", "U_Q": "W/m²",
            "U_QL": "W/m", "U_H": "W/(m²·K)", "U_K": "W/(m·K)", "U_R": "m²·K/W",
            "T": lambda f: (f - 32.0) / 1.8,
            "DT": lambda f: f / 1.8,
            "LEN": lambda inch: inch * IN_MM,
            "Q": lambda v: v * BTU_HFT2_TO_WM2,
            "QL": lambda v: v * BTU_HFT_TO_WM,
            "HC": lambda v: v * H_TO_SI,
            "KE": lambda v: v * K_TO_SI,
            "RR": lambda v: v * R_TO_SI,
            "r_fmt": ".4f", "len_fmt": ".0f",
        }
    return {
        "U_T": "°F", "U_DT": "°F", "U_LEN": "in.", "U_Q": "Btu/(h·ft²)",
        "U_QL": "Btu/(h·ft)", "U_H": "Btu/(h·ft²·°F)", "U_K": "Btu·in/(h·ft²·°F)", "U_R": "h·ft²·°F/Btu",
        "T": lambda v: v, "DT": lambda v: v, "LEN": lambda v: v,
        "Q": lambda v: v, "QL": lambda v: v, "HC": lambda v: v,
        "KE": lambda v: v, "RR": lambda v: v,
        "r_fmt": ".3f", "len_fmt": ".1f",
    }


def render_solution_details(geom, d, lining_id, thick, t0, t1, t_prof, q_flux, ke_list, R_list, h_tot, hr, hc, si_flag):
    u = get_unit_converters(si_flag)
    T, DT, LEN, Q, QL, HC, KE, RR = u["T"], u["DT"], u["LEN"], u["Q"], u["QL"], u["HC"], u["KE"], u["RR"]
    U_T, U_DT, U_LEN, U_Q, U_QL, U_H, U_K, U_R = u["U_T"], u["U_DT"], u["U_LEN"], u["U_Q"], u["U_QL"], u["U_H"], u["U_K"], u["U_R"]

    ts = float(t_prof[-1])
    if ts < 200:
        grad, status = "linear-gradient(135deg,#2e9e6b,#1f7a52)", "Cool shell"
    elif ts < 300:
        grad, status = "linear-gradient(135deg,#e0a21b,#c27d0e)", "Warm shell"
    else:
        grad, status = "linear-gradient(135deg,#e4572e,#b83a17)", "Hot shell"

    st.markdown(
        f"""
<div class="hero" style="background:{grad}">
  <div class="lbl">Predicted shell temperature · {LININGS[lining_id][0]}</div>
  <div class="big">{T(ts):.0f} {U_T}</div>
  <div class="note">{status} · {DT(ts - t0):+.0f} {U_DT} vs ambient</div>
</div>
""",
        unsafe_allow_html=True,
    )

    total_th = sum(thick)
    if geom in ("hcyl", "vcyl"):
        third = (f"{QL(pi * d['D'] * q_flux):,.0f}", f"{U_QL} of length", "Heat loss per length")
    else:
        third = (f"{LEN(total_th):{u['len_fmt']}}", "mm" if si_flag else "inches", "Lining thickness")

    st.markdown(
        f"""
<div class="grid">
  <div class="kpi"><div class="l">Heat loss</div><div class="v">{Q(q_flux):,.0f}</div><div class="u">{U_Q}</div></div>
  <div class="kpi"><div class="l">{third[2]}</div><div class="v">{third[0]}</div><div class="u">{third[1]}</div></div>
  <div class="kpi"><div class="l">Outer film h</div><div class="v">{HC(h_tot):.2f}</div><div class="u">{U_H}</div></div>
  <div class="kpi"><div class="l">Radiation / convection</div><div class="v">{HC(hr):.1f} / {HC(hc):.1f}</div><div class="u">{U_H}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Temperature profile chart
    st.markdown('<div class="sec">Temperature through the lining</div>', unsafe_allow_html=True)
    _, layers_info = LININGS[lining_id]
    edges = LEN(np.concatenate(([0.0], np.cumsum(thick))))
    t_disp = np.array([T(v) for v in t_prof])
    names = [n for n, _, _ in layers_info]
    y_lo = T(min(t0, ts) - 0.05 * abs(t1 - t0))
    y_hi = T(t1 + 0.05 * abs(t1 - t0))
    bands = pd.DataFrame({
        "Layer": names,
        "x0": edges[:-1],
        "x1": edges[1:],
        "y0": [y_lo] * len(names),
        "y1": [y_hi] * len(names),
    })
    color_scale = alt.Scale(domain=names, range=PALETTE[:len(names)])
    bg = alt.Chart(bands).mark_rect(opacity=0.22).encode(
        x=alt.X("x0:Q", title=f"Distance from hot face ({U_LEN})"),
        x2="x1:Q",
        y=alt.Y("y0:Q", title=f"Temperature ({U_T})"),
        y2="y1:Q",
        color=alt.Color("Layer:N", scale=color_scale, legend=alt.Legend(orient="bottom", title=None)),
    )
    line_df = pd.DataFrame({"x": edges, "T": t_disp})
    line = alt.Chart(line_df).mark_line(strokeWidth=3, color="#d62728", point=alt.OverlayMarkDef(size=70, color="#d62728")).encode(
        x="x:Q", y="T:Q", tooltip=[alt.Tooltip("x:Q", title=f"Depth ({U_LEN})", format=".1f"),
                                    alt.Tooltip("T:Q", title=f"Temp ({U_T})", format=".0f")])
    labels = alt.Chart(line_df).mark_text(dy=-12, fontWeight="bold", fontSize=12).encode(
        x="x:Q", y="T:Q", text=alt.Text("T:Q", format=".0f"))
    chart = (bg + line + labels).properties(height=300).configure_axis(grid=False)
    st.altair_chart(chart, width="stretch")

    # Layer cards
    st.markdown('<div class="sec">Layers</div>', unsafe_allow_html=True)
    cards = ""
    for i, nm in enumerate(names):
        cards += f"""
<div class="lay">
  <div class="bar" style="background:{PALETTE[i]}"></div>
  <div style="flex:1">
    <div class="nm">{nm} · {LEN(thick[i]):{'.0f' if si_flag else '.2f'}} {U_LEN}</div>
    <div class="tp">{T(t_prof[i]):.0f} {U_T} → {T(t_prof[i+1]):.0f} {U_T} &nbsp;(drop {DT(t_prof[i]-t_prof[i+1]):.0f} {U_DT})</div>
    <div class="tm">k<sub>eff</sub> {KE(ke_list[i]):.{3 if si_flag else 2}f} {U_K} · R {RR(R_list[i]):{u['r_fmt']}} {U_R if si_flag else ''}</div>
  </div>
</div>"""
    st.markdown(cards, unsafe_allow_html=True)

    # Where the temperature drop happens
    st.markdown('<div class="sec">Where the temperature drop happens</div>', unsafe_allow_html=True)
    drops = [float(t_prof[i] - t_prof[i + 1]) for i in range(len(names))] + [max(ts - t0, 0.0)]
    dn = names + ["Outside air film"]
    dc = PALETTE[:len(names)] + ["#9aa5b1"]
    tot = sum(drops) or 1.0
    seg = "".join(
        f'<div style="width:{v / tot * 100:.2f}%;background:{c}" title="{n}: {DT(v):.0f} {U_DT}"></div>'
        for v, c, n in zip(drops, dc, dn)
    )
    leg = "".join(
        f'<span><i style="background:{c}"></i>{n} {v / tot * 100:.0f}%</span>' for v, c, n in zip(drops, dc, dn)
    )
    st.markdown(f'<div class="stack">{seg}</div><div class="leg">{leg}</div>', unsafe_allow_html=True)


# ==========================================
# 5. MAIN STREAMLIT PAGE SETUP
# ==========================================
st.set_page_config(page_title="API RP 985 Calculator & Optimizer", page_icon="🔥", layout="centered",
                   initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
.block-container {padding: 1rem 0.9rem 3rem; max-width: 820px;}
h1 {font-size: 1.5rem !important; margin-bottom: 0 !important;}
.sub {opacity: .65; font-size: .85rem; margin-bottom: .4rem;}
.hero {border-radius: 18px; padding: 18px 16px; margin: 8px 0 12px; color: #fff;}
.hero .lbl {font-size: .8rem; opacity: .9; letter-spacing: .04em; text-transform: uppercase;}
.hero .big {font-size: 3rem; font-weight: 800; line-height: 1.1;}
.hero .note {font-size: .85rem; opacity: .95; margin-top: 2px;}
.grid {display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px;}
.cat-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin: 10px 0 16px;}
.cat-card {border: 1px solid rgba(128,128,128,.28); border-radius: 16px; padding: 12px 14px; background: rgba(128,128,128,.03);}
.cat-card .badge {font-size: .72rem; font-weight: 800; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 4px;}
.cat-card .title {font-size: .95rem; font-weight: 700; margin-bottom: 6px;}
.cat-card .stat {font-size: .82rem; opacity: .85; line-height: 1.45;}
.kpi {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 10px 12px;}
.kpi .l {font-size: .72rem; opacity: .65; text-transform: uppercase; letter-spacing: .03em;}
.kpi .v {font-size: 1.25rem; font-weight: 700;}
.kpi .u {font-size: .72rem; opacity: .6;}
.sec {font-size: 1.05rem; font-weight: 700; margin: 18px 0 6px;}
.lay {display: flex; gap: 10px; border: 1px solid rgba(128,128,128,.28); border-radius: 14px;
      padding: 10px 12px; margin-bottom: 8px;}
.lay .bar {width: 6px; border-radius: 4px; flex: none;}
.lay .nm {font-weight: 700; font-size: .95rem;}
.lay .tp {font-size: .8rem; opacity: .75;}
.lay .tm {font-size: .78rem; opacity: .6; margin-top: 2px;}
.stack {display: flex; height: 26px; border-radius: 8px; overflow: hidden; margin: 6px 0;}
.leg {display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: .78rem; opacity: .85;}
.leg i {display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px;}
.chk {font-size: .8rem; opacity: .7; margin-top: 10px; text-align: center;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🔥 API RP 985 Refractory Thermal Suite")
st.markdown('<div class="sub">Steady-state heat loss calculator & evolutionary reverse lining designer</div>', unsafe_allow_html=True)

SI = st.toggle("SI units (m · mm · °C · W)  —  off = US units (ft · in · °F · Btu)",
               key="si_units", on_change=convert_inputs_on_toggle)

GEOM_LABELS = {
    "hcyl": "Horizontal cylinder",
    "vcyl": "Vertical cylinder",
    "wall": "Vertical wall",
    "roof": "Roof (heat flows up)",
    "floor": "Floor (heat flows down)",
    "sphere": "Sphere",
}
LABEL_TO_GEOM = {v: k for k, v in GEOM_LABELS.items()}

_unit_reg = {}


def num(label, kind, us_default, us_min, us_max, us_step, si_step, key):
    """Number input shown in the active unit system; ALWAYS returns the value in US units."""
    _unit_reg[key] = kind
    kw = {} if key in st.session_state else {"value": None}
    if SI:
        if "value" in kw:
            kw["value"] = float(to_si(kind, us_default))
        v = st.number_input(f"{label} ({SI_UNIT[kind]})", min_value=to_si(kind, us_min),
                            max_value=to_si(kind, us_max), step=si_step, format="%.3f" if kind == "len" else "%.2f",
                            key=key, **kw)
        return float(to_us(kind, v))
    if "value" in kw:
        kw["value"] = float(us_default)
    return float(st.number_input(f"{label} ({US_UNIT[kind]})", min_value=us_min, max_value=us_max,
                                 step=us_step, format="%.2f", key=key, **kw))


# Top-level tabs: Forward Calculator vs Reverse Designer
main_tab_fwd, main_tab_rev = st.tabs([
    "🧮 Forward Calculator (Analyze Lining)",
    "🎯 Reverse Designer (Evolutionary Optimizer)"
])

# ==========================================
# TAB 1: FORWARD CALCULATOR (Original Mode)
# ==========================================
with main_tab_fwd:
    with st.expander("⚙️ Inputs", expanded=True):
        preset_name = st.selectbox("Start from a benchmark case", list(PRESETS.keys()))
        preset = PRESETS[preset_name]
        K = preset_name

        tab_g, tab_l, tab_c = st.tabs(["📐 Shape", "🧱 Lining", "🌡️ Conditions"])

        with tab_g:
            default_geom = GEOM_LABELS[preset[0]] if preset else "Horizontal cylinder"
            geom_label = st.selectbox("Equipment type", list(GEOM_LABELS.values()),
                                      index=list(GEOM_LABELS.values()).index(default_geom), key=f"geom_{K}")
            geom = LABEL_TO_GEOM[geom_label]
            d = {}
            if geom in ("hcyl", "vcyl", "sphere"):
                d["D"] = num("Shell inside diameter", "len", float(preset[1].get("D", 8.0)) if preset else 8.0,
                             0.5, None, 0.5, 0.1, f"D_{K}")
            if geom in ("vcyl", "wall"):
                d["h"] = num("Height", "len", float(preset[1].get("h", 30.0)) if preset else 30.0,
                             0.5, None, 1.0, 0.5, f"h_{K}")
            if geom in ("wall", "roof", "floor"):
                d["l"] = num("Length in wind direction", "len", float(preset[1].get("l", 30.0)) if preset else 30.0,
                             0.5, None, 1.0, 0.5, f"l_{K}")
            if geom in ("roof", "floor"):
                d["w"] = num("Width", "len", float(preset[1].get("w", 6.0)) if preset else 6.0,
                             0.5, None, 1.0, 0.5, f"w_{K}")

        with tab_l:
            def_lin = preset[2] if preset else 4
            lining_id = st.selectbox("Lining system", list(LININGS.keys()),
                                     format_func=lambda k: LININGS[k][0],
                                     index=list(LININGS.keys()).index(def_lin), key=f"lin_{K}")
            _, layers_info = LININGS[lining_id]
            thick = []
            for idx, (lname, def_th, _) in enumerate(layers_info):
                pth = float(preset[3][idx]) if (preset and preset[2] == lining_id and idx < len(preset[3])) else float(def_th)
                thick.append(num(f"{idx+1}. {lname} thickness", "thk", pth, 0.1, None, 0.5, 5.0,
                                 f"th_{K}_{lining_id}_{idx}"))

        with tab_c:
            t1 = num("Hot face temperature", "temp", float(preset[7]) if preset else 2000.0,
                     -200.0, 4000.0, 25.0, 10.0, f"t1_{K}")
            t0 = num("Ambient temperature", "temp", float(preset[5]) if preset else 100.0,
                     -100.0, 500.0, 5.0, 2.0, f"t0_{K}")
            wind = num("Wind speed", "wind", float(preset[6]) if preset else 0.0,
                       0.0, 150.0, 1.0, 0.5, f"wind_{K}")
            eps = st.slider("Shell emissivity (ε)", 0.05, 1.00,
                            float(preset[4]) if preset else 0.80, 0.05, key=f"eps_{K}")

    try:
        t_prof, q_flux, ke_list, R_list, h_tot, hr, hc, iters = solve_api985(geom, d, lining_id, thick, eps, t0, wind, t1)
        render_solution_details(geom, d, lining_id, thick, t0, t1, t_prof, q_flux, ke_list, R_list, h_tot, hr, hc, SI)
        if preset:
            u = get_unit_converters(SI)
            ts = float(t_prof[-1])
            ok = abs(ts - preset[8]) <= 2 and abs(q_flux - preset[9]) <= 3
            icon = "✅" if ok else "⚠️"
            st.markdown(
                f'<div class="chk">{icon} Table A.8 target: {u["T"](preset[8]):.0f} {u["U_T"]} · {u["Q"](preset[9]):.0f} {u["U_Q"]} '
                f'&nbsp;|&nbsp; calculated: {u["T"](ts):.0f} {u["U_T"]} · {u["Q"](q_flux):.0f}</div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"⚠️ {e}")


# ==========================================
# TAB 2: REVERSE DESIGNER (Evolutionary Optimizer)
# ==========================================
with main_tab_rev:
    st.markdown(
        "Enter your vessel geometry, operating conditions, and **maximum allowable shell temperature**. "
        "The **Differential Evolution (DE)** optimizer will search across API RP 985 lining systems and layer "
        "thicknesses to find optimal designs categorized by **Thermal Performance**, **Minimum Thickness**, and **Balanced Design**."
    )

    with st.expander("🎯 Design Requirements & Evolutionary Settings", expanded=True):
        rtab_geom, rtab_cond, rtab_adv = st.tabs([
            "📐 Shape & Dimensions",
            "🌡️ Conditions & Target",
            "🧬 Advanced & Layer Constraints"
        ])

        with rtab_geom:
            r_geom_label = st.selectbox("Equipment type", list(GEOM_LABELS.values()), index=0, key="rev_geom")
            r_geom = LABEL_TO_GEOM[r_geom_label]
            r_d = {}
            if r_geom in ("hcyl", "vcyl", "sphere"):
                r_d["D"] = num("Shell inside diameter", "len", 8.0, 0.5, None, 0.5, 0.1, "rev_D")
            if r_geom in ("vcyl", "wall"):
                r_d["h"] = num("Height", "len", 30.0, 0.5, None, 1.0, 0.5, "rev_h")
            if r_geom in ("wall", "roof", "floor"):
                r_d["l"] = num("Length in wind direction", "len", 30.0, 0.5, None, 1.0, 0.5, "rev_l")
            if r_geom in ("roof", "floor"):
                r_d["w"] = num("Width", "len", 6.0, 0.5, None, 1.0, 0.5, "rev_w")

        with rtab_cond:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                r_t1 = num("Hot face temperature", "temp", 2000.0, 100.0, 3000.0, 25.0, 10.0, "rev_t1")
                r_t0 = num("Ambient temperature", "temp", 100.0, -100.0, 500.0, 5.0, 2.0, "rev_t0")
                r_wind = num("Wind speed", "wind", 0.0, 0.0, 150.0, 1.0, 0.5, "rev_wind")
            with col_c2:
                r_target_ts = num("Target MAX shell temperature", "temp", 280.0, -50.0, 1500.0, 5.0, 2.0, "rev_target_ts")
                r_eps = st.slider("Shell emissivity (ε)", 0.05, 1.00, 0.80, 0.05, key="rev_eps")

        with rtab_adv:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                r_max_total = num("Max allowable total lining thickness", "thk", 18.0, 1.0, 60.0, 0.5, 10.0, "rev_max_tot")
                r_min_layer = num("Min individual layer thickness", "thk", 1.0, 0.25, 20.0, 0.25, 5.0, "rev_min_lay")
                r_max_layer = num("Max individual layer thickness", "thk", 14.0, 1.0, 48.0, 0.5, 10.0, "rev_max_lay")
            with col_a2:
                step_options_us = {
                    "Continuous (exact)": 0.0,
                    "0.25 in. (~6.4 mm)": 0.25,
                    "0.50 in. (~12.7 mm)": 0.50,
                    "1.00 in. (~25.4 mm)": 1.00,
                }
                step_label = st.selectbox("Practical thickness rounding increment", list(step_options_us.keys()), index=1)
                r_step_inc = step_options_us[step_label]
                r_pop = st.slider("DE Population size (per lining)", 12, 50, 24, 4)
                r_gens = st.slider("DE Generations", 15, 80, 35, 5)

            r_allowed_linings = st.multiselect(
                "Candidate API RP 985 Lining Systems to evaluate",
                options=list(LININGS.keys()),
                default=list(LININGS.keys()),
                format_func=lambda k: LININGS[k][0],
            )
            r_enforce_t_limits = st.checkbox(
                "Enforce API RP 985 Section 8.3 material temperature limits (e.g. Linings 1–4 ≤ 2000 °F, Lining 5 ≤ 2500 °F)",
                value=True
            )

            st.markdown("**🔒 Lock / Fix Specific Layer Thicknesses (Optional)**")
            st.caption("Check any layer below to lock its thickness to a fixed value while the evolutionary algorithm optimizes the remaining layers.")
            r_fixed_layers = {}
            r_layer_bounds = {}
            for lid in r_allowed_linings:
                _, l_info = LININGS[lid]
                with st.expander(f"Configure layers for {LININGS[lid][0]}", expanded=False):
                    for l_idx, (lname, def_th, _) in enumerate(l_info):
                        c_chk, c_val = st.columns([1, 2])
                        with c_chk:
                            is_locked = st.checkbox(f"Fix Layer {l_idx+1}: {lname}", value=False, key=f"lock_{lid}_{l_idx}")
                        with c_val:
                            locked_val = num(
                                f"Fixed {lname} thickness", "thk", float(def_th),
                                0.1, 48.0, 0.25, 5.0, f"lockval_{lid}_{l_idx}"
                            )
                        if is_locked:
                            r_fixed_layers[(lid, l_idx)] = locked_val
                        r_layer_bounds[(lid, l_idx)] = (r_min_layer, max(r_min_layer + 0.5, r_max_layer))

    run_btn = st.button("🚀 Run Evolutionary Optimization", type="primary", use_container_width=True)

    if run_btn:
        if not r_allowed_linings:
            st.warning("Please select at least one candidate lining system.")
        elif r_target_ts <= r_t0 + 2.0:
            st.error("Target shell temperature must be higher than ambient temperature.")
        else:
            with st.spinner("Running Differential Evolution across candidate lining systems..."):
                categories, ranked_list = run_evolutionary_design(
                    geom=r_geom,
                    d=r_d,
                    eps=r_eps,
                    t0=r_t0,
                    wind=r_wind,
                    t1=r_t1,
                    target_ts=r_target_ts,
                    allowed_linings=r_allowed_linings,
                    max_total_thick=r_max_total,
                    layer_bounds=r_layer_bounds,
                    fixed_layers=r_fixed_layers,
                    step_inc=r_step_inc,
                    pop_size=r_pop,
                    generations=r_gens,
                    enforce_temp_limits=r_enforce_t_limits,
                )
                st.session_state["rev_categories"] = categories
                st.session_state["rev_ranked"] = ranked_list
                st.session_state["rev_context"] = (r_geom, r_d, r_t0, r_t1, r_target_ts, r_max_total)

    # Display Results if available in session_state
    if "rev_categories" in st.session_state and st.session_state["rev_categories"]:
        categories = st.session_state["rev_categories"]
        ranked_list = st.session_state["rev_ranked"]
        ctx_geom, ctx_d, ctx_t0, ctx_t1, ctx_target_ts, ctx_max_tot = st.session_state["rev_context"]
        u = get_unit_converters(SI)
        T, LEN, Q = u["T"], u["LEN"], u["Q"]
        U_T, U_LEN, U_Q = u["U_T"], u["U_LEN"], u["U_Q"]

        any_feasible = any(s["feasible"] for s in ranked_list)
        if not any_feasible:
            st.warning(
                "⚠️ No solution strictly satisfied all constraints (Target Shell Temp / Max Total Thickness / Material Temp Limits). "
                "Showing the closest plausible designs with lowest constraint violation."
            )

        st.markdown('<div class="sec">🏆 Categorized Optimal Designs</div>', unsafe_allow_html=True)

        cat_meta = [
            ("thermal", "🔥 Best Thermal Performance", "Least heat loss found", "#e4572e"),
            ("thickness", "📏 Minimum Thickness", "Least total thickness found", "#2e9e6b"),
            ("balanced", "⚖️ Balanced Design", "Optimal Pareto trade-off (Heat Loss vs. Thickness)", "#3b82f6"),
        ]

        cards_html = '<div class="cat-grid">'
        for c_key, c_title, c_sub, c_color in cat_meta:
            sol = categories[c_key]
            lid = sol["lining_id"]
            l_names = [n for n, _, _ in LININGS[lid][1]]
            th_str = " + ".join(
                f"{LEN(v):{'.0f' if SI else '.2f'}} {U_LEN} ({nm})"
                for v, nm in zip(sol["thick"], l_names)
            )
            feas_badge = "✅ Meets Target" if sol["feasible"] else "⚠️ Near Target"
            cards_html += f"""
<div class="cat-card" style="border-top: 4px solid {c_color}">
  <div class="badge" style="color:{c_color}">{c_title} · {feas_badge}</div>
  <div class="title">Lining {lid}: {LININGS[lid][0].split('·')[1].strip()}</div>
  <div class="stat">
    <b>Shell Temp:</b> {T(sol['ts']):.1f} {U_T} (Target ≤ {T(ctx_target_ts):.0f} {U_T})<br>
    <b>Heat Loss:</b> {Q(sol['q']):,.0f} {U_Q}<br>
    <b>Total Thickness:</b> {LEN(sol['total_thick']):{'.0f' if SI else '.2f'}} {U_LEN}<br>
    <b>Layers:</b> {th_str}
  </div>
</div>"""
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

        # Ranked Solutions Table
        st.markdown('<div class="sec">📊 Ranked Evolutionary Solutions</div>', unsafe_allow_html=True)
        rows = []
        for s in ranked_list:
            lid = s["lining_id"]
            l_names = [n for n, _, _ in LININGS[lid][1]]
            layers_desc = " | ".join(
                f"{nm}: {LEN(v):{'.0f' if SI else '.2f'}} {U_LEN}"
                for nm, v in zip(l_names, s["thick"])
            )
            tags = []
            if s["lining_id"] == categories["thermal"]["lining_id"] and s["thick"] == categories["thermal"]["thick"]:
                tags.append("🔥 Best Thermal")
            if s["lining_id"] == categories["thickness"]["lining_id"] and s["thick"] == categories["thickness"]["thick"]:
                tags.append("📏 Min Thickness")
            if s["lining_id"] == categories["balanced"]["lining_id"] and s["thick"] == categories["balanced"]["thick"]:
                tags.append("⚖️ Balanced")

            rows.append({
                "Rank": s["rank"],
                "Category": ", ".join(tags) if tags else ("✅ Feasible" if s["feasible"] else "⚠️ Violates Limit"),
                "Lining System": f"Lining {lid}",
                f"Layer Thicknesses ({U_LEN})": layers_desc,
                f"Total Thickness ({U_LEN})": round(LEN(s["total_thick"]), 1 if SI else 2),
                f"Shell Temp ({U_T})": round(T(s["ts"]), 1),
                f"Heat Loss ({U_Q})": round(Q(s["q"]), 0),
                "Balanced Score": round(s["balanced_score"], 3),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Interactive Profile Inspector
        st.markdown('<div class="sec">🔍 Inspect Solution Temperature Profile & Layer Breakdown</div>', unsafe_allow_html=True)
        inspect_options = {
            f"🔥 Category 1: Best Thermal Performance (Lining {categories['thermal']['lining_id']} · {LEN(categories['thermal']['total_thick']):{'.0f' if SI else '.2f'}} {U_LEN})": categories["thermal"],
            f"📏 Category 2: Minimum Thickness (Lining {categories['thickness']['lining_id']} · {LEN(categories['thickness']['total_thick']):{'.0f' if SI else '.2f'}} {U_LEN})": categories["thickness"],
            f"⚖️ Category 3: Balanced Design (Lining {categories['balanced']['lining_id']} · {LEN(categories['balanced']['total_thick']):{'.0f' if SI else '.2f'}} {U_LEN})": categories["balanced"],
        }
        for s in ranked_list:
            lbl = (
                f"Rank #{s['rank']}: Lining {s['lining_id']} · Total {LEN(s['total_thick']):{'.0f' if SI else '.2f'}} {U_LEN} · "
                f"Ts {T(s['ts']):.0f} {U_T} · q {Q(s['q']):.0f} {U_Q}"
            )
            inspect_options[lbl] = s

        selected_lbl = st.selectbox("Select a design to view its full temperature profile:", list(inspect_options.keys()))
        sel_sol = inspect_options[selected_lbl]

        render_solution_details(
            geom=ctx_geom,
            d=ctx_d,
            lining_id=sel_sol["lining_id"],
            thick=sel_sol["thick"],
            t0=ctx_t0,
            t1=ctx_t1,
            t_prof=sel_sol["t_prof"],
            q_flux=sel_sol["q"],
            ke_list=sel_sol["ke_list"],
            R_list=sel_sol["R_list"],
            h_tot=sel_sol["h_tot"],
            hr=sel_sol["hr"],
            hc=sel_sol["hc"],
            si_flag=SI,
        )

st.session_state["_unit_reg"] = _unit_reg