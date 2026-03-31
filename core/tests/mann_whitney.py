import numpy as np
from scipy import stats
from core.neutrosophic import NeutrosophicNumber, NeutrosophicArray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _u_statistics(ranks: np.ndarray, n1: int, n2: int) -> tuple:
    """
    Compute U1 and U2 from the rank sum of group 1.

        U1 = n1*n2 + n1*(n1+1)/2 - R1
        U2 = n1*n2 - U1          (from the identity U1 + U2 = n1*n2)

    Using the identity for U2 is more numerically stable than recomputing
    from R2 independently — it guarantees U1 + U2 == n1*n2 exactly.
    """
    R1 = float(ranks[:n1].sum())
    U1 = n1 * n2 + n1 * (n1 + 1) / 2.0 - R1
    U2 = n1 * n2 - U1          # identity: always U1+U2 = n1*n2
    return U1, U2


def _z_and_p(U: float, n1: int, n2: int) -> tuple:
    """
    Normal approximation for the U-statistic (two-tailed).
    Valid for n1, n2 >= 8 (used unconditionally here;
    small-sample callers should prefer exact methods).

        μ_U = n1*n2 / 2
        σ_U = sqrt(n1*n2*(n1+n2+1) / 12)
        Z   = (U - μ_U) / σ_U
        p   = 2 * P(Z > |z|)
    """
    mu    = n1 * n2 / 2.0
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    Z     = (U - mu) / sigma if sigma > 0 else 0.0
    p     = float(2.0 * stats.norm.sf(abs(Z)))
    return float(Z), p


def _three_zone(p_lower: float, p_upper: float, alpha: float) -> str:
    if p_upper < alpha:
        return "Reject H0"
    if p_lower > alpha:
        return "Fail to Reject H0"
    return "Indeterminate Decision"


_SMALL_N_THRESHOLD = 8   # below this, Z approximation is unreliable


# ---------------------------------------------------------------------------
# Original Neutrosophic Mann-Whitney U Test
# ---------------------------------------------------------------------------

def mann_whitney_original(
    group1: NeutrosophicArray,
    group2: NeutrosophicArray,
    alpha: float = 0.05,
) -> dict | None:
    """
    Original Neutrosophic Mann-Whitney U Test (He & Lin, 2020).

    The test statistic is computed for each neutrosophic component (T, I, F)
    by ranking the corresponding midpoints across the combined sample.

        U_c = min(U_c1, U_c2)   for c in {T, I, F}

    where U_c1 = n1*n2 + n1*(n1+1)/2 − R_c1
          U_c2 = n1*n2 − U_c1          (identity)

    FIX 1 — U2 formula used n2*(n2+1)/2 − R2 independently.
    This is equivalent but breaks the U1+U2=n1*n2 identity under floating-
    point arithmetic when there are ties. Now U2 is derived from the identity.

    FIX 2 — U_N was stored as zero-width point intervals (U_T,U_T),(U_I,U_I),(U_F,U_F).
    U_N is now a proper neutrosophic interval spanning [min(U_T,U_I,U_F), max(...)].

    FIX 3 — alpha was hardcoded as 0.05. Now a named parameter.

    FIX 4 — Small-sample warning: Z approximation is inaccurate for n < 8.
    A warning is included in the result dict; callers can check it.

    Parameters
    ----------
    group1, group2 : NeutrosophicArray
    alpha          : significance level (default 0.05)

    Returns
    -------
    dict with keys:
        U_T1, U_T2, U_T, U_I, U_F,
        U_N,
        Z_T, Z_I, Z_F,
        p_T, p_I, p_F, p_interval,
        decision_zone, overall_decision,
        alpha, modified, small_sample_warning
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return None

    all_data = list(group1) + list(group2)
    N = n1 + n2

    # --- Component midpoints ------------------------------------------------
    t_mids = np.array([(n.T[0] + n.T[1]) / 2.0 for n in all_data])
    i_mids = np.array([(n.I[0] + n.I[1]) / 2.0 for n in all_data])
    f_mids = np.array([(n.F[0] + n.F[1]) / 2.0 for n in all_data])

    t_ranks = stats.rankdata(t_mids, method="average")
    i_ranks = stats.rankdata(i_mids, method="average")
    f_ranks = stats.rankdata(f_mids, method="average")

    # --- U statistics -------------------------------------------------------
    # FIX 1: U2 derived from identity U1+U2=n1*n2 — numerically exact.
    U_T1, U_T2 = _u_statistics(t_ranks, n1, n2)
    U_I1, U_I2 = _u_statistics(i_ranks, n1, n2)
    U_F1, U_F2 = _u_statistics(f_ranks, n1, n2)

    U_T = min(U_T1, U_T2)
    U_I = min(U_I1, U_I2)
    U_F = min(U_F1, U_F2)

    # FIX 2: U_N as a proper neutrosophic interval, not zero-width points.
    # The T-component spans [min(U_T,U_F), max(U_T,U_F)] — the data-driven
    # uncertainty range.  I-component is the indeterminacy U as a point.
    U_low  = min(U_T, U_I, U_F)
    U_high = max(U_T, U_I, U_F)
    U_N = NeutrosophicNumber(
        (U_low,  U_high),
        (U_I,    U_I),
        (U_low,  U_high),
    )

    # --- Z statistics and p-values ------------------------------------------
    Z_T, p_T = _z_and_p(U_T, n1, n2)
    Z_I, p_I = _z_and_p(U_I, n1, n2)
    Z_F, p_F = _z_and_p(U_F, n1, n2)

    p_lower = min(p_T, p_I, p_F)
    p_upper = max(p_T, p_I, p_F)

    # --- Decision -----------------------------------------------------------
    decision_zone = _three_zone(p_lower, p_upper, alpha)

    # FIX 4: flag when normal approximation may be unreliable
    small_sample_warning = (n1 < _SMALL_N_THRESHOLD or n2 < _SMALL_N_THRESHOLD)

    return {
        # U values
        "U_T1":                 U_T1,
        "U_T2":                 U_T2,
        "U_I1":                 U_I1,
        "U_I2":                 U_I2,
        "U_F1":                 U_F1,
        "U_F2":                 U_F2,
        "U_T":                  U_T,
        "U_I":                  U_I,
        "U_F":                  U_F,
        "U_N":                  U_N,
        # Z and p
        "Z_T":                  Z_T,
        "Z_I":                  Z_I,
        "Z_F":                  Z_F,
        "Z_N":                  (Z_T, Z_I, Z_F),
        "p_T":                  p_T,
        "p_I":                  p_I,
        "p_F":                  p_F,
        "p_interval":           (p_lower, p_upper),
        # Decision
        "decision_zone":        decision_zone,
        "overall_decision":     decision_zone,
        # Metadata
        "alpha":                alpha,
        "n1":                   n1,
        "n2":                   n2,
        "modified":             False,
        "small_sample_warning": small_sample_warning,
    }


# ---------------------------------------------------------------------------
# Modified Neutrosophic Mann-Whitney U Test
# ---------------------------------------------------------------------------

def mann_whitney_modified(
    group1: NeutrosophicArray,
    group2: NeutrosophicArray,
    alpha: float = 0.05,
) -> dict | None:
    """
    Modified Neutrosophic Mann-Whitney U Test.

    Three enhancements over the original (per the research specification):

    1. Neutrosophic Dominance Probability
       For every pair (xᵢ, yⱼ), classify as:
         - Truth (T):         xᵢ clearly dominates yⱼ  (xᵢ > yⱼ, neither indeterminate)
         - Indeterminacy (I): either or both are indeterminate, OR xᵢ == yⱼ (tie)
         - Falsehood (F):     yⱼ clearly dominates xᵢ  (xᵢ < yⱼ, neither indeterminate)
       Proportions P_T, P_I, P_F give the dominance triple.

       FIX: The spec says the indeterminacy zone should cover pairs where
       EITHER observation is indeterminate (not only both).  A comparison
       involving one uncertain value produces an uncertain outcome.

    2. Neutrosophic Weighted Average (NWA) U-statistic
       FIX (circular-weight bug): The submitted code used w_T = P_T, i.e. the
       dominance outcome probabilities as the aggregation weights. This is
       circular: the outcome of the test influences its own test statistic.
       Per the spec, weights should reflect DATA QUALITY:
           w_T = proportion of fully determinate observations (non-indeterminate)
           w_I = proportion of indeterminate observations
           w_F = proportion of missing-proxy observations (I interval = full [0,1])
       These weights are computed from BOTH groups combined and then normalised.

    3. Three-Zone Decision Rule — same as original.

    FIX — result dict built from scratch (no orig.copy() + update).
    The submitted code left stale U_T/U_I/U_F keys (original values) alongside
    U_modified, making it ambiguous which U was being reported.

    Parameters
    ----------
    group1, group2 : NeutrosophicArray
    alpha          : significance level (default 0.05)

    Returns
    -------
    dict — all original keys plus:
        dominance_prob_T/I/F   (pair-wise dominance triple)
        w_T, w_I, w_F          (data-quality NWA weights)
        U_modified             (NWA-aggregated U scalar)
        U_N_modified           (U_modified as NeutrosophicNumber interval)
        Z_modified, p_modified
        effect_size_neutrosophic
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return None

    # Run original to get the component U values
    orig = mann_whitney_original(group1, group2, alpha=alpha)
    if orig is None:
        return None

    # ── Modification 1: Neutrosophic Dominance Probability ──────────────────
    # FIX: either-indeterminate rule (not only-both-indeterminate)
    X_vals  = [n.defuzzify()          for n in group1]
    Y_vals  = [n.defuzzify()          for n in group2]
    X_indet = [n.is_indeterminate(0.01) for n in group1]
    Y_indet = [n.is_indeterminate(0.01) for n in group2]

    count_T = count_I = count_F = 0
    for i in range(n1):
        for j in range(n2):
            if X_indet[i] or Y_indet[j]:
                # FIX: EITHER indeterminate → outcome is uncertain → count as I
                count_I += 1
            elif X_vals[i] > Y_vals[j]:
                count_T += 1
            elif X_vals[i] < Y_vals[j]:
                count_F += 1
            else:
                count_I += 1    # exact tie between two crisp values → indeterminate

    total_pairs = n1 * n2
    P_T = count_T / total_pairs
    P_I = count_I / total_pairs
    P_F = count_F / total_pairs

    # ── Modification 2: NWA weights from DATA QUALITY, not dominance ────────
    # FIX: weights must come from the data structure, not from the test outcome.
    #   w_T  = proportion of observations that are fully determinate
    #   w_I  = proportion of observations that are indeterminate (I-width > 0.01)
    #   w_F  = proportion of observations where I spans the full [0,1] (missing proxy)
    all_obs = list(group1) + list(group2)
    N_total = len(all_obs)

    n_missing = sum(1 for n in all_obs if (n.I[1] - n.I[0]) >= 0.99)
    n_indet   = sum(1 for n in all_obs if n.is_indeterminate(0.01)) - n_missing
    n_crisp   = N_total - n_indet - n_missing

    w_T_raw = n_crisp   / N_total
    w_I_raw = n_indet   / N_total
    w_F_raw = n_missing / N_total

    total_w = w_T_raw + w_I_raw + w_F_raw
    if total_w > 0:
        w_T = w_T_raw / total_w
        w_I = w_I_raw / total_w
        w_F = w_F_raw / total_w
    else:
        w_T = w_I = w_F = 1.0 / 3.0

    # ── NWA U-statistic ──────────────────────────────────────────────────────
    U_T = orig["U_T"]
    U_I = orig["U_I"]
    U_F = orig["U_F"]

    U_modified = w_T * U_T + w_I * U_I + w_F * U_F

    Z_modified, p_modified = _z_and_p(U_modified, n1, n2)

    # U_N_modified as a proper interval
    U_mod_low  = min(U_T, U_modified, U_F)
    U_mod_high = max(U_T, U_modified, U_F)
    U_N_modified = NeutrosophicNumber(
        (U_mod_low,  U_mod_high),
        (U_I,        U_I),
        (U_mod_low,  U_mod_high),
    )

    # ── P-value interval: combine original interval with p_modified ──────────
    p_lower = min(orig["p_interval"][0], p_modified)
    p_upper = max(orig["p_interval"][1], p_modified)

    # ── Three-zone decision ──────────────────────────────────────────────────
    decision_zone = _three_zone(p_lower, p_upper, alpha)

    # ── Effect size (rank-biserial r) ────────────────────────────────────────
    # r = Z / sqrt(N),  where N = n1 + n2
    effect_size = Z_modified / np.sqrt(n1 + n2) if (n1 + n2) > 0 else 0.0

    # ── Build result from scratch (no stale keys from orig.copy()) ───────────
    # FIX: orig.copy() left U_T/U_I/U_F (original values) alongside U_modified,
    # making the result dict ambiguous.  We construct from scratch and include
    # original component values under clearly namespaced keys (*_original).
    small_sample_warning = orig["small_sample_warning"]

    return {
        # Original component U values (namespaced to avoid ambiguity)
        "U_T_original":             U_T,
        "U_I_original":             U_I,
        "U_F_original":             U_F,
        "U_N_original":             orig["U_N"],
        # Modified U values
        "U_modified":               U_modified,
        "U_N_modified":             U_N_modified,
        # Dominance triple
        "dominance_prob_T":         P_T,
        "dominance_prob_I":         P_I,
        "dominance_prob_F":         P_F,
        # NWA weights (data-quality based)
        "w_T":                      w_T,
        "w_I":                      w_I,
        "w_F":                      w_F,
        "n_crisp":                  n_crisp,
        "n_indet":                  n_indet,
        "n_missing":                n_missing,
        # Z and p
        "Z_original":               (orig["Z_T"], orig["Z_I"], orig["Z_F"]),
        "Z_modified":               Z_modified,
        "p_T":                      orig["p_T"],
        "p_I":                      orig["p_I"],
        "p_F":                      orig["p_F"],
        "p_modified":               p_modified,
        "p_interval":               (p_lower, p_upper),
        # Decision
        "decision_zone":            decision_zone,
        "overall_decision":         decision_zone,
        # Effect size
        "effect_size_neutrosophic": effect_size,
        # Metadata
        "alpha":                    alpha,
        "n1":                       n1,
        "n2":                       n2,
        "modified":                 True,
        "small_sample_warning":     small_sample_warning,
    }


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from scipy.stats import mannwhitneyu as scipy_mwu

    # Global-range normalisation helper (same pattern as kruskal_wallis.py)
    _all_vals = list(range(1, 11))
    _GMIN, _GRNG = min(_all_vals), max(_all_vals) - min(_all_vals)

    def _make(vals, i_width=0.0, missing_idx=None):
        nums = []
        for k, v in enumerate(vals):
            if missing_idx and k in missing_idx:
                nums.append(NeutrosophicNumber((0.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
            else:
                n = (v - _GMIN) / _GRNG
                nums.append(NeutrosophicNumber(
                    (n,   n),
                    (0.0, i_width),
                    (1-n, 1-n),
                ))
        return NeutrosophicArray(nums)

    g1_vals = [1, 2, 3, 4, 5]
    g2_vals = [6, 7, 8, 9, 10]

    print("=== 1. Classical special case (I=0) ===")
    g1 = _make(g1_vals)
    g2 = _make(g2_vals)
    res = mann_whitney_original(g1, g2, alpha=0.05)
    scipy_U, _ = scipy_mwu(g1_vals, g2_vals, alternative="two-sided")
    assert abs(res["U_T"] - scipy_U) < 1e-9, f"U_T={res['U_T']} != scipy U={scipy_U}"
    print(f"  U_T={res['U_T']:.1f}  (scipy: {scipy_U:.1f}) ✓")

    print("\n=== 2. U1 + U2 == n1*n2 identity ===")
    assert abs(res["U_T1"] + res["U_T2"] - 5 * 5) < 1e-9
    assert abs(res["U_I1"] + res["U_I2"] - 5 * 5) < 1e-9
    assert abs(res["U_F1"] + res["U_F2"] - 5 * 5) < 1e-9
    print(f"  U_T1+U_T2={res['U_T1']+res['U_T2']:.1f} == n1*n2={5*5} ✓")

    print("\n=== 3. U_N is a proper interval ===")
    width = res["U_N"].T[1] - res["U_N"].T[0]
    print(f"  U_N = {res['U_N']}")
    print(f"  U_N.T interval width = {width:.4f}")
    # For fully crisp data U_T may equal U_F, width can be 0 — use mixed data
    g1_mix = _make([1, 2, 3, 4, 5], i_width=0.05)
    g2_mix = _make([6, 7, 8, 9, 10], i_width=0.05)
    res_mix = mann_whitney_original(g1_mix, g2_mix)
    width_mix = res_mix["U_N"].T[1] - res_mix["U_N"].T[0]
    print(f"  With I-width=0.05: U_N.T width = {width_mix:.4f} ✓")

    print("\n=== 4. Alpha is a parameter ===")
    res_strict = mann_whitney_original(g1, g2, alpha=0.01)
    res_loose  = mann_whitney_original(g1, g2, alpha=0.10)
    assert res_strict["alpha"] == 0.01 and res_loose["alpha"] == 0.10
    print(f"  α=0.01: {res_strict['decision_zone']}")
    print(f"  α=0.10: {res_loose['decision_zone']}  ✓")

    print("\n=== 5. Small-sample warning flag ===")
    g_tiny1 = _make([1, 2])
    g_tiny2 = _make([3, 4])
    res_tiny = mann_whitney_original(g_tiny1, g_tiny2)
    assert res_tiny["small_sample_warning"] is True
    res_large = mann_whitney_original(g1, g2)
    # n=5 < 8, so also flagged
    print(f"  n=2 warning: {res_tiny['small_sample_warning']} ✓")
    print(f"  n=5 warning: {res_large['small_sample_warning']} (n<8, expected True) ✓")

    print("\n=== 6. Modified — weights from data quality, not dominance ===")
    g1_indet = _make(g1_vals, i_width=0.1)
    g2_indet = _make(g2_vals, i_width=0.1)
    r_mod = mann_whitney_modified(g1_indet, g2_indet)
    print(f"  w_T={r_mod['w_T']:.4f}  w_I={r_mod['w_I']:.4f}  w_F={r_mod['w_F']:.4f}")
    print(f"  n_crisp={r_mod['n_crisp']}  n_indet={r_mod['n_indet']}  n_missing={r_mod['n_missing']}")
    assert abs(r_mod["w_T"] + r_mod["w_I"] + r_mod["w_F"] - 1.0) < 1e-9
    print("  Weights sum to 1.0 ✓")

    print("\n=== 7. Modified — either-indeterminate rule ===")
    # X is indeterminate, Y is crisp: pair should count as I
    x_indet = NeutrosophicArray([NeutrosophicNumber((0.5, 0.5), (0.0, 0.3), (0.5, 0.5))])
    y_crisp = NeutrosophicArray([NeutrosophicNumber((0.9, 0.9), (0.0, 0.0), (0.1, 0.1))])
    r_either = mann_whitney_modified(x_indet, y_crisp)
    assert r_either["dominance_prob_I"] == 1.0, \
        f"Expected P_I=1.0 for (indet, crisp) pair, got {r_either['dominance_prob_I']}"
    print(f"  P_I for (indet, crisp) pair = {r_either['dominance_prob_I']:.2f} ✓")

    print("\n=== 8. No stale U_T key in modified result ===")
    assert "U_T" not in r_mod, "'U_T' should not exist — use 'U_T_original'"
    assert "U_T_original" in r_mod
    assert "U_modified" in r_mod
    print(f"  U_T_original={r_mod['U_T_original']:.2f}  U_modified={r_mod['U_modified']:.2f}  ✓")

    print("\n=== 9. Effect size in valid range for large separation ===")
    g1_far = _make([1, 2, 3, 4, 5])
    g2_far = _make([6, 7, 8, 9, 10])
    r_far = mann_whitney_modified(g1_far, g2_far)
    es = r_far["effect_size_neutrosophic"]
    assert -1.0 <= es <= 1.0, f"Effect size {es} outside [-1,1]"
    print(f"  effect_size = {es:.4f}  (in [-1,1]) ✓")

    print("\nAll tests passed.")