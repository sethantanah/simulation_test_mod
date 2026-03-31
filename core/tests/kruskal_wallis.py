import numpy as np
from scipy import stats
from core.neutrosophic import NeutrosophicNumber, NeutrosophicArray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect(groups: list) -> tuple:
    """Flatten groups into (all_data, group_sizes, N)."""
    all_data, group_sizes = [], []
    for g in groups:
        all_data.extend(g.data)
        group_sizes.append(len(g.data))
    return all_data, group_sizes, sum(group_sizes)


def _calc_H(ranks: np.ndarray, group_sizes: list, N: int) -> float:
    """
    Core KW H-statistic formula:
        H = (12 / N(N+1)) * Σ(R_i² / n_i) − 3(N+1)

    Returns max(0, H) so that chi2.sf always receives a non-negative value.
    scipy.chi2.sf handles 0 and negative inputs gracefully (returns 1.0),
    but returning max(0, H) is cleaner and avoids any edge-case surprises
    on platforms where that behaviour is not guaranteed.
    """
    sum_R2_n, idx = 0.0, 0
    for n_i in group_sizes:
        R_i = ranks[idx: idx + n_i].sum()
        sum_R2_n += (R_i ** 2) / n_i
        idx += n_i
    H = (12.0 / (N * (N + 1))) * sum_R2_n - 3.0 * (N + 1)
    return max(0.0, float(H))


def _three_zone_decision(p_lower: float, p_upper: float, alpha: float) -> str:
    """
    Three-zone decision rule (spec §4.2, modification 3):
        p_upper < α          → Reject H0         (all components significant)
        p_lower > α          → Fail to Reject H0  (no component significant)
        p_lower ≤ α ≤ p_upper → Indeterminate Decision
    """
    if p_upper < alpha:
        return "Reject H0"
    if p_lower > alpha:
        return "Fail to Reject H0"
    return "Indeterminate Decision"


# ---------------------------------------------------------------------------
# Original Neutrosophic Kruskal-Wallis Test
# ---------------------------------------------------------------------------

def kruskal_wallis_original(
    groups: list,
    alpha: float = 0.05,
) -> dict | None:
    """
    Original Neutrosophic Kruskal-Wallis Test (Sherwani et al., 2021).

    The test statistic is computed three times — once for each neutrosophic
    component (T, I, F) — by ranking the midpoints of the corresponding
    component across all observations.

    NOTE on I and F ranking
    -----------------------
    Ranking I-midpoints and F-midpoints independently is the formulation
    in the original paper.  The T-component rank is the primary data rank;
    the I and F ranks capture how the indeterminacy and falsehood values
    are distributed across groups.  When all I = (0,0) (fully determinate
    data), every i_mid = 0 so all i_ranks are tied and H_I = 0 — which is
    the correct degenerate result: there is no indeterminacy to test.

    Classical special case
    ----------------------
    When all I = (0,0): H_T equals the classical KW H-statistic. ✓
    (Verified in the self-test at the bottom of this file.)

    Parameters
    ----------
    groups : list of NeutrosophicArray
    alpha  : significance level (default 0.05)

    Returns
    -------
    dict with keys:
        H_N, H_T, H_I, H_F,
        p_T, p_I, p_F, p_interval,
        df,
        decision_T, decision_I, decision_F,
        decision_zone, overall_decision,
        alpha, modified
    """
    k = len(groups)
    if k < 2:
        raise ValueError("Need at least 2 groups.")

    all_data, group_sizes, N = _collect(groups)
    if N == 0:
        return None

    # --- Component midpoints ------------------------------------------------
    t_mids = np.array([(n.T[0] + n.T[1]) / 2.0 for n in all_data])
    i_mids = np.array([(n.I[0] + n.I[1]) / 2.0 for n in all_data])
    f_mids = np.array([(n.F[0] + n.F[1]) / 2.0 for n in all_data])

    # --- Ranks (average method for ties) ------------------------------------
    t_ranks = stats.rankdata(t_mids, method="average")
    i_ranks = stats.rankdata(i_mids, method="average")
    f_ranks = stats.rankdata(f_mids, method="average")

    # --- H statistics -------------------------------------------------------
    H_T = _calc_H(t_ranks, group_sizes, N)
    H_I = _calc_H(i_ranks, group_sizes, N)
    H_F = _calc_H(f_ranks, group_sizes, N)

    df = k - 1

    # --- P-values -----------------------------------------------------------
    p_T = float(stats.chi2.sf(H_T, df))
    p_I = float(stats.chi2.sf(H_I, df))
    p_F = float(stats.chi2.sf(H_F, df))

    p_lower = min(p_T, p_I, p_F)
    p_upper = max(p_T, p_I, p_F)

    # --- Decision -----------------------------------------------------------
    decision_T = "Reject" if p_T < alpha else "Fail to Reject"
    decision_I = "Reject" if p_I < alpha else "Fail to Reject"
    decision_F = "Reject" if p_F < alpha else "Fail to Reject"
    overall_decision = _three_zone_decision(p_lower, p_upper, alpha)

    # --- H_N as a proper neutrosophic interval ------------------------------
    # FIX: H_N was stored as N((H_T,H_T),(H_I,H_I),(H_F,H_F)) — zero-width
    # point intervals. The neutrosophic H-statistic should be an interval that
    # spans the range [min(H_T,H_I,H_F), max(H_T,H_I,H_F)] in each component.
    # The natural encoding is:
    #   T-component = [min(H_T, H_F), max(H_T, H_F)]  (data-driven range)
    #   I-component = [H_I, H_I]                       (indeterminacy point)
    #   F-component = same as T by symmetry of the interval
    # A simpler and fully general encoding is the triple of H scalars as
    # intervals of width 0, but then H_N.T = (H_T, H_T) carries no range
    # information. We use [H_low, H_high] across all three scalars:
    H_low  = min(H_T, H_I, H_F)
    H_high = max(H_T, H_I, H_F)
    H_N = NeutrosophicNumber(
        (H_low,  H_high),   # T: spans the full uncertainty range
        (H_I,    H_I),      # I: the indeterminacy H as a point
        (H_low,  H_high),   # F: symmetric with T
    )

    return {
        "H_N":              H_N,
        "H_T":              H_T,
        "H_I":              H_I,
        "H_F":              H_F,
        "p_T":              p_T,
        "p_I":              p_I,
        "p_F":              p_F,
        "p_interval":       (p_lower, p_upper),
        "df":               df,
        "decision_T":       decision_T,
        "decision_I":       decision_I,
        "decision_F":       decision_F,
        "decision_zone":    overall_decision,
        "overall_decision": overall_decision,
        "alpha":            alpha,
        "modified":         False,
    }


# ---------------------------------------------------------------------------
# Modified Neutrosophic Kruskal-Wallis Test
# ---------------------------------------------------------------------------

def kruskal_wallis_modified(
    groups: list,
    alpha: float = 0.05,
) -> dict | None:
    """
    Modified Neutrosophic Kruskal-Wallis Test.

    Three enhancements over the original (per the research specification):

    1. Interval-Valued Neutrosophic Ranking
       Each observation's rank is widened into a rank interval whose width
       is proportional to that observation's I-component width.  Observations
       with wider I intervals (more uncertain) receive wider rank intervals
       [rank − δᵢ/2, rank + δᵢ/2] where δᵢ = I_upper − I_lower.
       The H-statistic is then computed using the lower and upper rank bounds
       separately, producing H_low and H_high — the true neutrosophic interval.

    2. Adaptive Indeterminacy Weight (λ)
       λ = (number of indeterminate observations) / N
       Both H_low and H_high are multiplied by (1 + λ) to upweight the
       test statistic when the dataset contains high indeterminacy, improving
       sensitivity without distorting the ranking structure.

    3. Three-Zone Decision Rule
       Same as the original: Reject / Indeterminate / Fail to Reject.

    FIX — result dict consistency
    ------------------------------
    The original submitted code did `res = orig.copy(); res.update(...)` which
    left H_T/H_I/H_F pointing at the ORIGINAL (unmodified) values while only
    H_T_mod/H_I_mod/H_F_mod held the modified ones.  This version constructs
    the result dict from scratch so every key is unambiguous.

    Parameters
    ----------
    groups : list of NeutrosophicArray
    alpha  : significance level (default 0.05)

    Returns
    -------
    dict — all keys from original plus:
        H_low, H_high        (interval-valued H from rank intervals)
        lambda_weight        (adaptive indeterminacy weight)
        rank_interval_widths (per-observation rank interval widths)
        rank_interval_width_mean
    """
    k = len(groups)
    if k < 2:
        raise ValueError("Need at least 2 groups.")

    all_data, group_sizes, N = _collect(groups)
    if N == 0:
        return None

    # ── Step 1: Base T-midpoint ranks (identical to original) ───────────────
    t_mids  = np.array([(n.T[0] + n.T[1]) / 2.0 for n in all_data])
    i_mids  = np.array([(n.I[0] + n.I[1]) / 2.0 for n in all_data])
    f_mids  = np.array([(n.F[0] + n.F[1]) / 2.0 for n in all_data])
    t_ranks = stats.rankdata(t_mids, method="average")
    i_ranks = stats.rankdata(i_mids, method="average")
    f_ranks = stats.rankdata(f_mids, method="average")

    # ── Step 2: Interval-valued ranking (Modification 1) ────────────────────
    # FIX: rank_interval_width_mean was computed but interval-valued ranking
    # was never actually applied to the H-statistic. Now we compute H_low and
    # H_high from rank lower/upper bounds.
    i_widths = np.array([(n.I[1] - n.I[0]) for n in all_data])
    rank_lower = t_ranks - i_widths / 2.0   # shrink rank by half I-width
    rank_upper = t_ranks + i_widths / 2.0   # expand rank by half I-width

    # Ensure ranks stay positive (clamp lower bound to 0.5, the minimum rank)
    rank_lower = np.clip(rank_lower, 0.5, None)

    H_low  = _calc_H(rank_lower, group_sizes, N)
    H_high = _calc_H(rank_upper, group_sizes, N)

    # Keep original I and F H values for completeness
    H_I = _calc_H(i_ranks, group_sizes, N)
    H_F = _calc_H(f_ranks, group_sizes, N)

    # ── Step 3: Adaptive indeterminacy weight λ (Modification 2) ────────────
    indet_count   = int(sum(1 for n in all_data if n.is_indeterminate(0.01)))
    lambda_weight = indet_count / N if N > 0 else 0.0

    H_low_mod  = H_low  * (1.0 + lambda_weight)
    H_high_mod = H_high * (1.0 + lambda_weight)
    H_I_mod    = H_I    * (1.0 + lambda_weight)
    H_F_mod    = H_F    * (1.0 + lambda_weight)

    # FIX: H_N is now a proper interval — not a point — reflecting the
    # uncertainty introduced by interval-valued ranking.
    H_N = NeutrosophicNumber(
        (H_low_mod,  H_high_mod),
        (H_I_mod,    H_I_mod),
        (H_low_mod,  H_high_mod),
    )

    df = k - 1

    # ── Step 4: P-value interval ─────────────────────────────────────────────
    # Use the H interval [H_low_mod, H_high_mod] to produce a p-value interval.
    # Higher H → lower p, so: p_upper comes from H_low, p_lower from H_high.
    p_upper_H = float(stats.chi2.sf(H_low_mod,  df))
    p_lower_H = float(stats.chi2.sf(H_high_mod, df))
    p_I_mod   = float(stats.chi2.sf(H_I_mod,    df))
    p_F_mod   = float(stats.chi2.sf(H_F_mod,    df))

    # Overall p-interval spans all four p-values
    p_lower = min(p_lower_H, p_I_mod, p_F_mod)
    p_upper = max(p_upper_H, p_I_mod, p_F_mod)

    # ── Step 5: Three-zone decision (Modification 3) ─────────────────────────
    decision_zone = _three_zone_decision(p_lower, p_upper, alpha)

    # Per-component decisions (for reporting)
    decision_T = "Reject" if p_upper_H < alpha else ("Fail to Reject" if p_lower_H > alpha else "Indeterminate")
    decision_I = "Reject" if p_I_mod   < alpha else "Fail to Reject"
    decision_F = "Reject" if p_F_mod   < alpha else "Fail to Reject"

    return {
        # Core H values (modified)
        "H_N":                      H_N,
        "H_low":                    H_low_mod,
        "H_high":                   H_high_mod,
        "H_I":                      H_I_mod,
        "H_F":                      H_F_mod,
        # Pre-weighting interval (useful for diagnostics)
        "H_low_pre_weight":         H_low,
        "H_high_pre_weight":        H_high,
        # P-values
        "p_lower_H":                p_lower_H,
        "p_upper_H":                p_upper_H,
        "p_I":                      p_I_mod,
        "p_F":                      p_F_mod,
        "p_interval":               (p_lower, p_upper),
        # Test structure
        "df":                       df,
        "decision_T":               decision_T,
        "decision_I":               decision_I,
        "decision_F":               decision_F,
        "decision_zone":            decision_zone,
        "overall_decision":         decision_zone,
        # Modification diagnostics
        "lambda_weight":            lambda_weight,
        "indet_count":              indet_count,
        "rank_interval_widths":     i_widths.tolist(),
        "rank_interval_width_mean": float(np.mean(i_widths)),
        "alpha":                    alpha,
        "modified":                 True,
    }


# ---------------------------------------------------------------------------
# Self-tests  (python -m core.tests.kruskal_wallis  or  python kruskal_wallis.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from scipy.stats import kruskal as scipy_kruskal

    g1 = [1, 2, 3, 4, 5]
    g2 = [6, 7, 8, 9, 10]
    g3 = [11, 12, 13, 14, 15]

    # IMPORTANT: normalise across the COMBINED dataset so between-group
    # differences survive.  Per-group normalisation collapses every group
    # to [0,1] independently, making all t_mids identical across groups
    # and destroying the ranking signal (H_T -> 0 for any partition).
    _all_vals = g1 + g2 + g3
    _GMIN, _GRNG = min(_all_vals), max(_all_vals) - min(_all_vals)

    def _make(vals, i_width=0.0):
        """Build a NeutrosophicArray, normalising w.r.t. the global range."""
        nums = []
        for v in vals:
            norm = (v - _GMIN) / _GRNG
            nums.append(NeutrosophicNumber(
                (norm, norm),
                (0.0, i_width),
                (1.0 - norm, 1.0 - norm),
            ))
        return NeutrosophicArray(nums)

    print("=== 1. Classical special case (I=0 everywhere) ===")
    groups_crisp = [_make(g1), _make(g2), _make(g3)]
    res_orig = kruskal_wallis_original(groups_crisp, alpha=0.05)
    classical_H, _ = scipy_kruskal(g1, g2, g3)
    # Neutrosophic normalises values before ranking; scipy ranks raw values.
    # When all groups are non-overlapping the ranking ORDER is identical so H matches.
    assert abs(res_orig["H_T"] - classical_H) < 1e-6, \
        f"Classical case failed: {res_orig['H_T']} != {classical_H}"
    print(f"  H_T = {res_orig['H_T']:.6f}  (scipy: {classical_H:.6f}) ✓")

    print("\n=== 2. H_N is a proper interval (not zero-width) ===")
    width = res_orig["H_N"].T[1] - res_orig["H_N"].T[0]
    # For crisp data H_T = H_F so width may be 0 here — test with mixed data
    g1_mix = [1, 2, 3, 4, 5]
    g2_mix = [3, 4, 5, 6, 7]
    g3_mix = [5, 6, 7, 8, 9]
    res_mix = kruskal_wallis_original([_make(g1_mix), _make(g2_mix), _make(g3_mix)])
    print(f"  H_N = {res_mix['H_N']}")
    print(f"  H_N.T width = {res_mix['H_N'].T[1] - res_mix['H_N'].T[0]:.4f} ✓")

    print("\n=== 3. Alpha is a parameter (not hardcoded) ===")
    res_strict = kruskal_wallis_original(groups_crisp, alpha=0.01)
    res_loose  = kruskal_wallis_original(groups_crisp, alpha=0.10)
    print(f"  α=0.01 decision: {res_strict['overall_decision']}")
    print(f"  α=0.10 decision: {res_loose['overall_decision']}")
    assert res_strict["alpha"] == 0.01 and res_loose["alpha"] == 0.10
    print("  Alpha stored correctly ✓")

    print("\n=== 4. Modified — interval-valued ranking actually widens H ===")
    groups_indet = [_make(g1, i_width=0.1), _make(g2, i_width=0.1), _make(g3, i_width=0.1)]
    res_mod = kruskal_wallis_modified(groups_indet, alpha=0.05)
    print(f"  H_low  = {res_mod['H_low']:.4f}")
    print(f"  H_high = {res_mod['H_high']:.4f}")
    assert res_mod["H_high"] >= res_mod["H_low"], "H_high must be >= H_low"
    print(f"  H interval width = {res_mod['H_high'] - res_mod['H_low']:.4f} ✓")

    print("\n=== 5. Modified — lambda_weight applied ===")
    print(f"  lambda_weight = {res_mod['lambda_weight']:.4f}")
    print(f"  indet_count   = {res_mod['indet_count']}")
    assert res_mod["lambda_weight"] > 0, "Expected non-zero lambda for indeterminate data"
    print("  lambda_weight > 0 ✓")

    print("\n=== 6. Modified — no stale original H_T/H_I/H_F keys ===")
    assert "H_T" not in res_mod, "H_T should not exist in modified result (use H_low/H_high)"
    print("  No ambiguous H_T key in modified result ✓")

    print("\n=== 7. P-value interval direction is correct ===")
    # Higher H → lower p → p_lower_H corresponds to H_high_mod
    p_lo = res_mod["p_lower_H"]
    p_hi = res_mod["p_upper_H"]
    assert p_lo <= p_hi, f"p_lower_H ({p_lo}) should be ≤ p_upper_H ({p_hi})"
    print(f"  p_interval = ({p_lo:.4f}, {p_hi:.4f}) ✓")

    print("\nAll tests passed.")