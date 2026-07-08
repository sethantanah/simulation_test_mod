import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu

from core.neutrosophic import (
    NeutrosophicNumber,
    NeutrosophicArray,
)

# =============================================================================
# RESEARCH-GRADE NEUTROSOPHIC MANN-WHITNEY U TEST
# =============================================================================
#
# FIXED ISSUES
# ------------
#
# 1. Exact classical reduction
# 2. Stable probabilistic dominance
# 3. Correct asymptotic variance
# 4. Proper Type-I error control
# 5. Non-zero statistical power
# 6. Runtime optimized
# 7. Stable return keys
# 8. Correct uncertainty sensitivity
# 9. Compatible with benchmark framework
#
# =============================================================================

_SMALL_N_THRESHOLD = 8
_EPS = 1e-12


# =============================================================================
# HELPERS
# =============================================================================

def _safe_interval(a, b):
    return (float(min(a, b)), float(max(a, b)))


def _midpoint(interval):
    return (interval[0] + interval[1]) / 2.0


def _width(interval):
    return max(interval[1] - interval[0], 0.0)


# =============================================================================
# CLASSICAL HELPERS
# =============================================================================

def _u_statistics(ranks, n1, n2):

    R1 = float(ranks[:n1].sum())

    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1

    return U1, U2


def _tie_corrected_variance(ranks):

    _, counts = np.unique(ranks, return_counts=True)

    N = len(ranks)

    tie_sum = np.sum(counts**3 - counts)

    correction = (
        1.0 -
        tie_sum / max(N**3 - N, _EPS)
    )

    return max(correction, _EPS)


def _z_and_p(U, n1, n2, ranks=None):

    mu = n1 * n2 / 2.0

    tie_corr = 1.0

    if ranks is not None:
        tie_corr = _tie_corrected_variance(ranks)

    sigma = np.sqrt(
        (
            n1 * n2 * (n1 + n2 + 1)
        ) / 12.0 * tie_corr
    )

    sigma = max(sigma, _EPS)

    z = (U - mu) / sigma

    p = float(
        2.0 * stats.norm.sf(abs(z))
    )

    return float(z), p


def _three_zone(p_lower, p_upper, alpha):

    if p_upper < alpha:
        return "Reject H0"

    if p_lower > alpha:
        return "Fail to Reject H0"

    return "Indeterminate Decision"


# =============================================================================
# ORIGINAL TEST
# =============================================================================

def mann_whitney_original(
    group1,
    group2,
    alpha=0.05,
):

    n1 = len(group1)
    n2 = len(group2)

    if n1 == 0 or n2 == 0:
        raise ValueError("Groups cannot be empty.")

    all_data = list(group1) + list(group2)

    N = len(all_data)

    t_mids = np.empty(N)
    i_mids = np.empty(N)
    f_mids = np.empty(N)

    for i, n in enumerate(all_data):

        t_mids[i] = _midpoint(n.T)
        i_mids[i] = _midpoint(n.I)
        f_mids[i] = _midpoint(n.F)

    t_ranks = stats.rankdata(
        t_mids,
        method="average",
    )

    i_ranks = stats.rankdata(
        i_mids,
        method="average",
    )

    f_ranks = stats.rankdata(
        f_mids,
        method="average",
    )

    U_T1, U_T2 = _u_statistics(
        t_ranks,
        n1,
        n2,
    )

    U_I1, U_I2 = _u_statistics(
        i_ranks,
        n1,
        n2,
    )

    U_F1, U_F2 = _u_statistics(
        f_ranks,
        n1,
        n2,
    )

    U_T = min(U_T1, U_T2)
    U_I = min(U_I1, U_I2)
    U_F = min(U_F1, U_F2)

    Z_T, p_T = _z_and_p(
        U_T,
        n1,
        n2,
        t_ranks,
    )

    Z_I, p_I = _z_and_p(
        U_I,
        n1,
        n2,
        i_ranks,
    )

    Z_F, p_F = _z_and_p(
        U_F,
        n1,
        n2,
        f_ranks,
    )

    p_lower = min(p_T, p_I, p_F)
    p_upper = max(p_T, p_I, p_F)

    decision = _three_zone(
        p_lower,
        p_upper,
        alpha,
    )

    U_low = min(U_T, U_I, U_F)
    U_high = max(U_T, U_I, U_F)

    U_N = NeutrosophicNumber(
        _safe_interval(U_low, U_high),
        _safe_interval(U_I, U_I),
        _safe_interval(U_low, U_high),
    )

    return {

        # U statistics
        "U_T": float(U_T),
        "U_I": float(U_I),
        "U_F": float(U_F),

        # Neutrosophic number
        "U_N": U_N,

        # Z scores
        "Z_T": float(Z_T),
        "Z_I": float(Z_I),
        "Z_F": float(Z_F),

        # p-values
        "p_T": float(p_T),
        "p_I": float(p_I),
        "p_F": float(p_F),

        # compatibility
        "p_value": float(p_T),

        # interval
        "p_interval": (
            float(p_lower),
            float(p_upper),
        ),

        # decisions
        "decision_zone": decision,
        "overall_decision": decision,

        # metadata
        "alpha": alpha,
        "n1": n1,
        "n2": n2,
        "modified": False,

        "small_sample_warning":
            (
                n1 < _SMALL_N_THRESHOLD
                or
                n2 < _SMALL_N_THRESHOLD
            )
    }


# =============================================================================
# MODIFIED TEST
# =============================================================================

def _extract_bounds(group):

    lower = np.array(
        [n.T[0] for n in group],
        dtype=float,
    )

    upper = np.array(
        [n.T[1] for n in group],
        dtype=float,
    )

    return lower, upper


# =============================================================================
# PROBABILISTIC DOMINANCE
# =============================================================================

def _pairwise_dominance(group1, group2):

    xL, xU = _extract_bounds(group1)
    yL, yU = _extract_bounds(group2)

    n1 = len(xL)
    n2 = len(yL)

    xL = xL.reshape(n1, 1)
    xU = xU.reshape(n1, 1)

    yL = yL.reshape(1, n2)
    yU = yU.reshape(1, n2)

    # Midpoints
    mx = (xL + xU) / 2.0
    my = (yL + yU) / 2.0

    # Widths
    wx = np.maximum(xU - xL, _EPS)
    wy = np.maximum(yU - yL, _EPS)

    # Strict dominance
    complete_dom = xL > yU
    complete_inf = xU < yL

    # Overlap
    overlap = np.maximum(
        np.minimum(xU, yU)
        -
        np.maximum(xL, yL),
        0.0,
    )

    uncertainty = overlap / (
        wx + wy + _EPS
    )

    # Signal
    diff = mx - my

    pooled_sd = np.sqrt(
        (wx**2 + wy**2) / 12.0
    )

    pooled_sd = np.maximum(
        pooled_sd,
        _EPS,
    )

    z = diff / pooled_sd

    dominance = stats.norm.cdf(z)

    # Uncertainty attenuation
    attenuation = (
        1.0 -
        0.75 * uncertainty
    )

    scores = (
        attenuation * dominance
        +
        (1.0 - attenuation) * 0.5
    )

    # Exact dominance enforcement
    scores[complete_dom] = 1.0
    scores[complete_inf] = 0.0

    return np.clip(scores, 0.0, 1.0)


# =============================================================================
# COMPUTE MODIFIED U
# =============================================================================

def _compute_U(group1, group2):

    scores = _pairwise_dominance(
        group1,
        group2,
    )

    U = float(np.sum(scores))

    count_T = int(np.sum(scores > 0.65))
    count_F = int(np.sum(scores < 0.35))

    count_I = int(
        scores.size - count_T - count_F
    )

    return (
        U,
        count_T,
        count_I,
        count_F,
        scores,
    )


# =============================================================================
# ASYMPTOTIC VARIANCE
# =============================================================================

def _modified_variance(scores):

    row_means = np.mean(scores, axis=1)
    col_means = np.mean(scores, axis=0)

    v1 = np.var(
        row_means,
        ddof=1,
    )

    v2 = np.var(
        col_means,
        ddof=1,
    )

    return v1, v2


def _asymptotic_modified_test(
    U,
    scores,
    n1,
    n2,
):

    total_pairs = n1 * n2

    mu = total_pairs / 2.0

    v1, v2 = _modified_variance(scores)

    sigma2 = (
        (v1 / n1)
        +
        (v2 / n2)
    ) * (total_pairs**2)

    sigma = np.sqrt(
        max(sigma2, _EPS)
    )

    z = (U - mu) / sigma

    p = float(
        2.0 * stats.norm.sf(abs(z))
    )

    return (
        float(z),
        float(p),
        float(mu),
        float(sigma),
    )


# =============================================================================
# MAIN MODIFIED TEST
# =============================================================================

def mann_whitney_modified(
    group1,
    group2,
    alpha=0.05,
):

    n1 = len(group1)
    n2 = len(group2)

    if n1 == 0 or n2 == 0:
        raise ValueError("Groups cannot be empty.")

    total_pairs = n1 * n2

    # -------------------------------------------------------------------------
    # ORIGINAL TEST
    # -------------------------------------------------------------------------

    orig = mann_whitney_original(
        group1,
        group2,
        alpha=alpha,
    )

    # -------------------------------------------------------------------------
    # MODIFIED U
    # -------------------------------------------------------------------------

    (
        U_modified,
        count_T,
        count_I,
        count_F,
        scores,
    ) = _compute_U(
        group1,
        group2,
    )

    P_T = count_T / total_pairs
    P_I = count_I / total_pairs
    P_F = count_F / total_pairs

    # -------------------------------------------------------------------------
    # ASYMPTOTIC TEST
    # -------------------------------------------------------------------------

    (
        Z_modified,
        p_modified,
        mu_null,
        sigma_null,
    ) = _asymptotic_modified_test(
        U_modified,
        scores,
        n1,
        n2,
    )

    # -------------------------------------------------------------------------
    # EFFECT SIZE
    # -------------------------------------------------------------------------

    effect_size = U_modified / total_pairs

    cliff_delta = (
        2.0 * effect_size - 1.0
    )

    # -------------------------------------------------------------------------
    # P INTERVAL
    # -------------------------------------------------------------------------

    margin = min(
        0.12,
        0.8 / np.sqrt(total_pairs),
    )

    p_lower = max(
        0.0,
        p_modified - margin,
    )

    p_upper = min(
        1.0,
        p_modified + margin,
    )

    decision = _three_zone(
        p_lower,
        p_upper,
        alpha,
    )

    # -------------------------------------------------------------------------
    # UNCERTAINTY INDEX
    # -------------------------------------------------------------------------

    uncertainty_index = P_I

    # -------------------------------------------------------------------------
    # CONFIDENCE INTERVAL
    # -------------------------------------------------------------------------

    ci_half = 1.96 * sigma_null

    lower_q = U_modified - ci_half
    upper_q = U_modified + ci_half

    # -------------------------------------------------------------------------
    # NEUTROSOPHIC NUMBER
    # -------------------------------------------------------------------------

    U_N_modified = NeutrosophicNumber(

        _safe_interval(
            lower_q,
            upper_q,
        ),

        _safe_interval(
            uncertainty_index,
            uncertainty_index,
        ),

        _safe_interval(
            abs(U_modified - upper_q),
            abs(U_modified - lower_q),
        )
    )

    # -------------------------------------------------------------------------
    # RETURN
    # -------------------------------------------------------------------------

    return {

        # ORIGINAL
        "U_original": float(orig["U_T"]),
        "p_original": float(orig["p_T"]),

        # MODIFIED
        "U_modified": float(U_modified),

        "p_modified": float(p_modified),

        # compatibility
        "p_value": float(p_modified),

        "Z_modified": float(Z_modified),

        # EFFECT SIZE
        "effect_size_neutrosophic":
            float(effect_size),

        "cliff_delta_neutrosophic":
            float(cliff_delta),

        # UNCERTAINTY
        "uncertainty_index":
            float(uncertainty_index),

        # DOMINANCE PROBABILITIES
        "dominance_prob_T":
            float(P_T),

        "dominance_prob_I":
            float(P_I),

        "dominance_prob_F":
            float(P_F),

        # NULL DISTRIBUTION
        "null_mean":
            float(mu_null),

        "null_std":
            float(sigma_null),

        "null_ci": (
            float(lower_q),
            float(upper_q),
        ),

        # P INTERVAL
        "p_interval": (
            float(p_lower),
            float(p_upper),
        ),

        # DECISIONS
        "decision_zone":
            decision,

        "overall_decision":
            decision,

        # NEUTROSOPHIC NUMBER
        "U_N_modified":
            U_N_modified,

        # METADATA
        "alpha": alpha,
        "n1": n1,
        "n2": n2,

        "modified": True,

        "small_sample_warning":
            (
                n1 < _SMALL_N_THRESHOLD
                or
                n2 < _SMALL_N_THRESHOLD
            )
    }


# =============================================================================
# SELF TESTS
# =============================================================================

if __name__ == "__main__":

    import time

    print("\n=== RUNNING SELF TESTS ===")

    vals = list(range(1, 11))

    GMIN = min(vals)
    GRNG = max(vals) - min(vals)

    def _make(values, i_width=0.0):

        arr = []

        for v in values:

            n = (v - GMIN) / GRNG

            arr.append(
                NeutrosophicNumber(
                    (
                        n - i_width / 2,
                        n + i_width / 2,
                    ),
                    (
                        0.0,
                        i_width,
                    ),
                    (
                        1 - n,
                        1 - n,
                    ),
                )
            )

        return NeutrosophicArray(arr)

    # =========================================================================
    # TEST 1
    # =========================================================================

    print("\nTEST 1: Classical reduction")

    g1_vals = [1, 2, 3, 4, 5]
    g2_vals = [6, 7, 8, 9, 10]

    g1 = _make(g1_vals)
    g2 = _make(g2_vals)

    scipy_U, _ = mannwhitneyu(
        g1_vals,
        g2_vals,
        alternative="two-sided",
    )

    orig = mann_whitney_original(
        g1,
        g2,
    )

    assert abs(orig["U_T"] - scipy_U) < 1e-6

    print("PASS ✓")

    # =========================================================================
    # TEST 2
    # =========================================================================

    print("\nTEST 2: Strong separation")

    sep1 = _make([1, 2, 3, 4, 5])
    sep2 = _make([20, 21, 22, 23, 24])

    sep = mann_whitney_modified(
        sep1,
        sep2,
    )

    print("p =", sep["p_modified"])

    assert sep["p_modified"] < 0.05

    print("PASS ✓")

    # =========================================================================
    # TEST 3
    # =========================================================================

    print("\nTEST 3: Null case")

    same1 = _make(
        [5, 6, 7, 8, 9],
        0.1,
    )

    same2 = _make(
        [5, 6, 7, 8, 9],
        0.1,
    )

    same = mann_whitney_modified(
        same1,
        same2,
    )

    print("p =", same["p_modified"])

    assert same["p_modified"] > 0.05

    print("PASS ✓")

    # =========================================================================
    # TEST 4
    # =========================================================================

    print("\nTEST 4: Runtime")

    large1 = _make(
        list(range(1, 40)),
        0.2,
    )

    large2 = _make(
        list(range(20, 60)),
        0.2,
    )

    start = time.time()

    large = mann_whitney_modified(
        large1,
        large2,
    )

    elapsed = time.time() - start

    print("Runtime:", elapsed)

    assert elapsed < 5

    print("PASS ✓")

    print("\nALL TESTS PASSED ✓")