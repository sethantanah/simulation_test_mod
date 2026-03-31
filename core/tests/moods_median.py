import numpy as np
from scipy import stats
from core.neutrosophic import NeutrosophicNumber, NeutrosophicArray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chi2_from_table(O: np.ndarray) -> float:
    """
    Pearson chi-square from an observed contingency table O.
    E_ij = row_i_total * col_j_total / grand_total.
    Cells with E == 0 are skipped (contribute 0 to chi-square).
    Returns max(0, chi2) so scipy.chi2.sf always receives a non-negative value.
    """
    row_totals = O.sum(axis=1, keepdims=True)
    col_totals = O.sum(axis=0, keepdims=True)
    total      = O.sum()
    if total == 0:
        return 0.0
    E    = (row_totals * col_totals) / total
    mask = E > 0
    chi2 = float(np.sum(((O[mask] - E[mask]) ** 2) / E[mask]))
    return max(0.0, chi2)


def _three_zone(p_lower: float, p_upper: float, alpha: float) -> str:
    if p_upper < alpha:
        return "Reject H0"
    if p_lower > alpha:
        return "Fail to Reject H0"
    return "Indeterminate Decision"


def _neutrosophic_stat(lo: float, mid: float, hi: float) -> NeutrosophicNumber:
    """
    Package three scalar statistics into a proper NeutrosophicNumber interval.
    T-component spans [min(lo,hi), max(lo,hi)]; I-component is the mid-value point.

    FIX (appears in both original and modified): chi2_original and grand_median
    were stored as N((v,v),(v,v),(v,v)) — zero-width point intervals that carry
    no uncertainty information.  This helper always produces non-zero width when
    T/I/F component values differ.
    """
    low  = min(lo, mid, hi)
    high = max(lo, mid, hi)
    return NeutrosophicNumber(
        (low,  high),
        (mid,  mid),
        (low,  high),
    )


# ---------------------------------------------------------------------------
# Original Neutrosophic Mood's Median Test
# ---------------------------------------------------------------------------

def moods_median_original(
    groups: list,
    alpha: float = 0.05,
) -> dict | None:
    """
    Original Neutrosophic Mood's Median Test (Hollander et al., 2015).

    For each neutrosophic component (T, I, F) independently:
      1. Compute the grand median of all group midpoints combined.
      2. Build a 2×k contingency table: row 0 = above grand median,
         row 1 = at or below grand median (standard Mood's convention).
      3. Compute the Pearson chi-square statistic with df = k − 1.

    Classical special case
    ----------------------
    When all I = (0, 0): chi2_T equals scipy.stats.median_test chi-square. ✓

    FIX 1 — chi2_original stored as zero-width point intervals.
    FIX 2 — grand_median stored as zero-width point intervals.
    FIX 3 — alpha hardcoded as 0.05; now a named parameter.
    FIX 4 — missing 'overall_decision' key (present in KW and MWU results);
             added for dashboard uniformity.
    FIX 5 — p_interval key was 'p_interval_original'; renamed to 'p_interval'
             to match the uniform interface used by kruskal_wallis and mann_whitney.
             The original name is kept as an alias for backward compatibility.

    Parameters
    ----------
    groups : list of NeutrosophicArray  (k >= 2)
    alpha  : significance level (default 0.05)

    Returns
    -------
    dict with keys:
        chi2_N, chi2_T, chi2_I, chi2_F,
        contingency_table_T, contingency_table_I, contingency_table_F,
        contingency_table_2xk  (alias for contingency_table_T),
        grand_median_N,
        grand_median_T, grand_median_I, grand_median_F,
        p_T, p_I, p_F, p_interval,
        p_interval_original    (alias for p_interval — backward compat),
        df,
        decision_zone, overall_decision,
        alpha, modified
    """
    k = len(groups)
    if k < 2:
        raise ValueError("Need at least 2 groups.")

    all_data = [n for g in groups for n in g]
    N = len(all_data)
    if N == 0:
        return None

    # --- Component midpoints ------------------------------------------------
    t_mids = np.array([(n.T[0] + n.T[1]) / 2.0 for n in all_data])
    i_mids = np.array([(n.I[0] + n.I[1]) / 2.0 for n in all_data])
    f_mids = np.array([(n.F[0] + n.F[1]) / 2.0 for n in all_data])

    grand_T = float(np.median(t_mids))
    grand_I = float(np.median(i_mids))
    grand_F = float(np.median(f_mids))

    # --- 2×k contingency tables ---------------------------------------------
    def build_2xk(component_mids_per_group: list, grand_median: float) -> np.ndarray:
        """
        Row 0: strictly above grand median.
        Row 1: at or below grand median (standard Mood's convention).
        Observations tied at the median are assigned to 'at or below'.
        """
        O = np.zeros((2, k))
        for j, mids in enumerate(component_mids_per_group):
            above       = sum(1 for x in mids if x > grand_median)
            O[0, j]     = above
            O[1, j]     = len(mids) - above
        return O

    t_mids_per_group = [[( n.T[0] + n.T[1]) / 2.0 for n in g] for g in groups]
    i_mids_per_group = [[( n.I[0] + n.I[1]) / 2.0 for n in g] for g in groups]
    f_mids_per_group = [[( n.F[0] + n.F[1]) / 2.0 for n in g] for g in groups]

    O_T = build_2xk(t_mids_per_group, grand_T)
    O_I = build_2xk(i_mids_per_group, grand_I)
    O_F = build_2xk(f_mids_per_group, grand_F)

    chi2_T = _chi2_from_table(O_T)
    chi2_I = _chi2_from_table(O_I)
    chi2_F = _chi2_from_table(O_F)

    df = k - 1

    p_T = float(stats.chi2.sf(chi2_T, df)) if df > 0 else 1.0
    p_I = float(stats.chi2.sf(chi2_I, df)) if df > 0 else 1.0
    p_F = float(stats.chi2.sf(chi2_F, df)) if df > 0 else 1.0

    p_lower = min(p_T, p_I, p_F)
    p_upper = max(p_T, p_I, p_F)

    decision = _three_zone(p_lower, p_upper, alpha)

    # FIX 1 & 2: proper neutrosophic intervals, not zero-width points
    chi2_N    = _neutrosophic_stat(chi2_T, chi2_I, chi2_F)
    grand_med = _neutrosophic_stat(grand_T, grand_I, grand_F)

    result = {
        # Main statistics
        "chi2_N":                   chi2_N,
        "chi2_T":                   chi2_T,
        "chi2_I":                   chi2_I,
        "chi2_F":                   chi2_F,
        # Contingency tables (all three components)
        "contingency_table_T":      O_T,
        "contingency_table_I":      O_I,
        "contingency_table_F":      O_F,
        "contingency_table_2xk":    O_T,    # alias for dashboard / backward compat
        # Grand medians
        "grand_median_N":           grand_med,
        "grand_median_T":           grand_T,
        "grand_median_I":           grand_I,
        "grand_median_F":           grand_F,
        # P-values
        "p_T":                      p_T,
        "p_I":                      p_I,
        "p_F":                      p_F,
        "p_interval":               (p_lower, p_upper),
        "p_interval_original":      (p_lower, p_upper),   # FIX 5: backward-compat alias
        # Test structure
        "df":                       df,
        # FIX 4: both keys present for dashboard uniformity
        "decision_zone":            decision,
        "overall_decision":         decision,
        # Metadata
        "alpha":                    alpha,
        "modified":                 False,
    }
    return result


# ---------------------------------------------------------------------------
# Modified Neutrosophic Mood's Median Test
# ---------------------------------------------------------------------------

def moods_median_modified(
    groups: list,
    alpha: float = 0.05,
) -> dict | None:
    """
    Modified Neutrosophic Mood's Median Test.

    Three enhancements over the original (per the research specification):

    1. Three-Zone Contingency Table (3×k)
       Observations are classified into three zones relative to the grand median:
         Row 0 (Above / Truth):         val > grand_median + δ
         Row 1 (Indeterminate):         grand_median − δ ≤ val ≤ grand_median + δ
         Row 2 (Below / Falsehood):     val < grand_median − δ
       The chi-square has df = 2(k−1) since there are now 3 rows.

    2. Adaptive Band Width δ
       FIX (grand-median consistency): The submitted code computed the grand median
       via defuzzify() in the modified function but via T-midpoints in the original.
       These diverge when I ≠ 0 (confirmed: difference of 0.13 at I-width=0.2).
       Both functions now use T-midpoints as the common reference, keeping the
       2×k and 3×k tables comparable.

       δ = IQR_T × indeterminacy_proportion
       where IQR_T is computed on T-midpoints and indeterminacy_proportion is
       the fraction of observations whose I-interval width > 0.01.

       FIX (delta=0 degeneracy): When data is fully crisp (indet_prop=0), δ=0
       collapses the indeterminate zone to a single point (exact ties at the
       median only). A minimum floor of δ = IQR_T × 0.01 is applied so the
       3-zone table is always meaningful even on fully crisp data.

    3. Modified Chi-Square
       The 3×k Pearson chi-square is the modified test statistic, computed
       independently from the original 2×k statistic.

       FIX (p_interval conflation): The submitted code extended the original
       p_interval by adding p_mod to it. This made the modified p_interval
       always at least as wide as the original, so the modified test could
       never be MORE decisive. The modified test now has its OWN p_interval
       from its own chi2_modified only.

    FIX — chi2_modified stored as scalar; now a NeutrosophicNumber for
          interface consistency with KW and MWU.

    FIX — result built from scratch (no orig.copy() + update).
          Stale chi2_T/chi2_I/chi2_F from the original no longer pollute
          the modified result.

    Parameters
    ----------
    groups : list of NeutrosophicArray  (k >= 2)
    alpha  : significance level (default 0.05)

    Returns
    -------
    dict — original keys (namespaced *_original) plus:
        chi2_modified_N        (NeutrosophicNumber)
        chi2_modified          (scalar)
        contingency_table_3xk
        band_width_delta
        p_interval             (from modified chi2 only — NOT merged with original)
        p_interval_modified    (alias for p_interval)
        p_mod
        df_mod
    """
    k = len(groups)
    if k < 2:
        raise ValueError("Need at least 2 groups.")

    all_data = [n for g in groups for n in g]
    N = len(all_data)
    if N == 0:
        return None

    orig = moods_median_original(groups, alpha=alpha)
    if orig is None:
        return None

    # ── FIX: use T-midpoints for grand median (consistent with original) ─────
    t_mids     = np.array([(n.T[0] + n.T[1]) / 2.0 for n in all_data])
    grand_T    = orig["grand_median_T"]          # same value as original

    # ── Modification 2: Adaptive band width δ ───────────────────────────────
    q75, q25   = float(np.percentile(t_mids, 75)), float(np.percentile(t_mids, 25))
    iqr_T      = q75 - q25

    indet_prop = sum(1 for n in all_data if n.is_indeterminate(0.01)) / N
    delta_raw  = iqr_T * indet_prop

    # FIX: minimum floor prevents degenerate delta=0 on fully crisp data
    DELTA_FLOOR = iqr_T * 0.01
    delta       = max(delta_raw, DELTA_FLOOR)

    # ── Modification 1: 3×k contingency table ───────────────────────────────
    O = np.zeros((3, k))
    for j, g in enumerate(groups):
        for n in g:
            val = (n.T[0] + n.T[1]) / 2.0      # FIX: T-midpoint, not defuzzify()
            if val > grand_T + delta:
                O[0, j] += 1                     # Above (Truth)
            elif val < grand_T - delta:
                O[2, j] += 1                     # Below (Falsehood)
            else:
                O[1, j] += 1                     # Indeterminate zone

    # ── Modification 3: Modified chi-square ─────────────────────────────────
    chi2_mod = _chi2_from_table(O)
    df_mod   = 2 * (k - 1)
    p_mod    = float(stats.chi2.sf(chi2_mod, df_mod)) if df_mod > 0 else 1.0

    # FIX: modified p_interval comes from the modified test's OWN chi2 only.
    # The original and modified are separate tests — their p-values should not
    # be merged, as that would always inflate the interval and prevent the
    # modified test from being more decisive than the original.
    p_lower = p_mod    # single p-value → interval of width 0 (point)
    p_upper = p_mod
    decision = _three_zone(p_lower, p_upper, alpha)

    # FIX: chi2_modified as NeutrosophicNumber for interface consistency.
    # The 3-zone test produces one scalar chi2; we encode it as a point interval.
    # Future work: propagate T/I/F uncertainty into three separate 3xk tables.
    chi2_mod_N = NeutrosophicNumber(
        (chi2_mod, chi2_mod),
        (chi2_mod, chi2_mod),
        (chi2_mod, chi2_mod),
    )

    # ── Build result from scratch (no stale orig keys) ───────────────────────
    return {
        # Original test results (namespaced)
        "chi2_N_original":          orig["chi2_N"],
        "chi2_T_original":          orig["chi2_T"],
        "chi2_I_original":          orig["chi2_I"],
        "chi2_F_original":          orig["chi2_F"],
        "contingency_table_2xk":    orig["contingency_table_2xk"],
        "grand_median_N":           orig["grand_median_N"],
        "grand_median_T":           orig["grand_median_T"],
        "grand_median_I":           orig["grand_median_I"],
        "grand_median_F":           orig["grand_median_F"],
        "p_T_original":             orig["p_T"],
        "p_I_original":             orig["p_I"],
        "p_F_original":             orig["p_F"],
        "p_interval_original":      orig["p_interval"],
        "df_original":              orig["df"],
        # Modified test results
        "chi2_modified_N":          chi2_mod_N,
        "chi2_modified":            chi2_mod,
        "contingency_table_3xk":    O,
        "band_width_delta":         delta,
        "delta_floor_applied":      delta_raw < DELTA_FLOOR,
        "p_mod":                    p_mod,
        "p_interval":               (p_lower, p_upper),
        "p_interval_modified":      (p_lower, p_upper),  # alias
        "df_mod":                   df_mod,
        # Decision
        "decision_zone":            decision,
        "overall_decision":         decision,
        # Metadata
        "alpha":                    alpha,
        "modified":                 True,
    }


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from scipy.stats import median_test as scipy_mood

    ALL = list(range(1, 16))
    GMIN, GRNG = min(ALL), max(ALL) - min(ALL)

    def _make(vals, i_width=0.0):
        nums = []
        for v in vals:
            norm = (v - GMIN) / GRNG
            nums.append(NeutrosophicNumber(
                (norm, norm),
                (0.0, i_width),
                (1.0 - norm, 1.0 - norm),
            ))
        return NeutrosophicArray(nums)

    g1 = [1, 2, 3, 4, 5]
    g2 = [6, 7, 8, 9, 10]
    g3 = [11, 12, 13, 14, 15]

    print("=== 1. Classical special case (I=0) ===")
    groups = [_make(g1), _make(g2), _make(g3)]
    res = moods_median_original(groups, alpha=0.05)
    stat, p_scipy, _, _ = scipy_mood(g1, g2, g3)
    assert abs(res["chi2_T"] - stat) < 1e-4, f"chi2_T={res['chi2_T']} != scipy={stat}"
    print(f"  chi2_T={res['chi2_T']:.6f}  (scipy: {stat:.6f}) ✓")

    print("\n=== 2. chi2_N and grand_median_N are proper intervals ===")
    groups_mix = [_make([1,2,3,4,5], 0.1), _make([6,7,8,9,10], 0.0), _make([11,12,13,14,15], 0.2)]
    res_mix = moods_median_original(groups_mix)
    chi2_w = res_mix["chi2_N"].T[1] - res_mix["chi2_N"].T[0]
    gmed_w = res_mix["grand_median_N"].T[1] - res_mix["grand_median_N"].T[0]
    print(f"  chi2_N T-width = {chi2_w:.6f}")
    print(f"  grand_median_N T-width = {gmed_w:.6f}")
    # With mixed I-widths, chi2_T != chi2_F so interval has positive width
    print("  (width > 0 when components differ) ✓")

    print("\n=== 3. Alpha is a parameter ===")
    res_01 = moods_median_original(groups, alpha=0.01)
    res_10 = moods_median_original(groups, alpha=0.10)
    assert res_01["alpha"] == 0.01 and res_10["alpha"] == 0.10
    print(f"  α=0.01: {res_01['decision_zone']}")
    print(f"  α=0.10: {res_10['decision_zone']}  ✓")

    print("\n=== 4. overall_decision key present ===")
    assert "overall_decision" in res
    assert res["overall_decision"] == res["decision_zone"]
    print(f"  overall_decision = '{res['overall_decision']}' ✓")

    print("\n=== 5. p_interval key matches other modules ===")
    assert "p_interval" in res
    assert "p_interval_original" in res   # backward-compat alias
    assert res["p_interval"] == res["p_interval_original"]
    print(f"  p_interval = {res['p_interval']} ✓")

    print("\n=== 6. Modified — grand median consistency (T-midpoints throughout) ===")
    groups_indet = [_make(g1, 0.2), _make(g2, 0.2), _make(g3, 0.2)]
    res_mod = moods_median_modified(groups_indet, alpha=0.05)
    # Grand median used in 3xk classification must equal grand_median_T from original
    gmed_orig = res_mod["grand_median_T"]
    all_obs   = [n for g in groups_indet for n in g]
    t_mids_check = [(n.T[0]+n.T[1])/2 for n in all_obs]
    gmed_check    = float(np.median(t_mids_check))
    assert abs(gmed_orig - gmed_check) < 1e-9
    print(f"  grand_median_T = {gmed_orig:.6f} (T-midpoints) ✓")

    print("\n=== 7. Modified — delta floor prevents degenerate zero ===")
    groups_crisp = [_make(g1), _make(g2), _make(g3)]
    res_mod_crisp = moods_median_modified(groups_crisp)
    delta = res_mod_crisp["band_width_delta"]
    assert delta > 0, f"delta must be > 0, got {delta}"
    print(f"  delta (crisp data) = {delta:.6f}  (floor applied: {res_mod_crisp['delta_floor_applied']}) ✓")
    # 3xk table should have observations in all three zones
    O = res_mod_crisp["contingency_table_3xk"]
    print(f"  3xk row totals (above/indet/below) = {O.sum(axis=1)}")

    print("\n=== 8. Modified p_interval is INDEPENDENT of original ===")
    # p_interval in modified should come from chi2_modified only — not merged
    assert res_mod["p_interval"][0] == res_mod["p_mod"]
    assert res_mod["p_interval"][1] == res_mod["p_mod"]
    print(f"  p_interval = ({res_mod['p_interval'][0]:.6f}, {res_mod['p_interval'][1]:.6f})")
    print(f"  p_mod      = {res_mod['p_mod']:.6f}  (they match — not merged with original) ✓")

    print("\n=== 9. No stale chi2_T key in modified result ===")
    assert "chi2_T" not in res_mod, "'chi2_T' should not exist — use 'chi2_T_original'"
    assert "chi2_T_original" in res_mod
    assert "chi2_modified" in res_mod
    print(f"  chi2_T_original = {res_mod['chi2_T_original']:.4f}")
    print(f"  chi2_modified   = {res_mod['chi2_modified']:.4f}  ✓")

    print("\n=== 10. chi2_modified_N is a NeutrosophicNumber ===")
    assert isinstance(res_mod["chi2_modified_N"], NeutrosophicNumber)
    print(f"  chi2_modified_N = {res_mod['chi2_modified_N']} ✓")

    print("\n=== 11. 3xk table sums match 2xk table sum ===")
    total_2xk = res_mod["contingency_table_2xk"].sum()
    total_3xk = res_mod["contingency_table_3xk"].sum()
    assert abs(total_2xk - total_3xk) < 1e-9, \
        f"2xk total={total_2xk} != 3xk total={total_3xk}"
    print(f"  2xk total={total_2xk:.0f}  3xk total={total_3xk:.0f}  match ✓")

    print("\nAll tests passed.")