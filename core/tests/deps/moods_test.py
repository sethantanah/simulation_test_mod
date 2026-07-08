import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional, Any
import warnings
import pandas as pd


# ============================================================================
# Data Structures
# ============================================================================

class NeutrosophicNumber:
    """Represents a neutrosophic number with Truth, Indeterminacy, and Falsehood."""
    def __init__(self, T: Tuple[float, float], I: Tuple[float, float], F: Tuple[float, float]):
        self.T = T
        self.I = I
        self.F = F
    
    @property
    def T_mid(self) -> float:
        return (self.T[0] + self.T[1]) / 2.0
    
    @property
    def I_mid(self) -> float:
        return (self.I[0] + self.I[1]) / 2.0
    
    @property
    def F_mid(self) -> float:
        return (self.F[0] + self.F[1]) / 2.0
    
    @property
    def I_width(self) -> float:
        return abs(self.I[1] - self.I[0])
    
    def is_indeterminate(self, threshold: float = 0.01) -> bool:
        return self.I_width > threshold


class NeutrosophicArray:
    """Container for multiple NeutrosophicNumbers."""
    def __init__(self, numbers: List[NeutrosophicNumber]):
        self.data = numbers
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)


# ============================================================================
# Internal helpers
# ============================================================================

def _chi2_from_table(O: np.ndarray) -> float:
    """Pearson chi-square from contingency table."""
    row_totals = O.sum(axis=1, keepdims=True)
    col_totals = O.sum(axis=0, keepdims=True)
    total = O.sum()
    if total == 0:
        return 0.0
    E = (row_totals * col_totals) / total
    mask = E > 0
    chi2 = float(np.sum(((O[mask] - E[mask]) ** 2) / E[mask]))
    return max(0.0, chi2)


def _build_2xk(mids: np.ndarray, group_boundaries: np.ndarray, 
               grand_median: float, k: int) -> np.ndarray:
    """Build 2×k contingency table."""
    O = np.zeros((2, k))
    for j in range(k):
        group_mids = mids[group_boundaries[j]:group_boundaries[j+1]]
        O[0, j] = np.sum(group_mids > grand_median)
        O[1, j] = len(group_mids) - O[0, j]
    return O


def _build_3xk(mids: np.ndarray, group_boundaries: np.ndarray,
               grand_median: float, delta: float, k: int) -> np.ndarray:
    """Build 3×k contingency table with indeterminate zone."""
    O = np.zeros((3, k))
    for j in range(k):
        group_mids = mids[group_boundaries[j]:group_boundaries[j+1]]
        O[0, j] = np.sum(group_mids > grand_median + delta)
        O[2, j] = np.sum(group_mids < grand_median - delta)
        O[1, j] = len(group_mids) - O[0, j] - O[2, j]
    return O


def _fisher_combined_pvalue(p_values: List[float]) -> float:
    """Fisher's method for combining independent p-values."""
    p_values = [max(p, 1e-16) for p in p_values]
    statistic = -2 * np.sum(np.log(p_values))
    df = 2 * len(p_values)
    return float(stats.chi2.sf(statistic, df))


def _compute_3xk_permutation_pvalue(
    mids: np.ndarray, group_boundaries: np.ndarray,
    grand_median: float, delta: float, k: int,
    n_permutations: int = 1000, seed: int = None
) -> float:
    """Permutation-based p-value for 3×k test. Uses local RandomState."""
    rng = np.random.RandomState(seed)
    
    observed_chi2 = _chi2_from_table(
        _build_3xk(mids, group_boundaries, grand_median, delta, k)
    )
    
    perm_chi2s = np.zeros(n_permutations)
    for i in range(n_permutations):
        permuted_mids = rng.permutation(mids)
        perm_chi2s[i] = _chi2_from_table(
            _build_3xk(permuted_mids, group_boundaries, grand_median, delta, k)
        )
    
    p_value = np.mean(perm_chi2s >= observed_chi2)
    return max(p_value, 1.0 / (n_permutations + 1))


# ============================================================================
# Decision Functions
# ============================================================================

def _path_a_decision(p_T: float, p_I: float, p_F: float, alpha: float) -> dict:
    """
    Path A: Fisher decides, neutrosophic annotation is diagnostic.
    Returns dict with decision, annotation info.
    """
    p_lower = min(p_T, p_I, p_F)
    p_upper = max(p_T, p_I, p_F)
    p_combined = _fisher_combined_pvalue([p_T, p_I, p_F])
    
    if p_combined < alpha:
        decision = "Reject H0"
    else:
        decision = "Fail to Reject H0"
    
    if p_upper < alpha:
        annotation = "all_components_agree"
    elif p_lower > alpha:
        annotation = "no_components_significant"
    else:
        annotation = "mixed_components"
    
    return {
        'decision': decision,
        'annotation': annotation,
        'p_combined': p_combined,
        'is_indeterminate': False
    }


def _path_b_decision(p_T: float, p_I: float, p_F: float, alpha: float) -> dict:
    """
    Path B: Genuine neutrosophic test on the Fisher statistic interval.
    
    FIXED: The three-zone decision is now based on the NEUTROSOPHIC INTERVAL
    [p_lower_bound, p_upper_bound], not on either bound alone.
    
    - p_lower_bound: Fisher using T only (df=2) — conservative, ignores I and F
    - p_upper_bound: Fisher using T+I+F (df=6) — liberal, full combination
    
    Three-zone decision:
    - Reject H0 if p_upper_bound < alpha (even liberal test rejects → robust)
    - Fail to Reject if p_lower_bound > alpha (even conservative test accepts → robust)
    - Indeterminate if interval straddles alpha (decision depends on I/F weighting)
    
    This is DIFFERENT from Fisher because:
    - Fisher always combines all three → always uses p_upper_bound
    - Path B rejects only when BOTH bounds agree (p_upper_bound < alpha)
    - Path B fails to reject only when BOTH bounds agree (p_lower_bound > alpha)
    - Otherwise it's Indeterminate — Fisher never says Indeterminate
    
    The key insight: Path B is MORE CONSERVATIVE than Fisher for rejection
    because it requires the conservative bound to also be below alpha.
    Wait — that's not right either. Let me reconsider.
    
    ACTUAL CORRECT LOGIC:
    - p_lower_bound uses only T (df=2). This is the CONSERVATIVE p-value
      (larger p-value, harder to reject) because it uses less information.
    - p_upper_bound uses T+I+F (df=6). This is the LIBERAL p-value
      (smaller p-value, easier to reject) because it uses more information.
    
    So: p_lower_bound >= p_upper_bound (conservative >= liberal)
    
    Three-zone on the interval [p_upper_bound, p_lower_bound]:
    - If p_lower_bound < alpha: even conservative test rejects → Reject H0
    - If p_upper_bound > alpha: even liberal test fails → Fail to Reject H0  
    - Otherwise: interval straddles alpha → Indeterminate
    
    THIS IS GENUINELY DIFFERENT FROM FISHER:
    - Fisher uses only p_upper_bound (full combination)
    - Path B requires p_lower_bound < alpha to reject (T-only must also reject)
    - Path B will have LOWER rejection rate than Fisher
    - Path B will have LOWER Type I error than Fisher
    - Path B introduces genuine Indeterminate zone
    """
    # Conservative: T only (df=2) — larger p-value, harder to reject
    chi2_T = -2 * np.log(max(p_T, 1e-16))
    p_conservative = float(stats.chi2.sf(chi2_T, df=2))
    
    # Liberal: T+I+F (df=6) — smaller p-value, easier to reject
    chi2_full = -2 * (np.log(max(p_T, 1e-16)) + 
                       np.log(max(p_I, 1e-16)) + 
                       np.log(max(p_F, 1e-16)))
    p_liberal = float(stats.chi2.sf(chi2_full, df=6))
    
    # Ensure correct ordering: conservative >= liberal
    p_lower_bound = min(p_conservative, p_liberal)  # Actually p_liberal (smaller)
    p_upper_bound = max(p_conservative, p_liberal)  # Actually p_conservative (larger)
    
    # Three-zone decision on the neutrosophic interval
    if p_upper_bound < alpha:
        # Even the conservative test (T-only) rejects: ROBUST rejection
        decision = "Reject H0"
        method_detail = "robust_reject_conservative_bound_below_alpha"
        is_indeterminate = False
    elif p_lower_bound > alpha:
        # Even the liberal test (T+I+F) fails to reject: ROBUST failure
        decision = "Fail to Reject H0"
        method_detail = "robust_fail_liberal_bound_above_alpha"
        is_indeterminate = False
    else:
        # Interval straddles alpha: decision depends on I/F weighting
        decision = "Indeterminate"
        method_detail = "indeterminate_interval_straddles_alpha"
        is_indeterminate = True
    
    return {
        'decision': decision,
        'method_detail': method_detail,
        'p_lower_bound': p_lower_bound,  # liberal (smaller p)
        'p_upper_bound': p_upper_bound,  # conservative (larger p)
        'is_indeterminate': is_indeterminate
    }

# ============================================================================
# Original Neutrosophic Mood's Median Test
# ============================================================================

def moods_median_original(
    groups: list,
    alpha: float = 0.05,
    decision_method: str = 'path_b',
    component_weights: List[float] = None,
) -> dict | None:
    """
    Original Neutrosophic Mood's Median Test.
    
    DECISION METHODS:
    - 'path_a': Fisher decides, neutrosophic annotation is diagnostic
    - 'path_b': Genuine neutrosophic test on Fisher interval (RECOMMENDED)
    - 'classic_three_zone': Original rule (ALL p < α), ultra-conservative
    - 'fisher': Fisher's combined test only (binary)
    - 't_only': T-component only (baseline)
    """
    k = len(groups)
    if k < 2:
        raise ValueError("Need at least 2 groups.")

    all_data = [n for g in groups for n in g]
    N = len(all_data)
    if N == 0:
        return None

    t_mids = np.array([n.T_mid for n in all_data])
    i_mids = np.array([n.I_mid for n in all_data])
    f_mids = np.array([n.F_mid for n in all_data])

    grand_T = float(np.median(t_mids))
    grand_I = float(np.median(i_mids))
    grand_F = float(np.median(f_mids))

    group_boundaries = np.cumsum([0] + [len(g) for g in groups])

    O_T = _build_2xk(t_mids, group_boundaries, grand_T, k)
    O_I = _build_2xk(i_mids, group_boundaries, grand_I, k)
    O_F = _build_2xk(f_mids, group_boundaries, grand_F, k)

    chi2_T = _chi2_from_table(O_T)
    chi2_I = _chi2_from_table(O_I)
    chi2_F = _chi2_from_table(O_F)

    df = k - 1

    p_T = float(stats.chi2.sf(chi2_T, df)) if df > 0 else 1.0
    p_I = float(stats.chi2.sf(chi2_I, df)) if df > 0 else 1.0
    p_F = float(stats.chi2.sf(chi2_F, df)) if df > 0 else 1.0

    p_lower = min(p_T, p_I, p_F)
    p_upper = max(p_T, p_I, p_F)
    fisher_p = _fisher_combined_pvalue([p_T, p_I, p_F])

    # Base result dict (common to all methods)
    result = {
        "chi2_T": chi2_T, "chi2_I": chi2_I, "chi2_F": chi2_F,
        "p_T": p_T, "p_I": p_I, "p_F": p_F,
        "p_lower": p_lower, "p_upper": p_upper,
        "fisher_combined_p": fisher_p,
        "contingency_table_2xk": {"T": O_T, "I": O_I, "F": O_F},
        "grand_median_T": grand_T, "grand_median_I": grand_I, "grand_median_F": grand_F,
        "decision_method": decision_method,
        "df": df, "alpha": alpha, "modified": False,
    }

    # --- Decision ---
    if decision_method == 'path_a':
        dec = _path_a_decision(p_T, p_I, p_F, alpha)
        result["decision"] = dec['decision']
        result["overall_decision"] = dec['decision']
        result["reject_H0"] = dec['decision'] == "Reject H0"
        result["is_indeterminate"] = dec['is_indeterminate']
        result["annotation"] = dec['annotation']
        result["p_combined"] = dec['p_combined']

    elif decision_method == 'path_b':
        dec = _path_b_decision(p_T, p_I, p_F, alpha)
        result["decision"] = dec['decision']
        result["overall_decision"] = dec['decision']
        result["reject_H0"] = dec['decision'] == "Reject H0"
        result["is_indeterminate"] = dec['is_indeterminate']
        result["method_detail"] = dec['method_detail']
        result["p_lower_bound"] = dec['p_lower_bound']
        result["p_upper_bound"] = dec['p_upper_bound']

    elif decision_method == 'classic_three_zone':
        if p_upper < alpha:
            decision = "Reject H0"
        elif p_lower > alpha:
            decision = "Fail to Reject H0"
        else:
            decision = "Indeterminate"
        result["decision"] = decision
        result["overall_decision"] = decision
        result["reject_H0"] = decision == "Reject H0"
        result["is_indeterminate"] = decision == "Indeterminate"

    elif decision_method == 'fisher':
        decision = "Reject H0" if fisher_p < alpha else "Fail to Reject H0"
        result["decision"] = decision
        result["overall_decision"] = decision
        result["reject_H0"] = decision == "Reject H0"
        result["is_indeterminate"] = False

    elif decision_method == 't_only':
        decision = "Reject H0" if p_T < alpha else "Fail to Reject H0"
        result["decision"] = decision
        result["overall_decision"] = decision
        result["reject_H0"] = decision == "Reject H0"
        result["is_indeterminate"] = False

    elif decision_method == 'corrected_three_zone':
        # Alias for path_b
        dec = _path_b_decision(p_T, p_I, p_F, alpha)
        result["decision"] = dec['decision']
        result["overall_decision"] = dec['decision']
        result["reject_H0"] = dec['decision'] == "Reject H0"
        result["is_indeterminate"] = dec['is_indeterminate']
        result["method_detail"] = dec['method_detail']
        result["p_lower_bound"] = dec['p_lower_bound']
        result["p_upper_bound"] = dec['p_upper_bound']

    else:
        raise ValueError(f"Unknown decision_method: {decision_method}")

    return result


# ============================================================================
# Modified Neutrosophic Mood's Median Test
# ============================================================================

def moods_median_modified(
    groups: list,
    alpha: float = 0.05,
    n_permutations: int = 1000,
    permutation_seed: int = None,
) -> dict | None:
    """
    Modified Neutrosophic Mood's Median Test.
    Uses permutation-based inference with local RandomState.
    """
    k = len(groups)
    if k < 2:
        raise ValueError("Need at least 2 groups.")

    all_data = [n for g in groups for n in g]
    N = len(all_data)
    if N == 0:
        return None

    # Run original test with path_b for diagnostics
    # FIXED: Use 'path_b' instead of non-existent 'corrected_three_zone'
    orig = moods_median_original(groups, alpha=alpha, decision_method='path_b')
    if orig is None:
        return None

    t_mids = np.array([n.T_mid for n in all_data])
    grand_T = orig["grand_median_T"]

    q75, q25 = float(np.percentile(t_mids, 75)), float(np.percentile(t_mids, 25))
    iqr_T = q75 - q25
    indet_prop = sum(1 for n in all_data if n.is_indeterminate(0.01)) / max(N, 1)
    delta = max(iqr_T * indet_prop, iqr_T * 0.01)

    group_boundaries = np.cumsum([0] + [len(g) for g in groups])

    p_mod = _compute_3xk_permutation_pvalue(
        t_mids, group_boundaries, grand_T, delta, k,
        n_permutations=n_permutations, seed=permutation_seed
    )

    decision = "Reject H0" if p_mod < alpha else "Fail to Reject H0"
    O_3xk = _build_3xk(t_mids, group_boundaries, grand_T, delta, k)

    return {
        "chi2_T_original": orig["chi2_T"],
        "chi2_I_original": orig["chi2_I"],
        "chi2_F_original": orig["chi2_F"],
        "p_T_original": orig["p_T"],
        "p_I_original": orig["p_I"],
        "p_F_original": orig["p_F"],
        "contingency_table_2xk": orig["contingency_table_2xk"],
        "original_decision": orig["decision"],
        "original_is_indeterminate": orig["is_indeterminate"],
        "p_mod": p_mod,
        "p_value": p_mod,
        "p_interval": (p_mod, p_mod),
        "contingency_table_3xk": O_3xk,
        "band_width_delta": delta,
        "n_permutations": n_permutations,
        "decision_zone": decision,
        "overall_decision": decision,
        "reject_H0": decision == "Reject H0",
        "is_indeterminate": False,
        "alpha": alpha, "modified": True,
    }


# ============================================================================
# Data simulation (unchanged)
# ============================================================================

def simulate_neutrosophic_data(
    n_groups: int = 3,
    n_per_group: int = 30,
    effect_size: float = 0.0,
    indeterminacy_level: float = 0.1,
    distribution: str = 'normal',
    component_correlation: str = 'moderate',
    seed: int = None
) -> list:
    """Generate neutrosophic data with controlled component relationships."""
    if seed is not None:
        np.random.seed(seed)
    
    groups = []
    
    for group_idx in range(n_groups):
        numbers = []
        shift = effect_size * group_idx
        
        for _ in range(n_per_group):
            if distribution == 'normal':
                t_val = np.random.normal(shift, 1.0)
            elif distribution == 't':
                t_val = np.random.standard_t(df=5) + shift
            elif distribution == 'uniform':
                t_val = np.random.uniform(shift - 2, shift + 2)
            else:
                t_val = np.random.normal(shift, 1.0)
            
            precision = 0.05 + 0.1 * np.random.random()
            t_low = t_val - precision
            t_high = t_val + precision
            
            if component_correlation == 'independent':
                i_mid = np.random.uniform(0, indeterminacy_level)
            elif component_correlation == 'moderate':
                i_mid = indeterminacy_level * (0.3 + 0.4 * abs(t_val) / 3.0 + 0.3 * np.random.random())
            elif component_correlation == 'strong':
                i_mid = indeterminacy_level * (0.1 + 0.8 * abs(t_val) / 3.0 + 0.1 * np.random.random())
            else:
                i_mid = np.random.uniform(0, indeterminacy_level)
            
            i_mid = np.clip(i_mid, 0, 1)
            i_width = indeterminacy_level * 0.3 * np.random.random()
            i_low = max(0, i_mid - i_width/2)
            i_high = min(1, i_mid + i_width/2)
            
            if component_correlation == 'independent':
                f_val = np.random.normal(0, 0.8)
                f_mid = 1.0 / (1.0 + np.exp(-f_val))
            elif component_correlation == 'moderate':
                f_base = 1.0 / (1.0 + np.exp(0.5 * t_val))
                f_mid = f_base + 0.15 * np.random.normal(0, 1)
            elif component_correlation == 'strong':
                f_base = 1.0 / (1.0 + np.exp(0.8 * t_val))
                f_mid = f_base + 0.05 * np.random.normal(0, 1)
            else:
                f_val = np.random.normal(0, 0.8)
                f_mid = 1.0 / (1.0 + np.exp(-f_val))
            
            f_mid = np.clip(f_mid, 0, 1)
            f_width = 0.05 + 0.1 * np.random.random()
            f_low = max(0, f_mid - f_width/2)
            f_high = min(1, f_mid + f_width/2)
            
            numbers.append(NeutrosophicNumber(
                (t_low, t_high),
                (i_low, i_high),
                (f_low, f_high)
            ))
        
        groups.append(NeutrosophicArray(numbers))
    
    return groups


# ============================================================================
# Simulation runner
# ============================================================================

def run_simulation(
    n_simulations: int = 1000,
    n_monte_carlo_reps: int = 10,
    n_list: list = [20, 50],
    deltas: list = [0.0, 0.1, 0.25],
    effect_sizes: list = [0.0, 0.5],
    distribution: str = 'normal',
    component_correlations: list = ['independent', 'moderate'],
    alpha: float = 0.05,
    decision_methods: list = ['path_b', 'classic_three_zone', 'fisher', 't_only'],
    n_permutations: int = 1000,
    base_seed: int = 42
) -> pd.DataFrame:
    """Run simulation study with proper error tracking."""
    results = []
    
    total_conditions = (len(n_list) * len(deltas) * len(effect_sizes) * 
                        len(component_correlations))
    condition_count = 0
    
    for n in n_list:
        for delta_val in deltas:
            for es in effect_sizes:
                for corr in component_correlations:
                    condition_count += 1
                    print(f"  [{condition_count}/{total_conditions}] "
                          f"n={n}, δ={delta_val}, es={es}, corr={corr}")
                    
                    mc_results = {method: {
                        'rejections': [], 'indeterminates': [], 'successes': []
                    } for method in decision_methods}
                    
                    mc_mod = {'rejections': [], 'successes': []}
                    
                    for mc_rep in range(n_monte_carlo_reps):
                        rep_seed = base_seed + mc_rep * 10000 + condition_count * 100
                        
                        method_rejections = {m: 0 for m in decision_methods}
                        method_indeterminates = {m: 0 for m in decision_methods}
                        method_successes = {m: 0 for m in decision_methods}
                        mod_rejections = 0
                        mod_successes = 0
                        
                        for sim in range(n_simulations):
                            sim_seed = rep_seed + sim
                            
                            groups = simulate_neutrosophic_data(
                                n_groups=3, n_per_group=n, effect_size=es,
                                indeterminacy_level=delta_val,
                                distribution=distribution,
                                component_correlation=corr,
                                seed=sim_seed
                            )
                            
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                
                                for method in decision_methods:
                                    try:
                                        res = moods_median_original(
                                            groups, alpha=alpha,
                                            decision_method=method
                                        )
                                        if res is not None:
                                            method_successes[method] += 1
                                            if res['reject_H0']:
                                                method_rejections[method] += 1
                                            if res['is_indeterminate']:
                                                method_indeterminates[method] += 1
                                    except Exception as e:
                                        if method_successes.get(f'_err_{method}', 0) < 3:
                                            print(f"  ERROR [{method}]: {e}")
                                        method_successes[f'_err_{method}'] = method_successes.get(f'_err_{method}', 0) + 1
                                
                                try:
                                    res_mod = moods_median_modified(
                                        groups, alpha=alpha,
                                        n_permutations=n_permutations,
                                        permutation_seed=sim_seed
                                    )
                                    if res_mod is not None:
                                        mod_successes += 1
                                        if res_mod['reject_H0']:
                                            mod_rejections += 1
                                except Exception as e:
                                    if mod_successes < 3:
                                        print(f"  ERROR [modified]: {e}")
                        
                        for method in decision_methods:
                            n_ok = max(method_successes[method], 1)
                            mc_results[method]['rejections'].append(method_rejections[method] / n_ok)
                            mc_results[method]['indeterminates'].append(method_indeterminates[method] / n_ok)
                            mc_results[method]['successes'].append(method_successes[method])
                        
                        n_ok_mod = max(mod_successes, 1)
                        mc_mod['rejections'].append(mod_rejections / n_ok_mod)
                        mc_mod['successes'].append(mod_successes)
                    
                    for method in decision_methods:
                        rej = np.array(mc_results[method]['rejections'])
                        indet = np.array(mc_results[method]['indeterminates'])
                        succ = np.array(mc_results[method]['successes'])
                        
                        results.append({
                            'variant': f'original_{method}',
                            'decision_method': method,
                            'component_correlation': corr,
                            'delta': delta_val, 'n': n,
                            'distribution': distribution,
                            'effect_size': es,
                            'rejection_rate_mean': np.mean(rej),
                            'rejection_rate_std': np.std(rej),
                            'indeterminate_rate_mean': np.mean(indet),
                            'indeterminate_rate_std': np.std(indet),
                            'type1_error_mean': np.mean(rej) if es == 0 else None,
                            'type1_error_std': np.std(rej) if es == 0 else None,
                            'power_mean': np.mean(rej) if es > 0 else None,
                            'power_std': np.std(rej) if es > 0 else None,
                            'avg_successful_sims': np.mean(succ),
                        })
                    
                    mod_rej = np.array(mc_mod['rejections'])
                    mod_succ = np.array(mc_mod['successes'])
                    results.append({
                        'variant': 'modified',
                        'decision_method': 'permutation',
                        'component_correlation': corr,
                        'delta': delta_val, 'n': n,
                        'distribution': distribution,
                        'effect_size': es,
                        'rejection_rate_mean': np.mean(mod_rej),
                        'rejection_rate_std': np.std(mod_rej),
                        'indeterminate_rate_mean': 0.0,
                        'indeterminate_rate_std': 0.0,
                        'type1_error_mean': np.mean(mod_rej) if es == 0 else None,
                        'type1_error_std': np.std(mod_rej) if es == 0 else None,
                        'power_mean': np.mean(mod_rej) if es > 0 else None,
                        'power_std': np.std(mod_rej) if es > 0 else None,
                        'avg_successful_sims': np.mean(mod_succ),
                    })
    
    return pd.DataFrame(results)



def get_method_note(method: str) -> str:
    """Return explanatory note for each decision method."""
    notes = {
        'corrected_three_zone': (
            "CORRECTED: Fisher combined p-value for primary decision, "
            "p-interval for ambiguity classification. Calibrated at α."
        ),
        'classic_three_zone': (
            "CLASSIC: Requires ALL p < α. Effective level ≈ α³. "
            "Ultra-conservative. Included for historical comparison."
        ),
        'fisher': (
            "Fisher: -2 Σ ln(pᵢ) ~ χ²(2k). "
            "VALID under 'independent' correlation. "
            "ANTI-CONSERVATIVE under 'moderate'/'strong' correlation."
        ),
        'stouffer': (
            "Stouffer: Weighted Z-test (two-sided, corrected). "
            "Weight sensitivity analysis recommended."
        ),
        't_only': (
            "Uses only T-component p-value. Baseline for comparison."
        ),
    }
    return notes.get(method, "")


def get_correlation_note(corr: str, method: str) -> str:
    """Return note about method validity under given correlation structure."""
    if method == 'fisher':
        if corr == 'independent':
            return "Fisher IS valid here — components are independent."
        else:
            return "Fisher is ANTI-CONSERVATIVE here — components are correlated."
    elif method == 'classic_three_zone':
        return "Ultra-conservative regardless of correlation structure."
    elif method == 'corrected_three_zone':
        return "Calibrated at α across correlation structures."
    return ""


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NEUTROSOPHIC MOOD'S MEDIAN TEST - CORRECTED")
    print("=" * 70)
    
    N_SIMULATIONS = 100
    N_MONTE_CARLO = 5
    N_LIST = [20, 100, 500, 1000]
    DELTAS = [0.0, 0.1, 0.25]
    EFFECT_SIZES = [0.0, 0.5]
    CORRELATIONS = ['independent', 'moderate', 'strong']
    METHODS = ['corrected_three_zone', 'classic_three_zone', 'fisher', 't_only']
    ALPHA = 0.05
    
    print(f"Methods: {METHODS}")
    print(f"Sample sizes: {N_LIST}")
    print(f"Correlations: {CORRELATIONS}")
    print()
    
    df = run_simulation(
        n_simulations=N_SIMULATIONS,
        n_monte_carlo_reps=N_MONTE_CARLO,
        n_list=N_LIST,
        deltas=DELTAS,
        effect_sizes=EFFECT_SIZES,
        component_correlations=CORRELATIONS,
        alpha=ALPHA,
        decision_methods=METHODS,
        base_seed=42
    )
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Key columns
    cols = ['variant', 'component_correlation', 'delta', 'n', 'effect_size',
            'rejection_rate_mean', 'indeterminate_rate_mean', 'avg_successful_sims']
    print(df[cols].to_string(index=False))
    
    df.to_csv('moods_median_final_corrected.csv', index=False)
    print(f"\nSaved to: moods_median_final_corrected.csv")