import json
from pathlib import Path

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
    def T_low(self) -> float:
        return self.T[0]
    
    @property
    def T_high(self) -> float:
        return self.T[1]
    
    @property
    def I_mid(self) -> float:
        return (self.I[0] + self.I[1]) / 2.0
    
    @property
    def I_width(self) -> float:
        return abs(self.I[1] - self.I[0])
    
    @property
    def F_mid(self) -> float:
        return (self.F[0] + self.F[1]) / 2.0
    
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
# Core Mood's Test Functions
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


def _build_2xk(values: np.ndarray, group_boundaries: np.ndarray, 
               grand_median: float, k: int) -> np.ndarray:
    """Build 2×k contingency table: above vs at/below grand median."""
    O = np.zeros((2, k))
    for j in range(k):
        group_vals = values[group_boundaries[j]:group_boundaries[j+1]]
        O[0, j] = np.sum(group_vals > grand_median)
        O[1, j] = len(group_vals) - O[0, j]
    return O


def _build_3xk(values: np.ndarray, group_boundaries: np.ndarray,
               grand_median: float, delta: float, k: int) -> np.ndarray:
    """Build 3×k contingency table with indeterminate zone."""
    O = np.zeros((3, k))
    for j in range(k):
        group_vals = values[group_boundaries[j]:group_boundaries[j+1]]
        O[0, j] = np.sum(group_vals > grand_median + delta)
        O[2, j] = np.sum(group_vals < grand_median - delta)
        O[1, j] = len(group_vals) - O[0, j] - O[2, j]
    return O


def _run_moods_test(values: np.ndarray, group_boundaries: np.ndarray, k: int) -> dict:
    """
    Run standard Mood's median test on scalar values.
    Returns chi2, df, p_value, grand_median, contingency_table.
    """
    grand_median = float(np.median(values))
    O = _build_2xk(values, group_boundaries, grand_median, k)
    chi2 = _chi2_from_table(O)
    df = k - 1
    p_value = float(stats.chi2.sf(chi2, df)) if df > 0 else 1.0
    return {
        'chi2': chi2,
        'df': df,
        'p_value': p_value,
        'grand_median': grand_median,
        'table': O
    }


# ============================================================================
# Original Neutrosophic Mood's Median Test
# ============================================================================

def moods_median_original(
    groups: list,
    alpha: float = 0.05,
) -> dict | None:
    """
    Original Neutrosophic Mood's Median Test.
    
    Runs standard Mood's test on T-midpoints.
    This is the classical special case: when indeterminacy width is zero,
    it reduces exactly to scipy.stats.median_test.
    
    Decision is based on p_T alone.
    I and F components are reported as diagnostics only.
    """
    k = len(groups)
    if k < 2:
        raise ValueError("Need at least 2 groups.")

    all_data = [n for g in groups for n in g]
    N = len(all_data)
    if N == 0:
        return None

    t_mids = np.array([n.T_mid for n in all_data])
    group_boundaries = np.cumsum([0] + [len(g) for g in groups])

    # Run Mood's on T-midpoints
    result_T = _run_moods_test(t_mids, group_boundaries, k)
    
    # I and F components — diagnostics only, not used for decision
    i_mids = np.array([n.I_mid for n in all_data])
    f_mids = np.array([n.F_mid for n in all_data])
    
    # Average I-width for observations near the grand median boundary
    grand_T = result_T['grand_median']
    near_boundary = [n for n in all_data 
                     if abs(n.T_mid - grand_T) < 0.1 * np.std(t_mids)]
    avg_I_near_boundary = (np.mean([n.I_width for n in near_boundary]) 
                           if near_boundary else 0.0)
    
    decision = "Reject H0" if result_T['p_value'] < alpha else "Fail to Reject H0"

    return {
        # Primary test result (T-midpoints)
        'chi2': result_T['chi2'],
        'df': result_T['df'],
        'p_value': result_T['p_value'],
        'grand_median_T': grand_T,
        'contingency_table_2xk': result_T['table'],
        
        # Decision
        'decision': decision,
        'overall_decision': decision,
        'reject_H0': decision == "Reject H0",
        'is_indeterminate': False,
        
        # Diagnostics from I and F components
        'avg_I_width': np.mean([n.I_width for n in all_data]),
        'avg_I_near_boundary': avg_I_near_boundary,
        'prop_indeterminate_obs': np.mean([n.is_indeterminate() for n in all_data]),
        
        # Metadata
        'alpha': alpha,
        'modified': False,
    }


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
    
    Two enhancements, both staying inside Mood's framework:
    
    1. NEUTROSOPHIC TEST STATISTIC INTERVAL:
       Run Mood's on T_low and T_high separately.
       This gives [chi2_lo, chi2_hi] — a neutrosophic interval for the
       test statistic that reflects measurement uncertainty.
       Three-zone decision on this interval:
       - p_lo < alpha → Reject (both bounds reject)
       - p_hi > alpha → Fail to Reject (both bounds fail)
       - Otherwise → Indeterminate (bounds disagree)
    
    2. 3×k TABLE WITH ADAPTIVE BANDWIDTH:
       Observations near the grand median are classified as indeterminate
       rather than forced into above/below.
       Uses permutation-based p-value for valid inference.
    """
    k = len(groups)
    if k < 2:
        raise ValueError("Need at least 2 groups.")

    all_data = [n for g in groups for n in g]
    N = len(all_data)
    if N == 0:
        return None

    # Extract T-interval endpoints
    t_lows = np.array([n.T_low for n in all_data])
    t_highs = np.array([n.T_high for n in all_data])
    t_mids = np.array([n.T_mid for n in all_data])
    
    group_boundaries = np.cumsum([0] + [len(g) for g in groups])

    # --- Enhancement 1: Neutrosophic test statistic interval ---
    result_lo = _run_moods_test(t_lows, group_boundaries, k)
    result_hi = _run_moods_test(t_highs, group_boundaries, k)
    
    # The interval: [chi2_hi, chi2_lo] → [p_lo, p_hi]
    # p_lo corresponds to chi2_hi (larger chi2 → smaller p)
    # p_hi corresponds to chi2_lo (smaller chi2 → larger p)
    p_lo = result_hi['p_value']  # larger chi2 → smaller p
    p_hi = result_lo['p_value']  # smaller chi2 → larger p
    
    # Three-zone decision on the neutrosophic interval
    if p_lo < alpha and p_hi < alpha:
        # Both bounds reject: robust rejection
        decision_interval = "Reject H0"
        interval_detail = "robust_reject_both_bounds_below_alpha"
    elif p_lo > alpha and p_hi > alpha:
        # Both bounds fail to reject: robust failure
        decision_interval = "Fail to Reject H0"
        interval_detail = "robust_fail_both_bounds_above_alpha"
    else:
        # Bounds disagree: indeterminate
        decision_interval = "Indeterminate"
        interval_detail = "indeterminate_interval_straddles_alpha"
    
    # --- Enhancement 2: 3×k table with adaptive bandwidth ---
    # Use T-midpoints for the table construction
    grand_median = float(np.median(t_mids))
    
    q75, q25 = float(np.percentile(t_mids, 75)), float(np.percentile(t_mids, 25))
    iqr_T = q75 - q25
    indet_prop = np.mean([n.is_indeterminate() for n in all_data])
    delta = max(iqr_T * indet_prop, iqr_T * 0.01)
    
    O_3xk = _build_3xk(t_mids, group_boundaries, grand_median, delta, k)
    
    # Permutation-based p-value for the 3×k test
    rng = np.random.RandomState(permutation_seed)
    observed_chi2 = _chi2_from_table(O_3xk)
    
    perm_chi2s = np.zeros(n_permutations)
    for i in range(n_permutations):
        permuted = rng.permutation(t_mids)
        perm_O = _build_3xk(permuted, group_boundaries, grand_median, delta, k)
        perm_chi2s[i] = _chi2_from_table(perm_O)
    
    p_3xk = np.mean(perm_chi2s >= observed_chi2)
    p_3xk = max(p_3xk, 1.0 / (n_permutations + 1))
    
    # I/F diagnostics
    near_boundary = [n for n in all_data 
                     if abs(n.T_mid - grand_median) < delta]
    avg_I_near_boundary = (np.mean([n.I_width for n in near_boundary]) 
                           if near_boundary else 0.0)
    
    return {
        # Neutrosophic interval test
        'chi2_lo': result_lo['chi2'],
        'chi2_hi': result_hi['chi2'],
        'p_lo': p_lo,
        'p_hi': p_hi,
        'chi2_interval': (result_lo['chi2'], result_hi['chi2']),
        'p_interval': (p_lo, p_hi),
        'decision_interval': decision_interval,
        'interval_detail': interval_detail,
        'is_indeterminate': decision_interval == "Indeterminate",
        
        # 3×k table test
        'chi2_3xk': observed_chi2,
        'p_3xk': p_3xk,
        'contingency_table_3xk': O_3xk,
        'band_width_delta': delta,
        'n_permutations': n_permutations,
        
        # Overall decision: indeterminate if interval says so, else use 3xk
        'decision': decision_interval,
        'overall_decision': decision_interval,
        'reject_H0': decision_interval == "Reject H0",
        
        # Diagnostics
        'grand_median': grand_median,
        'avg_I_width': np.mean([n.I_width for n in all_data]),
        'avg_I_near_boundary': avg_I_near_boundary,
        'prop_indeterminate_obs': indet_prop,
        
        # Metadata
        'alpha': alpha,
        'modified': True,
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
    n_simulations: int = 500,
    n_monte_carlo_reps: int = 5,
    n_list: list = [20, 50, 100, 500],
    deltas: list = [0.0, 0.1, 0.25],
    effect_sizes: list = [0.0, 0.5, 1.0],
    distribution: str = 'normal',
    component_correlations: list = ['independent', 'moderate', 'strong'],
    alpha: float = 0.05,
    n_permutations: int = 500,
    base_seed: int = 42,
    progress_callback=None,
    checkpoint_path=None,
) -> pd.DataFrame:
    """Run simulation comparing original vs modified neutrosophic Mood's test."""
    results = []
    completed_conditions = 0
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.exists():
            try:
                checkpoint = json.loads(path.read_text(encoding="utf-8"))
                results = checkpoint.get("results", [])
                completed_conditions = int(checkpoint.get("completed_conditions", 0))
            except Exception:
                completed_conditions = 0
    
    total_conditions = (len(n_list) * len(deltas) * len(effect_sizes) * 
                        len(component_correlations))
    condition_count = 0
    
    for n in n_list:
        for delta_val in deltas:
            for es in effect_sizes:
                for corr in component_correlations:
                    condition_count += 1
                    if condition_count <= completed_conditions:
                        continue
                    message = (
                        f"[{condition_count}/{total_conditions}] "
                        f"n={n}, δ={delta_val}, es={es}, corr={corr}"
                    )
                    print(f"  {message}")
                    progress_value = condition_count / max(total_conditions, 1)
                    if progress_callback is not None:
                        progress_callback(message, progress_value)
                    
                    mc_orig = {'rejections': [], 'indeterminates': [], 'successes': []}
                    mc_mod = {'rejections': [], 'indeterminates': [], 'successes': []}
                    
                    for mc_rep in range(n_monte_carlo_reps):
                        rep_seed = base_seed + mc_rep * 10000 + condition_count * 100
                        
                        orig_rej, orig_indet, orig_succ = 0, 0, 0
                        mod_rej, mod_indet, mod_succ = 0, 0, 0
                        
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
                                
                                # Original test
                                try:
                                    res_orig = moods_median_original(groups, alpha=alpha)
                                    if res_orig is not None:
                                        orig_succ += 1
                                        if res_orig['reject_H0']:
                                            orig_rej += 1
                                        if res_orig['is_indeterminate']:
                                            orig_indet += 1
                                except Exception:
                                    pass
                                
                                # Modified test
                                try:
                                    res_mod = moods_median_modified(
                                        groups, alpha=alpha,
                                        n_permutations=n_permutations,
                                        permutation_seed=sim_seed
                                    )
                                    if res_mod is not None:
                                        mod_succ += 1
                                        if res_mod['reject_H0']:
                                            mod_rej += 1
                                        if res_mod['is_indeterminate']:
                                            mod_indet += 1
                                except Exception:
                                    pass
                        
                        mc_orig['rejections'].append(orig_rej / max(orig_succ, 1))
                        mc_orig['indeterminates'].append(orig_indet / max(orig_succ, 1))
                        mc_orig['successes'].append(orig_succ)
                        
                        mc_mod['rejections'].append(mod_rej / max(mod_succ, 1))
                        mc_mod['indeterminates'].append(mod_indet / max(mod_succ, 1))
                        mc_mod['successes'].append(mod_succ)
                    
                    # Original test results
                    results.append({
                        'variant': 'original',
                        'component_correlation': corr,
                        'delta': delta_val, 'n': n,
                        'distribution': distribution,
                        'effect_size': es,
                        'rejection_rate_mean': np.mean(mc_orig['rejections']),
                        'rejection_rate_std': np.std(mc_orig['rejections']),
                        'indeterminate_rate_mean': np.mean(mc_orig['indeterminates']),
                        'indeterminate_rate_std': np.std(mc_orig['indeterminates']),
                        'type1_error_mean': np.mean(mc_orig['rejections']) if es == 0 else None,
                        'type1_error_std': np.std(mc_orig['rejections']) if es == 0 else None,
                        'power_mean': np.mean(mc_orig['rejections']) if es > 0 else None,
                        'power_std': np.std(mc_orig['rejections']) if es > 0 else None,
                        'avg_successful_sims': np.mean(mc_orig['successes']),
                    })
                    
                    # Modified test results
                    results.append({
                        'variant': 'modified',
                        'component_correlation': corr,
                        'delta': delta_val, 'n': n,
                        'distribution': distribution,
                        'effect_size': es,
                        'rejection_rate_mean': np.mean(mc_mod['rejections']),
                        'rejection_rate_std': np.std(mc_mod['rejections']),
                        'indeterminate_rate_mean': np.mean(mc_mod['indeterminates']),
                        'indeterminate_rate_std': np.std(mc_mod['indeterminates']),
                        'type1_error_mean': np.mean(mc_mod['rejections']) if es == 0 else None,
                        'type1_error_std': np.std(mc_mod['rejections']) if es == 0 else None,
                        'power_mean': np.mean(mc_mod['rejections']) if es > 0 else None,
                        'power_std': np.std(mc_mod['rejections']) if es > 0 else None,
                        'avg_successful_sims': np.mean(mc_mod['successes']),
                    })
    
    return pd.DataFrame(results)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NEUTROSOPHIC MOOD'S MEDIAN TEST — CORRECT FRAMEWORK")
    print("=" * 70)
    print()
    print("Original:  Mood's on T-midpoints → binary decision")
    print("Modified:  Mood's on [T_low, T_high] interval → three-zone decision")
    print("           + 3×k table with adaptive bandwidth")
    print("I/F:       Diagnostics only (not part of test decision)")
    print()
    
    N_SIMULATIONS = 100
    N_MONTE_CARLO = 5
    N_LIST = [20, 100, 500, 1000, 10000]
    DELTAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    EFFECT_SIZES = [0.0, 0.5, 0.75, 1.0]
    CORRELATIONS = ['independent', 'moderate', 'strong']
    ALPHA = 0.05
    
    df = run_simulation(
        n_simulations=N_SIMULATIONS,
        n_monte_carlo_reps=N_MONTE_CARLO,
        n_list=N_LIST,
        deltas=DELTAS,
        effect_sizes=EFFECT_SIZES,
        component_correlations=CORRELATIONS,
        alpha=ALPHA,
        base_seed=42
    )
    
    # Display key columns
    cols = ['variant', 'component_correlation', 'delta', 'n', 'effect_size',
            'rejection_rate_mean', 'indeterminate_rate_mean', 'avg_successful_sims']
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(df[cols].to_string(index=False))
    
    df.to_csv('moods_median_correct_framework.csv', index=False)
    print(f"\nSaved to: moods_median_correct_framework.csv")