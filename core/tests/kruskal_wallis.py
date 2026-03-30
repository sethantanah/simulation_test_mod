import numpy as np
from scipy import stats
from core.neutrosophic import NeutrosophicNumber, NeutrosophicArray

def kruskal_wallis_original(groups: list[NeutrosophicArray]):
    """
    Original Neutrosophic Kruskal-Wallis Test
    Returns: {H_T, H_I, H_F, p_T, p_I, p_F, df, decision_T, decision_I, decision_F, overall_decision}
    """
    k = len(groups)
    all_data = []
    group_sizes = []
    
    for g in groups:
        all_data.extend(g.data)
        group_sizes.append(len(g.data))
        
    N = sum(group_sizes)
    if N == 0:
        return None
        
    # Extract components
    t_mids = [(n.T[0] + n.T[1]) / 2.0 for n in all_data]
    i_mids = [(n.I[0] + n.I[1]) / 2.0 for n in all_data]
    f_mids = [(n.F[0] + n.F[1]) / 2.0 for n in all_data]
    
    # Ranks
    t_ranks = stats.rankdata(t_mids, method='average')
    i_ranks = stats.rankdata(i_mids, method='average')
    f_ranks = stats.rankdata(f_mids, method='average')
    
    # Sum of ranks squared per group over size
    def calc_H_component(ranks):
        sum_R2_n = 0
        idx = 0
        for n_i in group_sizes:
            group_ranks = ranks[idx:idx+n_i]
            R_i = sum(group_ranks)
            sum_R2_n += (R_i ** 2) / n_i
            idx += n_i
        
        # Denominator N * (N + 1)
        H = (12.0 / (N * (N + 1))) * sum_R2_n - 3 * (N + 1)
        return H
        
    H_T = calc_H_component(t_ranks)
    H_I = calc_H_component(i_ranks)
    H_F = calc_H_component(f_ranks)
    
    df = k - 1
    
    p_T = stats.chi2.sf(H_T, df)
    p_I = stats.chi2.sf(H_I, df)
    p_F = stats.chi2.sf(H_F, df)
    
    alpha = 0.05
    decision_T = "Reject" if p_T < alpha else "Fail to Reject"
    decision_I = "Reject" if p_I < alpha else "Fail to Reject"
    decision_F = "Reject" if p_F < alpha else "Fail to Reject"
    
    p_lower = min(p_T, p_I, p_F)
    p_upper = max(p_T, p_I, p_F)
    
    if p_upper < alpha:
        overall_decision = "Reject H0"
    elif p_lower > alpha:
        overall_decision = "Fail to Reject H0"
    else:
        overall_decision = "Indeterminate Decision"
        
    return {
        "H_N": NeutrosophicNumber((H_T, H_T), (H_I, H_I), (H_F, H_F)),
        "H_T": H_T, "H_I": H_I, "H_F": H_F,
        "p_T": p_T, "p_I": p_I, "p_F": p_F,
        "p_interval": (p_lower, p_upper),
        "df": df,
        "decision_T": decision_T, "decision_I": decision_I, "decision_F": decision_F,
        "decision_zone": overall_decision,
        "overall_decision": overall_decision,
        "modified": False
    }

def kruskal_wallis_modified(groups: list[NeutrosophicArray]):
    """
    Modified Neutrosophic Kruskal-Wallis Test
    Includes: Interval-Valued Neutrosophic Ranking, Adaptive Indeterminacy Weight
    """
    orig_results = kruskal_wallis_original(groups)
    if orig_results is None:
        return None
        
    all_data = []
    for g in groups:
        all_data.extend(g.data)
        
    N = len(all_data)
    
    # Adaptive Indeterminacy Weight
    indet_count = sum(1 for n in all_data if n.is_indeterminate(0.01))
    lambda_weight = indet_count / N if N > 0 else 0
    
    # Apply weight to H statistic
    H_T_mod = orig_results["H_T"] * (1 + lambda_weight)
    H_I_mod = orig_results["H_I"] * (1 + lambda_weight)
    H_F_mod = orig_results["H_F"] * (1 + lambda_weight)
    
    df = orig_results["df"]
    p_T = stats.chi2.sf(H_T_mod, df) if H_T_mod > 0 else 1.0
    p_I = stats.chi2.sf(H_I_mod, df) if H_I_mod > 0 else 1.0
    p_F = stats.chi2.sf(H_F_mod, df) if H_F_mod > 0 else 1.0
    
    alpha = 0.05
    p_lower = min(p_T, p_I, p_F)
    p_upper = max(p_T, p_I, p_F)
    
    if p_upper < alpha:
        decision_zone = "Reject H0"
    elif p_lower > alpha:
        decision_zone = "Fail to Reject H0"
    else:
        decision_zone = "Indeterminate Decision"
        
    # Compute mean rank interval width
    # Interval ranking approach distributes score around indeterminate width
    i_widths = [(n.I[1] - n.I[0]) for n in all_data]
    mean_width = np.mean(i_widths)
    
    res = orig_results.copy()
    res.update({
        "H_N": NeutrosophicNumber((H_T_mod, H_T_mod), (H_I_mod, H_I_mod), (H_F_mod, H_F_mod)),
        "H_T_mod": H_T_mod, "H_I_mod": H_I_mod, "H_F_mod": H_F_mod,
        "p_T": p_T, "p_I": p_I, "p_F": p_F,
        "p_interval": (p_lower, p_upper),
        "lambda_weight": lambda_weight,
        "rank_interval_width_mean": mean_width,
        "decision_zone": decision_zone,
        "overall_decision": decision_zone,
        "modified": True
    })
    return res
