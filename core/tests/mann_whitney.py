import numpy as np
from scipy import stats
from core.neutrosophic import NeutrosophicNumber, NeutrosophicArray

def mann_whitney_original(group1: NeutrosophicArray, group2: NeutrosophicArray):
    n1 = len(group1.data)
    n2 = len(group2.data)
    if n1 == 0 or n2 == 0: return None
    
    all_data = group1.data + group2.data
    
    t_mids = [(n.T[0] + n.T[1]) / 2.0 for n in all_data]
    i_mids = [(n.I[0] + n.I[1]) / 2.0 for n in all_data]
    f_mids = [(n.F[0] + n.F[1]) / 2.0 for n in all_data]
    
    t_ranks = stats.rankdata(t_mids, method='average')
    i_ranks = stats.rankdata(i_mids, method='average')
    f_ranks = stats.rankdata(f_mids, method='average')
    
    R_T1 = sum(t_ranks[:n1])
    R_I1 = sum(i_ranks[:n1])
    R_F1 = sum(f_ranks[:n1])
    
    R_T2 = sum(t_ranks[n1:])
    R_I2 = sum(i_ranks[n1:])
    R_F2 = sum(f_ranks[n1:])
    
    U_T1 = n1 * n2 + n1*(n1+1)/2 - R_T1
    U_I1 = n1 * n2 + n1*(n1+1)/2 - R_I1
    U_F1 = n1 * n2 + n1*(n1+1)/2 - R_F1
    
    U_T2 = n1 * n2 + n2*(n2+1)/2 - R_T2
    U_I2 = n1 * n2 + n2*(n2+1)/2 - R_I2
    U_F2 = n1 * n2 + n2*(n2+1)/2 - R_F2
    
    U_T = min(U_T1, U_T2)
    U_I = min(U_I1, U_I2)
    U_F = min(U_F1, U_F2)
    
    U_N = NeutrosophicNumber((U_T, U_T), (U_I, U_I), (U_F, U_F))
    
    # Z approximation for large samples (usually valid for n > 20)
    mu_U = n1 * n2 / 2.0
    sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    
    Z_T = (U_T - mu_U) / sigma_U if sigma_U > 0 else 0
    Z_I = (U_I - mu_U) / sigma_U if sigma_U > 0 else 0
    Z_F = (U_F - mu_U) / sigma_U if sigma_U > 0 else 0
    
    # two-tailed test
    p_T = 2 * stats.norm.sf(abs(Z_T))
    p_I = 2 * stats.norm.sf(abs(Z_I))
    p_F = 2 * stats.norm.sf(abs(Z_F))
    
    p_lower = min(p_T, p_I, p_F)
    p_upper = max(p_T, p_I, p_F)
    
    alpha = 0.05
    if p_upper < alpha: decision_zone = "Reject H0"
    elif p_lower > alpha: decision_zone = "Fail to Reject H0"
    else: decision_zone = "Indeterminate Decision"
    
    return {
        "U_T1": U_T1, "U_T2": U_T2,
        "U_T": U_T, "U_I": U_I, "U_F": U_F,
        "U_original": U_N,
        "p_T": p_T, "p_I": p_I, "p_F": p_F,
        "p_interval": (p_lower, p_upper),
        "Z_N": (Z_T, Z_I, Z_F),
        "decision_zone": decision_zone,
        "modified": False
    }

def mann_whitney_modified(group1: NeutrosophicArray, group2: NeutrosophicArray):
    orig = mann_whitney_original(group1, group2)
    if not orig: return None
    
    n1 = len(group1.data)
    n2 = len(group2.data)
    
    # 1. Dominance probability
    X_vals = [n.defuzzify() for n in group1.data]
    Y_vals = [n.defuzzify() for n in group2.data]
    X_indet = [n.is_indeterminate(0.01) for n in group1.data]
    Y_indet = [n.is_indeterminate(0.01) for n in group2.data]
    
    count_T = 0
    count_I = 0
    count_F = 0
    
    for i in range(n1):
        for j in range(n2):
            if X_indet[i] and Y_indet[j]:
                count_I += 1 # both indeterminate
            elif X_vals[i] > Y_vals[j]:
                count_T += 1 # X dominates Y
            elif X_vals[i] < Y_vals[j]:
                count_F += 1 # Y dominates X
            else:
                count_I += 1 # tie
                
    total_pairs = n1 * n2
    P_T = count_T / total_pairs
    P_I = count_I / total_pairs
    P_F = count_F / total_pairs
    
    # 2. Aggregated U via weights
    w_T = P_T
    w_I = P_I
    w_F = P_F
    
    # Normalize weights
    if w_T + w_I + w_F > 0:
        sum_w = w_T + w_I + w_F
        w_T /= sum_w; w_I /= sum_w; w_F /= sum_w
    else:
        w_T = w_I = w_F = 1/3.0
        
    U_T = orig["U_T"]
    U_I = orig["U_I"]
    U_F = orig["U_F"]
    
    U_modified = w_T * U_T + w_I * U_I + w_F * U_F
    
    # U_modified p-value
    mu_U = n1 * n2 / 2.0
    sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    Z_mod = (U_modified - mu_U) / sigma_U if sigma_U > 0 else 0
    p_mod = 2 * stats.norm.sf(abs(Z_mod))
    
    p_lower = min(orig["p_interval"][0], p_mod)
    p_upper = max(orig["p_interval"][1], p_mod)
    
    alpha = 0.05
    if p_upper < alpha: decision_zone = "Reject H0"
    elif p_lower > alpha: decision_zone = "Fail to Reject H0"
    else: decision_zone = "Indeterminate Decision"
    
    effect_size = Z_mod / np.sqrt(n1 + n2) if (n1+n2) > 0 else 0
    
    res = orig.copy()
    res.update({
        "U_modified": U_modified,
        "dominance_prob_T": P_T,
        "dominance_prob_I": P_I,
        "dominance_prob_F": P_F,
        "w_T": w_T, "w_I": w_I, "w_F": w_F,
        "p_mod": p_mod,
        "p_interval": (p_lower, p_upper),
        "effect_size_neutrosophic": effect_size,
        "decision_zone": decision_zone,
        "modified": True
    })
    return res
