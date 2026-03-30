import numpy as np
from scipy import stats
from core.neutrosophic import NeutrosophicNumber, NeutrosophicArray

def moods_median_original(groups: list[NeutrosophicArray]):
    k = len(groups)
    all_data = []
    for g in groups: all_data.extend(g.data)
    
    N = len(all_data)
    if N == 0: return None
    
    t_mids = [(n.T[0] + n.T[1]) / 2.0 for n in all_data]
    i_mids = [(n.I[0] + n.I[1]) / 2.0 for n in all_data]
    f_mids = [(n.F[0] + n.F[1]) / 2.0 for n in all_data]
    
    grand_median_T = np.median(t_mids)
    grand_median_I = np.median(i_mids)
    grand_median_F = np.median(f_mids)
    
    def calc_chi2(mids_for_groups, grand_median):
        O = np.zeros((2, k))
        group_sizes = []
        
        for j, g_mids in enumerate(mids_for_groups):
            n_j = len(g_mids)
            group_sizes.append(n_j)
            above = sum(1 for x in g_mids if x > grand_median)
            O[0, j] = above
            O[1, j] = n_j - above
            
        row_totals = O.sum(axis=1)
        col_totals = O.sum(axis=0)
        total = O.sum()
        
        chi2 = 0.0
        for i in range(2):
            for j in range(k):
                E = (row_totals[i] * col_totals[j]) / total if total > 0 else 0
                if E > 0:
                    chi2 += ((O[i,j] - E)**2) / E
        return chi2, O
        
    t_groups = [[(n.T[0] + n.T[1]) / 2.0 for n in g.data] for g in groups]
    i_groups = [[(n.I[0] + n.I[1]) / 2.0 for n in g.data] for g in groups]
    f_groups = [[(n.F[0] + n.F[1]) / 2.0 for n in g.data] for g in groups]
    
    chi2_T, O_T = calc_chi2(t_groups, grand_median_T)
    chi2_I, O_I = calc_chi2(i_groups, grand_median_I)
    chi2_F, O_F = calc_chi2(f_groups, grand_median_F)
    
    df = k - 1
    p_T = stats.chi2.sf(chi2_T, df) if df > 0 else 1.0
    p_I = stats.chi2.sf(chi2_I, df) if df > 0 else 1.0
    p_F = stats.chi2.sf(chi2_F, df) if df > 0 else 1.0
    
    p_lower = min(p_T, p_I, p_F)
    p_upper = max(p_T, p_I, p_F)
    
    alpha = 0.05
    if p_upper < alpha: decision = "Reject H0"
    elif p_lower > alpha: decision = "Fail to Reject H0"
    else: decision = "Indeterminate Decision"
    
    return {
        "chi2_original": NeutrosophicNumber((chi2_T, chi2_T), (chi2_I, chi2_I), (chi2_F, chi2_F)),
        "chi2_T": chi2_T, "chi2_I": chi2_I, "chi2_F": chi2_F,
        "contingency_table_2xk": O_T, # Using T for simplicity
        "p_interval_original": (p_lower, p_upper),
        "df": df,
        "grand_median": NeutrosophicNumber((grand_median_T, grand_median_T), 
                                          (grand_median_I, grand_median_I), 
                                          (grand_median_F, grand_median_F)),
        "decision_zone": decision,
        "modified": False
    }

def moods_median_modified(groups: list[NeutrosophicArray]):
    orig = moods_median_original(groups)
    if not orig: return None
    
    k = len(groups)
    all_data = []
    for g in groups: all_data.extend(g.data)
    
    crisp_vals = [n.defuzzify() for n in all_data]
    grand_median = np.median(crisp_vals)
    
    # Calculate IQR and indeterminacy delta
    q75, q25 = np.percentile(crisp_vals, [75, 25])
    iqr = q75 - q25
    indet_prop = sum(1 for n in all_data if n.is_indeterminate(0.01)) / len(all_data) if len(all_data)>0 else 0
    
    delta = iqr * indet_prop
    
    # 3xK contingency table
    O = np.zeros((3, k))
    
    for j, g in enumerate(groups):
        for n in g.data:
            val = n.defuzzify()
            if val > grand_median + delta:
                O[0, j] += 1 # Above
            elif val < grand_median - delta:
                O[2, j] += 1 # Below
            else:
                O[1, j] += 1 # Indeterminate
                
    row_totals = O.sum(axis=1)
    col_totals = O.sum(axis=0)
    total = O.sum()
    
    chi2_mod = 0.0
    for i in range(3):
        for j in range(k):
            E = (row_totals[i] * col_totals[j]) / total if total > 0 else 0
            if E > 0:
                chi2_mod += ((O[i,j] - E)**2) / E
                
    df_mod = 2 * (k - 1)
    p_mod = stats.chi2.sf(chi2_mod, df_mod) if df_mod > 0 else 1.0
    
    alpha = 0.05
    orig_p_lower = orig["p_interval_original"][0]
    orig_p_upper = orig["p_interval_original"][1]
    
    p_lower = min(orig_p_lower, p_mod)
    p_upper = max(orig_p_upper, p_mod)
    
    if p_upper < alpha: decision = "Reject H0"
    elif p_lower > alpha: decision = "Fail to Reject H0"
    else: decision = "Indeterminate Decision"
    
    res = orig.copy()
    res.update({
        "chi2_modified": chi2_mod,
        "contingency_table_3xk": O,
        "band_width_delta": delta,
        "p_interval_modified": (p_lower, p_upper),
        "decision_zone": decision,
        "df_mod": df_mod,
        "p_mod": p_mod,
        "modified": True
    })
    return res
