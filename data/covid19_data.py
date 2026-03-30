import numpy as np
import pandas as pd

def generate_covid19_data(n_regions=4, n_per_region=50, indeterminacy_rate=0.20, random_seed=42):
    rng = np.random.default_rng(seed=random_seed)
    
    regions = ['Northern', 'Southern', 'Eastern', 'Western Ghana'][:n_regions]
    data = []
    
    for i, region in enumerate(regions):
        # Base severity increases by effect size ~0.4 per region for demonstration
        base_severity = rng.normal(loc=5.0 + i * 1.5, scale=2.0, size=n_per_region)
        base_recovery = rng.normal(loc=14.0 + i * 1.0, scale=3.0, size=n_per_region)
        
        for j in range(n_per_region):
            sev = max(1.0, base_severity[j])
            rec = max(7.0, base_recovery[j])
            
            is_indet = rng.random() < indeterminacy_rate
            
            row = {'region': region, 'patient_id': f"{region[:3]}-{j+1:03d}"}
            
            # Severity
            row['symptom_severity_T_lower'] = sev
            row['symptom_severity_T_upper'] = sev
            row['symptom_severity_I_lower'] = 0.0
            row['symptom_severity_I_upper'] = 0.0 if not is_indet else rng.uniform(0.5, 2.0)
            row['symptom_severity_F_lower'] = 0.0
            row['symptom_severity_F_upper'] = 0.0
            
            # Recovery
            missing_rec = rng.random() < 0.10
            row['recovery_days_T_lower'] = rec if not missing_rec else 0.0
            row['recovery_days_T_upper'] = rec if not missing_rec else 0.0
            row['recovery_days_I_lower'] = 0.0
            row['recovery_days_I_upper'] = rng.uniform(5.0, 10.0) if missing_rec else (rng.uniform(1.0, 3.0) if is_indet else 0.0)
            row['recovery_days_F_lower'] = 0.0
            row['recovery_days_F_upper'] = 0.0
            
            row['test_result_T_lower'] = 1.0
            row['test_result_T_upper'] = 1.0
            row['test_result_I_lower'] = 0.0
            row['test_result_I_upper'] = 1.0 if is_indet else 0.0
            row['test_result_F_lower'] = 0.0
            row['test_result_F_upper'] = 0.0
            
            row['is_indeterminate'] = is_indet or missing_rec
            
            data.append(row)
            
    df = pd.DataFrame(data)
    metadata = {
        'description': 'Synthetic COVID-19 dataset matching Sherwani et al. (2021) patterns.',
        'indeterminacy_rate': indeterminacy_rate,
        'variables': ['symptom_severity', 'recovery_days', 'test_result']
    }
    return df, metadata
