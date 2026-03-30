import numpy as np
import pandas as pd

def generate_resettlement_data(n_households=120, indeterminacy_rate=0.25, random_seed=42):
    rng = np.random.default_rng(seed=random_seed)
    
    zones = ['Zone A (Core)', 'Zone B (Buffer)', 'Zone C (Periphery)']
    data = []
    
    for i, zone in enumerate(zones):
        n_zone = n_households // 3
        
        # Base compensation (decreasing away from core)
        base_comp = rng.normal(50000 - i*10000, 5000, n_zone)
        
        for j in range(n_zone):
            comp = base_comp[j]
            is_indet = rng.random() < indeterminacy_rate
            
            row = {
                'household_id': f'HH-{i+1}-{j:03d}',
                'zone': zone,
                'compensation_T_lower': comp,
                'compensation_T_upper': comp,
                'compensation_I_lower': 0.0,
                'compensation_I_upper': rng.uniform(5000, 15000) if is_indet else 0.0,
                'compensation_F_lower': 0.0,
                'compensation_F_upper': 0.0,
                'is_indeterminate': is_indet
            }
            data.append(row)
            
    df = pd.DataFrame(data)
    metadata = {
        'description': 'Tarkwa mining resettlement compensation data. Disputed bounds or missing deeds map to indeterminacy.',
        'variables': ['compensation'],
        'indeterminacy_rate': indeterminacy_rate
    }
    return df, metadata
