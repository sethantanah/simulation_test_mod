import numpy as np
import pandas as pd

def generate_exchange_rate_data(n_months=60, indeterminacy_rate=0.15, random_seed=42):
    rng = np.random.default_rng(seed=random_seed)
    months = pd.date_range(start='2020-01-01', periods=n_months, freq='ME')
    
    # Base trend
    base_rate = 5.5 + np.cumsum(rng.normal(0.05, 0.2, size=n_months))
    
    data = []
    for i in range(n_months):
        rate = base_rate[i]
        is_indet = rng.random() < indeterminacy_rate
        
        row = {
            'date': months[i],
            'period': 'COVID-19' if i < 24 else ('Recovery' if i < 48 else 'Recent'),
            'rate_T_lower': rate,
            'rate_T_upper': rate,
            'rate_I_lower': 0.0,
            'rate_I_upper': rng.uniform(0.1, 0.5) if is_indet else 0.0,
            'rate_F_lower': 0.0,
            'rate_F_upper': 0.0,
            'is_indeterminate': is_indet
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    metadata = {
        'description': 'GHS/USD monthly exchange rates with economic uncertainty periods.',
        'variables': ['rate']
    }
    return df, metadata

def generate_stock_price_data(n_stocks=5, n_days=252, indeterminacy_rate=0.12, random_seed=42):
    rng = np.random.default_rng(seed=random_seed)
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
    sectors = ['Finance', 'Tech', 'Energy', 'Mining', 'Agric']
    
    data = []
    for s in range(n_stocks):
        sector = sectors[s]
        price = 100.0
        
        for d in range(n_days):
            ret = rng.normal(0.0005, 0.02)
            price *= (1 + ret)
            
            is_indet = rng.random() < indeterminacy_rate
            
            row = {
                'date': dates[d],
                'sector': sector,
                'price_T_lower': price,
                'price_T_upper': price,
                'price_I_lower': 0.0,
                'price_I_upper': price * rng.uniform(0.01, 0.05) if is_indet else 0.0,
                'price_F_lower': 0.0,
                'price_F_upper': 0.0,
                'is_indeterminate': is_indet
            }
            data.append(row)
            
    df = pd.DataFrame(data)
    metadata = {
        'description': 'Simulated stock prices across sectors with trading halts mapping to indeterminacy.',
        'variables': ['price']
    }
    return df, metadata
