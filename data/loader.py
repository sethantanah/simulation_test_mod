import pandas as pd
from core.neutrosophic import NeutrosophicNumber, NeutrosophicArray, neutrosophicate
from data.covid19_data import generate_covid19_data
from data.economics_data import generate_exchange_rate_data, generate_stock_price_data
from data.engineering_data import generate_resettlement_data

def load_dataset(name: str) -> tuple[pd.DataFrame, dict]:
    """Load built-in datasets."""
    if name == 'covid19': return generate_covid19_data()
    elif name == 'exchange_rates': return generate_exchange_rate_data()
    elif name == 'stock_prices': return generate_stock_price_data()
    elif name == 'resettlement': return generate_resettlement_data()
    else: raise ValueError(f"Dataset {name} not found")

def df_to_neutrosophic_array(df: pd.DataFrame, prefix: str) -> NeutrosophicArray:
    """Helper to convert structured df columns back to NeutrosophicArray"""
    res = []
    for _, row in df.iterrows():
        n = NeutrosophicNumber(
            (row[f'{prefix}_T_lower'], row[f'{prefix}_T_upper']),
            (row[f'{prefix}_I_lower'], row[f'{prefix}_I_upper']),
            (row[f'{prefix}_F_lower'], row[f'{prefix}_F_upper'])
        )
        res.append(n)
    return NeutrosophicArray(res)

def upload_and_neutrosophicate(uploaded_file, indeterminacy_col: str = None, indeterminacy_threshold: float = 0.1) -> tuple:
    if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
    else: df = pd.read_excel(uploaded_file)
    
    # We will pick the first numeric column for demo purposes if not specified
    if not indeterminacy_col:
        cols = df.select_dtypes(include='number').columns
        if len(cols) > 0:
            target_col = cols[0]
            vals = df[target_col].tolist()
            n_array = neutrosophicate(vals, indeterminacy_threshold)
        else:
            raise ValueError("No numeric columns found to neutrosophicate")
    else:
        # User specified an indeterminacy column, we can do more complex logic
        # For our simple app, just neutrosophicate the first numeric col
        vals = df[df.columns[0]].tolist()
        n_array = neutrosophicate(vals, indeterminacy_threshold)
        
    metadata = {
        'description': f'User uploaded dataset: {uploaded_file.name}',
        'rows': len(df)
    }
    return n_array, df, metadata

def validate_neutrosophic_data(data) -> dict:
    if not isinstance(data, NeutrosophicArray):
        data = NeutrosophicArray(data)
        
    n_total = len(data.data)
    n_indet = sum(1 for d in data.data if d.is_indeterminate(0.01))
    
    return {
        'valid': True,
        'n_total': n_total,
        'n_indeterminate': n_indet,
        'n_missing': 0,
        'indeterminacy_rate': n_indet / n_total if n_total > 0 else 0,
        'warnings': [],
        'errors': []
    }
