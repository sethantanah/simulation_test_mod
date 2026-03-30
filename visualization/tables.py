import pandas as pd

def style_summary_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    return df.style.set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', '#1565C0'), ('color', 'white'), ('font-family', 'IBM Plex Sans')]},
        {'selector': 'tbody td', 'props': [('font-family', 'IBM Plex Mono')]}
    ]).format(precision=4)

def format_neutrosophic_str(t, i, f):
    return f"N([{t[0]:.3g},{t[1]:.3g}], [{i[0]:.3g},{i[1]:.3g}], [{f[0]:.3g},{f[1]:.3g}])"
