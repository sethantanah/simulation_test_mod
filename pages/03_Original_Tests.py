import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import load_css
from core.neutrosophic import neutrosophicate
from core.tests.kruskal_wallis import kruskal_wallis_original
from core.tests.mann_whitney import mann_whitney_original
from core.tests.moods_median import moods_median_original
from data.loader import load_dataset
from visualization.plots import plot_neutrosophic_boxplot, plot_contingency_heatmap, plot_pvalue_interval
from visualization.tables import style_summary_table

st.set_page_config(page_title="Original Neutrosophic Tests", layout="wide")
load_css()

st.title("📊 Original Neutrosophic Tests")
st.markdown("Run the unmodified neutrosophic non-parametric tests as proposed in current literature.")

tab_kw, tab_mwu, tab_mm = st.tabs(["Kruskal-Wallis", "Mann-Whitney U", "Mood's Median"])

# Global config
alpha = st.session_state.get('alpha', 0.05)
indet_thresh = st.session_state.get('indet_threshold', 0.10)

def render_test_ui(test_type):
    col_in, col_out = st.columns([4, 6])
    
    with col_in:
        st.markdown("### Data Configuration")
        datasource = st.radio("Select Data Source", [
            "Built-in: Medicine (COVID-19)", 
            "Built-in: Economics (Exchange Rates)", 
            "Built-in: Engineering (Resettlement)", 
            "Random Synthetic"
        ], key=f"ds_{test_type}")
        
        with st.expander("Mathematical Formulation"):
            if test_type == "KW":
                st.latex(r"H_N = \frac{12}{N(N+1)} \sum_{i=1}^k \frac{R_i^2}{n_i} - 3(N+1)")
                st.markdown("Where $R_i$ is the sum of ranks for the $i$-th group and $n_i$ is the sample size.")
            elif test_type == "MWU":
                st.latex(r"U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1")
                st.markdown("Where $n_1, n_2$ are sample sizes, and $R_1$ is the rank sum of the first sample.")
            elif test_type == "MM":
                st.latex(r"\chi^2 = \sum \frac{(O - E)^2}{E}")
                st.markdown("Using a $2 \times k$ contingency table computing expected and observed counts above and below the grand median.")
        
        if st.button("Run Test", type="primary", key=f"run_{test_type}"):
            with st.spinner("Processing..."):
                # Load data
                if "Engineering" in datasource:
                    df, _ = load_dataset('resettlement')
                    groups = [df[df['zone'] == z]['compensation_T_lower'].tolist() for z in df['zone'].unique()]
                elif "Medicine" in datasource:
                    df, _ = load_dataset('covid19')
                    groups = [df[df['region'] == r]['symptom_severity_T_lower'].tolist() for r in df['region'].unique()]
                elif "Economics" in datasource:
                    df, _ = load_dataset('exchange_rates')
                    groups = [df[df['period'] == p]['rate_T_lower'].tolist() for p in df['period'].unique()]
                else:
                    rng = np.random.default_rng(42)
                    groups = [rng.normal(10+i, 2, 30).tolist() for i in range(3)]
                
                if test_type == "MWU":
                    groups = groups[:2] # MWU only takes 2 groups
                    
                # Neutrosophicate
                n_groups = [neutrosophicate(g, indet_thresh) for g in groups]
                group_names = [f"Group {i+1}" for i in range(len(groups))]
                
                with col_out:
                    st.markdown("### Test Results")
                    
                    if test_type == "KW":
                        res = kruskal_wallis_original(n_groups)
                        st.markdown(f"**Statistic $H_N$:** `{res['H_N']}`")
                        
                    elif test_type == "MWU":
                        res = mann_whitney_original(n_groups[0], n_groups[1])
                        st.markdown(f"**Statistic $U_N$:** `{res['U_original']}`")
                        
                    elif test_type == "MM":
                        res = moods_median_original(n_groups)
                        st.markdown(f"**Statistic $\chi^2_N$:** `{res['chi2_original']}`")
                        st.markdown(f"**Grand Median $M_N$:** `{res['grand_median']}`")
                        
                    p_low, p_up = res['p_interval_original'] if 'p_interval_original' in res else res['p_interval']
                    
                    # Visuals
                    st.plotly_chart(plot_pvalue_interval(p_low, p_up, alpha), use_container_width=True)
                    
                    dec = res['decision_zone']
                    color = "#4CAF50" if "Fail" in dec else ("#F44336" if "Reject" in dec else "#FF9800")
                    st.markdown(f"<h3 style='text-align:center; color:{color};'>{dec}</h3>", unsafe_allow_html=True)
                    
                    st.plotly_chart(plot_neutrosophic_boxplot(n_groups, group_names, f"{test_type} Data Spread"), use_container_width=True)
                    
                    if test_type == "MM":
                        st.plotly_chart(plot_contingency_heatmap(res['contingency_table_2xk'], ["Above Median", "Below Median"], group_names, "2xk Contingency Table"), use_container_width=True)

with tab_kw: render_test_ui("KW")
with tab_mwu: render_test_ui("MWU")
with tab_mm: render_test_ui("MM")
