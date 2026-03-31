import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import load_css
from core.neutrosophic import neutrosophicate
from core.tests.kruskal_wallis import kruskal_wallis_modified
from core.tests.mann_whitney import mann_whitney_modified
from core.tests.moods_median import moods_median_modified
from data.loader import load_dataset
from visualization.plots import plot_neutrosophic_boxplot, plot_contingency_heatmap, plot_pvalue_interval, plot_dominance_triple
from visualization.tables import style_summary_table

st.set_page_config(page_title="Modified Neutrosophic Tests", layout="wide")
load_css()

st.title("📈 Modified Neutrosophic Tests")
st.markdown("Run the proposed **enhanced** neutrosophic non-parametric tests featuring interval rankings, NWA, and 3-zone contingency tables.")

tab_kw, tab_mwu, tab_mm = st.tabs(["Modified Kruskal-Wallis", "Modified Mann-Whitney U", "Modified Mood's Median"])

alpha = st.session_state.get('alpha', 0.05)
indet_thresh = st.session_state.get('indet_threshold', 0.10)

def render_modified_ui(test_type):
    col_in, col_out = st.columns([4, 6])
    
    with col_in:
        st.markdown("### Data Configuration")
        datasource = st.radio("Select Data Source", [
            "Built-in: Medicine (COVID-19)", 
            "Built-in: Economics (Exchange Rates)", 
            "Built-in: Engineering (Resettlement)", 
            "Random Synthetic"
        ], key=f"mod_ds_{test_type}")
        
        st.info("Modifications highlight explicit tracking of indeterminacy width (δ) and adaptive weighing (λ).", icon="ℹ️")
        
        with st.expander("Mathematical Formulation"):
            if test_type == "KW":
                st.latex(r"H_N^* = H_N \times (1 + \lambda)")
                st.markdown(r"Where $\lambda = \frac{\text{Indeterminate Count}}{N}$ handles indeterminacy probabilistically.")
            elif test_type == "MWU":
                st.latex(r"U_{mod} = w_T U_T + w_I U_I + w_F U_F")
                st.markdown(r"Where weights $w_T, w_I, w_F$ are Neutrosophic Weighted Average (NWA) values based on pairwise dominance probability.")
            elif test_type == "MM":
                st.latex(r"\chi^2_{mod} = \sum \frac{(O - E)^2}{E}")
                st.markdown(r"Using a $3 \times k$ contingency table which allocates items into **Above**, **Indeterminate (±δ)**, and **Below** median regions, explicitly capturing noise.")

        if st.button("Run Modified Test", type="primary", key=f"mod_run_{test_type}"):
            with st.spinner("Processing Modifications..."):
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
                    rng = np.random.default_rng(123)
                    groups = [rng.normal(5+i*0.5, 1, 40).tolist() for i in range(4)]
                
                if test_type == "MWU": groups = groups[:2]
                    
                n_groups = [neutrosophicate(g, indet_thresh) for g in groups]
                group_names = [f"Group {i+1}" for i in range(len(groups))]
                
                with col_out:
                    st.markdown("### Modified Test Results")
                    
                    if test_type == "KW":
                        res = kruskal_wallis_modified(n_groups)
                        st.markdown(f"**Weighted Statistic $H_N^*$:** `{res['H_N']}`")
                        st.metric("Adaptive Indeterminacy Weight (λ)", f"{res['lambda_weight']:.3f}")
                        st.metric("Mean Rank Interval Width", f"{res['rank_interval_width_mean']:.3f}")
                        
                    elif test_type == "MWU":
                        res = mann_whitney_modified(n_groups[0], n_groups[1])
                        st.markdown(f"**Aggregated Statistic $U_{{mod}}$:** `{res['U_modified']:.2f}`")
                        st.plotly_chart(plot_dominance_triple(res['dominance_prob_T'], res['dominance_prob_I'], res['dominance_prob_F']), use_container_width=True)
                        st.markdown(f"**NWA Weights:** $w_T={res['w_T']:.2f}$, $w_I={res['w_I']:.2f}$, $w_F={res['w_F']:.2f}$")
                        
                    elif test_type == "MM":
                        res = moods_median_modified(n_groups)
                        st.markdown(f"**Modified Statistic $\chi^2_{{mod}}$:** `{res['chi2_modified']:.3f}`")
                        st.metric("Adaptive Band Width (δ)", f"{res['band_width_delta']:.3f}")
                        
                        st.plotly_chart(plot_contingency_heatmap(res['contingency_table_3xk'], ["Above (+δ)", "Indeterminate (±δ)", "Below (-δ)"], group_names, "3xk Contingency Table"), use_container_width=True)
                        
                    p_low, p_up = res['p_interval_modified'] if 'p_interval_modified' in res else res['p_interval']
                    st.plotly_chart(plot_pvalue_interval(p_low, p_up, alpha), use_container_width=True)
                    
                    dec = res['decision_zone']
                    color = "#4CAF50" if "Fail" in dec else ("#F44336" if "Reject" in dec else "#FF9800")
                    st.markdown(f"<h3 style='text-align:center; color:{color};'>{dec}</h3>", unsafe_allow_html=True)
                    
                    st.plotly_chart(plot_neutrosophic_boxplot(n_groups, group_names, f"{test_type} Data Spread"), use_container_width=True)
                    
with tab_kw: render_modified_ui("KW")
with tab_mwu: render_modified_ui("MWU")
with tab_mm: render_modified_ui("MM")
