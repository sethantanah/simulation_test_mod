import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import load_css
from data.loader import load_dataset
from core.neutrosophic import neutrosophicate
from core.tests.kruskal_wallis import kruskal_wallis_modified
from core.tests.mann_whitney import mann_whitney_modified
from core.tests.moods_median import moods_median_modified
from visualization.plots import plot_neutrosophic_boxplot

st.set_page_config(page_title="Real-Life Data Application", layout="wide")
load_css()

st.title("🌍 Real-Life Data Application")
st.markdown("Applying modified tests to synthetic real-world datasets from various domains.")

tab_med, tab_econ, tab_eng = st.tabs(["Medicine (COVID-19)", "Economics", "Engineering"])

indet_thresh = st.session_state.indet_threshold

def apply_tests(df, group_col, target_col):
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    unique_groups = df[group_col].unique()
    groups = [df[df[group_col] == g][target_col].tolist() for g in unique_groups]
    n_groups = [neutrosophicate(g, indet_thresh) for g in groups]
    
    k = len(n_groups)
    st.markdown(f"**Research Question:** Do {target_col} differ significantly across {group_col}?")
    
    colA, colB = st.columns(2)
    
    with colA:
        res_kw = kruskal_wallis_modified(n_groups)
        st.success(f"**Modified KW Decision:** {res_kw['decision_zone']}")
        st.markdown(f"$H_N^*$: `{res_kw['H_N']}`")
        
        res_mm = moods_median_modified(n_groups)
        st.info(f"**Modified Mood's Median Decision:** {res_mm['decision_zone']}")
        st.markdown(f"$\chi^2_{{mod}}$: `{res_mm['chi2_modified']:.3f}`")
        
    with colB:
        if k == 2:
            res_mwu = mann_whitney_modified(n_groups[0], n_groups[1])
            st.warning(f"**Modified MWU Decision:** {res_mwu['decision_zone']}")
            st.markdown(f"$U_{{mod}}$: `{res_mwu['U_modified']:.3f}`")
        else:
            st.write("Mann-Whitney U skipped (k > 2 groups).")
            
    st.plotly_chart(plot_neutrosophic_boxplot(n_groups, unique_groups, f"Neutrosophic Spread of {target_col}"), use_container_width=True)

with tab_med:
    df, meta = load_dataset('covid19')
    st.markdown(f"**{meta['description']}** (Indeterminacy rate: {meta['indeterminacy_rate']})")
    apply_tests(df, 'region', 'symptom_severity_T_lower')

with tab_econ:
    df, meta = load_dataset('exchange_rates')
    st.markdown(f"**{meta['description']}**")
    apply_tests(df, 'period', 'rate_T_lower')

with tab_eng:
    df, meta = load_dataset('resettlement')
    st.markdown(f"**{meta['description']}** (Indeterminacy rate: {meta['indeterminacy_rate']})")
    apply_tests(df, 'zone', 'compensation_T_lower')
