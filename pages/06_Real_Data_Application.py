"""
Real-Life Data Application Dashboard

This module applies modified neutrosophic tests to real-world datasets from various domains,
with support for custom data upload and comprehensive analysis.

Author: Akua Agyapomah Oteng (PhD Candidate)
Institution: University of Mines and Technology (UMaT), Ghana
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import load_css
from data.loader import load_dataset
from core.neutrosophic import neutrosophicate, NeutrosophicArray
from core.tests.kruskal_wallis import kruskal_wallis_modified
from core.tests.mann_whitney import mann_whitney_modified
from core.tests.moods_median import moods_median_modified
from visualization.plots import plot_neutrosophic_boxplot, plot_pvalue_interval, plot_dominance_triple

st.set_page_config(
    page_title="Real-Life Data Application | PhD Research",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css()

# ============================================================================
# TITLE AND INTRODUCTION
# ============================================================================

st.markdown("""
<div class="hero-section" style="text-align: center; padding: 2rem; margin-bottom: 2rem; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius: 10px;">
    <h1 style="color: white; margin: 0;">🌍 Real-Life Data Application</h1>
    <p style="color: white; margin-top: 0.5rem; font-size: 1.1em;">
        Applying modified neutrosophic tests to real-world datasets from medicine, economics, and engineering
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Analysis Configuration")
    
    # Get global settings
    indet_thresh = st.session_state.get('indet_threshold', 0.10)
    alpha = st.session_state.get('alpha', 0.05)
    
    st.info(f"""
    **Current Settings:**
    - Indeterminacy Threshold: {indet_thresh}
    - Significance Level (α): {alpha}
    """)
    
    st.markdown("---")
    st.markdown("### 🧪 Select Tests to Run")
    
    # Test selection checkboxes
    run_kw = st.checkbox("📈 Kruskal-Wallis Test", value=True, 
                         help="Compares multiple groups for overall differences")
    run_mwu = st.checkbox("📉 Mann-Whitney U Test", value=True,
                          help="Compares two groups for pairwise differences (auto-runs for all pairs if >2 groups)")
    run_mm = st.checkbox("📊 Mood's Median Test", value=True,
                         help="Robust median comparison using contingency table")
    
    st.markdown("---")
    st.markdown("### 📊 Analysis Features")
    st.markdown("""
    For each dataset, the dashboard provides:
    
    - **Selected Tests** with modification metrics
    - **Uncertainty Metrics** - λ, rank interval widths, band width δ
    - **Visualizations** - Boxplots, p-value intervals, dominance triples
    - **Indeterminacy Analysis** - Distribution of uncertainty
    """)
    
    st.markdown("---")
    st.markdown("### 📁 Data Format Requirements")
    st.markdown("""
    **For custom upload:**
    - CSV or Excel file
    - One column for group labels (categorical)
    - One column for numerical values
    - Missing values allowed (will be treated as indeterminate)
    """)
    
    # Store selections in session state
    st.session_state['run_kw'] = run_kw
    st.session_state['run_mwu'] = run_mwu
    st.session_state['run_mm'] = run_mm

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_kruskal_wallis_analysis(n_groups, group_names, alpha):
    """Run and display Kruskal-Wallis test results"""
    st.markdown("#### 📈 Modified Kruskal-Wallis Test")
    with st.spinner("Running Kruskal-Wallis test..."):
        res_kw = kruskal_wallis_modified(n_groups, alpha=alpha)
    
    # Display KW results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Decision", res_kw['decision_zone'])
    with col2:
        st.metric("Adaptive Weight (λ)", f"{res_kw.get('lambda_weight', 0):.4f}")
    with col3:
        st.metric("Mean Rank Interval Width", f"{res_kw.get('rank_interval_width_mean', 0):.4f}")
    
    st.markdown(f"**H-statistic:** `{res_kw['H_N']}`")
    st.markdown(f"**H-interval:** [{res_kw['H_low']:.4f}, {res_kw['H_high']:.4f}]")
    
    # P-value interval
    p_low, p_up = res_kw.get('p_interval', (0, 1))
    st.plotly_chart(plot_pvalue_interval(p_low, p_up, alpha), use_container_width=True)
    
    return res_kw

def run_moods_median_analysis(n_groups, group_names, alpha):
    """Run and display Mood's Median test results"""
    st.markdown("#### 📊 Modified Mood's Median Test")
    with st.spinner("Running Mood's Median test..."):
        res_mm = moods_median_modified(n_groups, alpha=alpha)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Decision", res_mm['decision_zone'])
    with col2:
        st.metric("Band Width (δ)", f"{res_mm.get('band_width_delta', 0):.4f}")
    with col3:
        st.metric("χ²-statistic", f"{res_mm['chi2_modified']:.4f}")
    
    # Show contingency table
    if 'contingency_table_3xk' in res_mm:
        with st.expander("3×k Contingency Table"):
            st.dataframe(
                pd.DataFrame(
                    res_mm['contingency_table_3xk'],
                    index=["Above (+δ)", "Indeterminate (±δ)", "Below (-δ)"],
                    columns=group_names
                ),
                use_container_width=True
            )
    
    return res_mm

def run_mann_whitney_analysis(n_groups, group_names, k, alpha):
    """Run and display Mann-Whitney U test results"""
    if k == 2:
        st.markdown("#### 📉 Modified Mann-Whitney U Test")
        with st.spinner("Running Mann-Whitney test..."):
            res_mwu = mann_whitney_modified(n_groups[0], n_groups[1], alpha=alpha)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Decision", res_mwu['decision_zone'])
        with col2:
            st.metric("U-statistic", f"{res_mwu['U_modified']:.4f}")
        with col3:
            st.metric("Effect Size (r)", f"{res_mwu['effect_size_neutrosophic']:.4f}")
        
        # Dominance probabilities
        if 'dominance_prob_T' in res_mwu:
            st.plotly_chart(plot_dominance_triple(
                res_mwu['dominance_prob_T'],
                res_mwu['dominance_prob_I'],
                res_mwu['dominance_prob_F']
            ), use_container_width=True)
        
        # NWA weights
        if 'w_T' in res_mwu:
            with st.expander("📊 NWA Weights (Data Quality)"):
                st.markdown(f"- w_T (Determinate): {res_mwu['w_T']:.3f}")
                st.markdown(f"- w_I (Indeterminate): {res_mwu['w_I']:.3f}")
                st.markdown(f"- w_F (Missing): {res_mwu['w_F']:.3f}")
        
        return res_mwu
    
    else:
        st.markdown("#### 📉 Pairwise Comparisons (Mann-Whitney U)")
        st.info(f"Running pairwise comparisons for {k} groups...")
        
        # Create tabs for pairwise comparisons
        pairwise_tabs = st.tabs([f"{group_names[i]} vs {group_names[j]}" 
                                  for i in range(k) for j in range(i+1, k)])
        
        pairwise_results = []
        tab_idx = 0
        for i in range(k):
            for j in range(i+1, k):
                with pairwise_tabs[tab_idx]:
                    with st.spinner(f"Comparing {group_names[i]} vs {group_names[j]}..."):
                        res_pair = mann_whitney_modified(n_groups[i], n_groups[j], alpha=alpha)
                        pairwise_results.append({
                            'Group 1': group_names[i],
                            'Group 2': group_names[j],
                            'U-statistic': res_pair['U_modified'],
                            'Decision': res_pair['decision_zone'],
                            'Effect Size': res_pair['effect_size_neutrosophic'],
                            'P-Interval': res_pair.get('p_interval', (0, 1))
                        })
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Decision", res_pair['decision_zone'])
                        with col2:
                            st.metric("U-statistic", f"{res_pair['U_modified']:.4f}")
                        with col3:
                            st.metric("Effect Size", f"{res_pair['effect_size_neutrosophic']:.4f}")
                        
                        # Dominance probabilities
                        if 'dominance_prob_T' in res_pair:
                            st.plotly_chart(plot_dominance_triple(
                                res_pair['dominance_prob_T'],
                                res_pair['dominance_prob_I'],
                                res_pair['dominance_prob_F']
                            ), use_container_width=True)
                        
                        # P-value interval
                        p_low, p_up = res_pair.get('p_interval', (0, 1))
                        st.plotly_chart(plot_pvalue_interval(p_low, p_up, alpha), use_container_width=True)
                
                tab_idx += 1
        
        # Summary table of pairwise comparisons
        if pairwise_results:
            with st.expander("📊 Pairwise Comparison Summary"):
                summary_df = pd.DataFrame(pairwise_results)
                summary_df['U-statistic'] = summary_df['U-statistic'].round(4)
                summary_df['Effect Size'] = summary_df['Effect Size'].round(4)
                st.dataframe(summary_df, use_container_width=True)
        
        return pairwise_results

def plot_indeterminacy_analysis(n_groups, group_names):
    """Plot indeterminacy distribution analysis"""
    st.markdown("### 🔬 Indeterminacy Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # I-width distribution by group
        fig_iwidth = go.Figure()
        for i, g in enumerate(n_groups):
            i_widths = [n.I[1] - n.I[0] for n in g.data]
            fig_iwidth.add_trace(go.Box(
                y=i_widths,
                name=group_names[i],
                boxmean='sd',
                marker_color='lightblue'
            ))
        fig_iwidth.update_layout(
            title="Indeterminacy Width Distribution by Group",
            yaxis_title="I-interval Width",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_iwidth, use_container_width=True)
    
    with col2:
        # Proportion of indeterminate observations
        indet_props = []
        for i, g in enumerate(n_groups):
            n_indet = sum(1 for n in g.data if n.is_indeterminate())
            indet_props.append(n_indet / len(g.data))
        
        fig_prop = go.Figure(data=[
            go.Bar(x=group_names, y=indet_props, marker_color='#FF9800')
        ])
        fig_prop.update_layout(
            title="Proportion of Indeterminate Observations",
            yaxis_title="Indeterminate Proportion",
            yaxis=dict(tickformat=".0%"),
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_prop, use_container_width=True)

def apply_tests(df, group_col, target_col, dataset_name="Custom"):
    """Apply selected modified tests to the dataset with comprehensive analysis"""
    
    # Get test selection from session state
    run_kw = st.session_state.get('run_kw', True)
    run_mwu = st.session_state.get('run_mwu', True)
    run_mm = st.session_state.get('run_mm', True)
    alpha = st.session_state.get('alpha', 0.05)
    
    # Dataset info
    st.markdown(f"### 📋 Dataset: {dataset_name}")
    
    # Display basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Observations", len(df))
    with col2:
        st.metric("Number of Groups", df[group_col].nunique())
    with col3:
        st.metric("Missing Values", df[target_col].isna().sum())
    with col4:
        missing_pct = (df[target_col].isna().sum() / len(df)) * 100
        st.metric("Missing %", f"{missing_pct:.1f}%")
    
    # Data preview
    with st.expander("📊 Data Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
        
        # Summary statistics by group
        st.markdown("**Summary Statistics by Group:**")
        summary = df.groupby(group_col)[target_col].agg([
            ('Count', 'count'),
            ('Mean', 'mean'),
            ('Std', 'std'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Missing', lambda x: x.isna().sum())
        ]).round(3)
        st.dataframe(summary, use_container_width=True)
    
    # Neutrosophication
    unique_groups = df[group_col].unique()
    groups = []
    group_names = []
    
    for g in unique_groups:
        group_data = df[df[group_col] == g][target_col].tolist()
        # Remove NaN values for neutrosophication (they will be handled by neutrosophicate)
        group_data = [x for x in group_data if pd.notna(x)]
        if len(group_data) > 0:
            groups.append(group_data)
            group_names.append(str(g))
    
    if len(groups) < 2:
        st.error("Need at least 2 groups with non-missing data for testing.")
        return
    
    # Apply neutrosophication
    n_groups = [neutrosophicate(g, indet_thresh) for g in groups]
    k = len(n_groups)
    
    # Research question
    st.markdown(f"### 🎯 Research Question")
    st.markdown(f"Do **{target_col}** differ significantly across **{group_col}** categories?")
    
    # Check if any test selected
    if not (run_kw or run_mwu or run_mm):
        st.warning("⚠️ Please select at least one test to run from the sidebar.")
        return
    
    # Run selected tests
    results = {}
    
    if run_kw:
        with st.container():
            st.markdown("---")
            results['kw'] = run_kruskal_wallis_analysis(n_groups, group_names, alpha)
    
    if run_mm:
        with st.container():
            st.markdown("---")
            results['mm'] = run_moods_median_analysis(n_groups, group_names, alpha)
    
    if run_mwu:
        with st.container():
            st.markdown("---")
            results['mwu'] = run_mann_whitney_analysis(n_groups, group_names, k, alpha)
    
    # Visualizations (always shown)
    st.markdown("---")
    st.markdown("### 📊 Data Visualization")
    
    # Boxplot
    fig_box = plot_neutrosophic_boxplot(n_groups, group_names, f"Distribution of {target_col} by {group_col}")
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Indeterminacy analysis
    plot_indeterminacy_analysis(n_groups, group_names)
    
    # Interpretation summary
    with st.expander("📖 Statistical Interpretation", expanded=True):
        st.markdown("#### 🔬 Key Findings")
        
        if run_kw and 'kw' in results:
            kw_decision = results['kw']['decision_zone']
            if kw_decision == "Reject H0":
                st.success(f"""
                **Kruskal-Wallis Test:** Strong evidence that {target_col} differs significantly across {group_col} categories.
                The p-value interval is entirely below α = {alpha}, indicating robust rejection of the null hypothesis.
                """)
            elif kw_decision == "Indeterminate Decision":
                st.warning(f"""
                **Kruskal-Wallis Test:** Inconclusive result. The p-value interval contains α = {alpha},
                indicating that data uncertainty prevents a clear decision. Consider collecting more data
                or reducing measurement error.
                """)
            else:
                st.info(f"""
                **Kruskal-Wallis Test:** No significant difference detected. The p-value interval is entirely
                above α = {alpha}, indicating insufficient evidence of group differences.
                """)
            
            # Modification impact
            if 'lambda_weight' in results['kw']:
                st.markdown(f"""
                **Modification Impact:** The adaptive weight λ = {results['kw']['lambda_weight']:.4f} indicates
                {results['kw']['lambda_weight']*100:.1f}% of observations contain indeterminacy. The test statistic
                was scaled by factor {1+results['kw']['lambda_weight']:.4f} to account for this uncertainty.
                """)
        
        if run_mm and 'mm' in results:
            mm_decision = results['mm']['decision_zone']
            st.markdown(f"**Mood's Median Test:** {mm_decision}")
            if 'band_width_delta' in results['mm']:
                st.markdown(f"Adaptive band width δ = {results['mm']['band_width_delta']:.4f}")
        
        if run_mwu and 'mwu' in results:
            if k == 2:
                mwu_decision = results['mwu']['decision_zone']
                st.markdown(f"**Mann-Whitney U Test:** {mwu_decision}")
            else:
                st.markdown(f"**Pairwise Comparisons:** See detailed results above")

# ============================================================================
# DATA SOURCE SELECTION
# ============================================================================

data_source = st.radio(
    "Select Data Source",
    ["Built-in Datasets", "Upload Custom Dataset"],
    horizontal=True
)

if data_source == "Built-in Datasets":
    # Built-in dataset tabs
    tab_med, tab_econ, tab_eng = st.tabs(["🏥 Medicine (COVID-19)", "🏥 Medicine (COVID-19)", "🏗️ Engineering (Resettlement)"])
    
    with tab_med:
        df, meta = load_dataset('covid19')
        st.markdown(f"**{meta['description']}**")
        if 'indeterminacy_rate' in meta:
            st.caption(f"Indeterminacy rate: {meta['indeterminacy_rate']}")
        apply_tests(df, 'region', 'symptom_severity_T_lower', dataset_name="COVID-19 Patient Data")
    
    with tab_econ:
        df, meta = load_dataset('exchange_rates')
        st.markdown(f"**{meta['description']}**")
        apply_tests(df, 'period', 'rate_T_lower', dataset_name="Exchange Rate Data")
    
    with tab_eng:
        df, meta = load_dataset('resettlement')
        st.markdown(f"**{meta['description']}**")
        if 'indeterminacy_rate' in meta:
            st.caption(f"Indeterminacy rate: {meta['indeterminacy_rate']}")
        apply_tests(df, 'zone', 'compensation_T_lower', dataset_name="Resettlement Survey Data")

else:
    # Custom file upload
    st.markdown("### 📁 Upload Custom Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file with one column for group labels and one column for numerical values"
    )
    
    if uploaded_file is not None:
        # Read file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
            
            # Column selection
            st.markdown("### 🔧 Column Configuration")
            st.info("Select which columns contain group labels and numerical values.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                group_col = st.selectbox(
                    "Select Group/Label Column",
                    options=df.columns.tolist(),
                    help="Categorical column that defines groups"
                )
            
            with col2:
                target_col = st.selectbox(
                    "Select Numerical Value Column",
                    options=df.columns.tolist(),
                    help="Numerical column to analyze"
                )
            
            # Data preview
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
                st.markdown(f"**Data Types:**")
                st.dataframe(df.dtypes.to_frame('Type'), use_container_width=True)
            
            # Data cleaning options
            with st.expander("🔧 Data Cleaning Options"):
                clean_na = st.checkbox("Remove rows with missing values", value=False)
                if clean_na:
                    df = df.dropna(subset=[target_col])
                    st.info(f"Removed rows with missing values. New row count: {len(df)}")
                
                outlier_method = st.selectbox(
                    "Outlier Treatment",
                    ["None", "Winsorize (1%)", "Winsorize (5%)", "Remove Outliers (IQR)"]
                )
                
                if outlier_method != "None":
                    Q1 = df[target_col].quantile(0.25)
                    Q3 = df[target_col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if "Winsorize" in outlier_method:
                        pct = 0.01 if "1%" in outlier_method else 0.05
                        lower = df[target_col].quantile(pct)
                        upper = df[target_col].quantile(1-pct)
                        df[target_col] = df[target_col].clip(lower, upper)
                        st.info(f"Applied Winsorization at {pct*100:.0f}% level")
                    elif "Remove" in outlier_method:
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        original_len = len(df)
                        df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
                        st.info(f"Removed {original_len - len(df)} outliers. New row count: {len(df)}")
            
            # Run analysis button
            if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                if group_col and target_col:
                    # Validate columns
                    if df[group_col].nunique() < 2:
                        st.error("Need at least 2 unique groups for comparison.")
                    else:
                        apply_tests(df, group_col, target_col, dataset_name="Custom Dataset")
                else:
                    st.error("Please select both group and target columns.")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Please ensure your file is properly formatted (CSV or Excel).")
    
    else:
        st.info("👈 Upload a CSV or Excel file to begin analysis.")
        
        # Show example format
        with st.expander("📋 Example Data Format"):
            st.markdown("""
            **Example CSV structure:**
            
            ```csv
            region,recovery_time
            Accra,12.5
            Accra,11.8
            Kumasi,13.2
            Kumasi,12.1
            Takoradi,11.5
            Takoradi,14.0
            ### 📌 Data Requirements

Your dataset should contain:

- **One column** with **group labels** (categorical)
- **One column** with **numerical values**
- **Missing values (NaN)** are allowed and will be treated as **fully indeterminate**
""")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

st.markdown("""
<div style="text-align: center; padding: 1.2rem; color: #666; font-size: 0.9rem;">

<p><strong>© 2024–2026 Akua Agyapomah Oteng</strong></p>

<p>
University of Mines and Technology, Tarkwa, Ghana
</p>

<p style="margin-top: 0.5rem;">
Modified tests incorporate <strong>adaptive indeterminacy weighting (λ)</strong>, 
<strong>interval-valued ranking</strong>, and 
<strong>three-zone contingency tables</strong> 
for robust real-world analysis.
</p>

</div>
""", unsafe_allow_html=True)