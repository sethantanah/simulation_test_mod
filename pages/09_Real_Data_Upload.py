import pandas as pd
import streamlit as st

from core.tests.kruskal_wallis import (
    build_neutrosophic_group,
    kruskal_wallis_modified,
    kruskal_wallis_original,
)
from core.tests.moods_median import (
    moods_median_modified,
    moods_median_original,
)

st.set_page_config(page_title="Real Data Upload", page_icon="📁", layout="wide")

st.title("Upload a real dataset and run a non-parametric test")
st.caption("Upload a CSV file with one group column and one numeric measurement column, then run Kruskal-Wallis or Mood's Median and download the result as CSV.")


@st.cache_data(show_spinner=False)
def read_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def build_groups_from_frame(df, group_column, value_column):
    groups = []
    group_names = []
    for group_name, group_df in df.groupby(group_column, dropna=False):
        values = pd.to_numeric(group_df[value_column], errors="coerce").dropna().tolist()
        if not values:
            continue
        groups.append(build_neutrosophic_group(values, uncertainty=0.0, falsity_mode="zero"))
        group_names.append(str(group_name))
    return groups, group_names


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = read_csv(uploaded_file)

    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        group_column = st.selectbox("Select group column", options=df.columns.tolist())
    with col2:
        value_column = st.selectbox("Select numeric value column", options=df.columns.tolist())
    with col3:
        test_name = st.selectbox(
            "Select test",
            options=["Kruskal-Wallis", "Mood's Median"],
        )

    alpha = st.slider("Significance level (alpha)", 0.001, 0.1, 0.05, 0.001)

    if st.button("Run analysis", use_container_width=True):
        try:
            if group_column == value_column:
                st.error("Please choose different columns for the group label and the numeric measurement.")
                st.stop()

            numeric_values = pd.to_numeric(df[value_column], errors="coerce")
            if numeric_values.notna().sum() < 3:
                st.error("The selected value column must contain at least 3 numeric observations.")
                st.stop()

            groups, group_names = build_groups_from_frame(df, group_column, value_column)
            if len(groups) < 2:
                st.error("At least two groups are required for this analysis.")
                st.stop()

            with st.spinner("Running analysis..."):
                if test_name == "Kruskal-Wallis":
                    original_result = kruskal_wallis_original(groups, alpha=alpha)
                    modified_result = kruskal_wallis_modified(groups, alpha=alpha)
                    results = [
                        {
                            "test": "Kruskal-Wallis",
                            "variant": "original",
                            "statistic": original_result.get("H_statistic") or original_result.get("H"),
                            "p_value": original_result.get("p_value"),
                            "decision": original_result.get("decision"),
                            "alpha": alpha,
                            "n_groups": len(group_names),
                            "n_total": sum(len(g.data) for g in groups),
                            "groups": ", ".join(group_names),
                        },
                        {
                            "test": "Kruskal-Wallis",
                            "variant": "modified",
                            "statistic": modified_result.get("H_statistic") or modified_result.get("H"),
                            "p_value": modified_result.get("p_value"),
                            "decision": modified_result.get("decision"),
                            "alpha": alpha,
                            "n_groups": len(group_names),
                            "n_total": sum(len(g.data) for g in groups),
                            "groups": ", ".join(group_names),
                        },
                    ]
                else:
                    original_result = moods_median_original(groups, alpha=alpha)
                    modified_result = moods_median_modified(groups, alpha=alpha)
                    results = [
                        {
                            "test": "Mood's Median",
                            "variant": "original",
                            "statistic": original_result.get("chi2"),
                            "p_value": original_result.get("p_value"),
                            "decision": original_result.get("decision"),
                            "alpha": alpha,
                            "n_groups": len(group_names),
                            "n_total": sum(len(g.data) for g in groups),
                            "groups": ", ".join(group_names),
                        },
                        {
                            "test": "Mood's Median",
                            "variant": "modified",
                            "statistic": modified_result.get("chi2"),
                            "p_value": modified_result.get("p_value"),
                            "decision": modified_result.get("decision"),
                            "alpha": alpha,
                            "n_groups": len(group_names),
                            "n_total": sum(len(g.data) for g in groups),
                            "groups": ", ".join(group_names),
                        },
                    ]

            results_df = pd.DataFrame(results)
            st.subheader("Analysis results")
            st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results as CSV",
                data=csv_bytes,
                file_name=f"{test_name.lower().replace(' ', '_')}_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

            with st.expander("Group sizes"):
                group_sizes = pd.Series({name: len(group_df) for name, group_df in df.groupby(group_column, dropna=False)})
                st.dataframe(group_sizes.rename("n_obs"), use_container_width=True)

        except Exception as exc:
            st.error(f"The analysis failed: {exc}")
else:
    st.info("Please upload a CSV file to start.")
