"""
Monte Carlo Simulation Dashboard for Neutrosophic Tests

Author: Akua Agyapomah Oteng (PhD Candidate)
Institution: University of Mines and Technology (UMaT), Ghana
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app4 import load_css
from core.simulation import MonteCarloSimulation
from visualization.plots import (
    plot_power_curves, plot_type1_heatmap, plot_relative_efficiency,
    plot_decision_stability,
)

st.set_page_config(
    page_title="Monte Carlo Simulation Dashboard | PhD Research",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()

# ============================================================================
# TITLE
# ============================================================================

st.markdown("""
<div class="hero-section" style="text-align:center;padding:2rem;margin-bottom:2rem;
     background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:10px;">
  <h1 style="color:white;margin:0;">🖥️ Monte Carlo Simulation Dashboard</h1>
  <p style="color:white;margin-top:0.5rem;font-size:1.1em;">
    Rigorous statistical simulations to evaluate original vs. modified neutrosophic
    test performance
  </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Simulation Configuration")

    test_type = st.selectbox(
        "Test to Simulate",
        ["kruskal_wallis", "mann_whitney", "moods_median"],
        format_func=lambda x: {
            "kruskal_wallis": "📈 Kruskal-Wallis Test",
            "mann_whitney":   "📉 Mann-Whitney U Test",
            "moods_median":   "📊 Mood's Median Test",
        }.get(x, x),
    )

    # IMPROVEMENT: invalidate cached results when test_type changes
    if st.session_state.get("_last_test_type") != test_type:
        st.session_state.pop(f"sim_results_{test_type}", None)
        st.session_state["_last_test_type"] = test_type

    st.markdown("---")
    st.markdown("### 📊 Experimental Design Parameters")

    sample_sizes = st.multiselect(
        "Sample Sizes (n per group)", [20, 50, 100, 200], default=[20, 50],
        help="Number of observations in each group",
    )
    indet_levels = st.multiselect(
        "Indeterminacy Levels (δ)", [0.0, 0.10, 0.25, 0.40], default=[0.0, 0.10, 0.25],
        help="Proportion of observations with added indeterminacy",
    )
    distributions = st.multiselect(
        "Distributions",
        ["normal", "skewed", "heavy_tailed", "uniform", "bimodal"],
        default=["normal", "skewed"],
        help="Distribution types for data generation",
    )
    effect_sizes = st.multiselect(
        "Effect Sizes (Cohen's d)", [0.0, 0.2, 0.5, 0.8, 1.0], default=[0.0, 0.5],
        help="0.0 = null hypothesis (Type I error), >0 = alternative hypothesis (power)",
    )

    st.markdown("---")
    st.markdown("### 🔬 Simulation Settings")

    n_sims = st.session_state.get("n_sims", 1000)
    st.info(f"📊 Using **{n_sims}** simulations per condition")
    st.caption("Adjust in Global Settings in main app")

    # IMPROVEMENT: show cost estimate before user clicks Run
    all_params_set = bool(sample_sizes and indet_levels and distributions and effect_sizes)
    if all_params_set:
        n_conditions = (len(sample_sizes) * len(indet_levels)
                        * len(distributions) * len(effect_sizes))
        total_iters  = n_conditions * n_sims
        est_min      = max(1, int(total_iters * 0.002 / 60))   # ~2 ms per iteration
        st.markdown(f"**Estimated workload:**")
        st.markdown(
            f"- {n_conditions} conditions × {n_sims} sims = **{total_iters:,} iterations**"
        )
        if est_min > 5:
            st.warning(f"⏱️ Estimated runtime: ~{est_min} min. Consider reducing parameters.")
        else:
            st.success(f"⏱️ Estimated runtime: ~{est_min} min")

    st.markdown("---")
    st.markdown("### 📈 Expected Performance")
    st.success("""
    **Modified Tests Expected Benefits:**
    - 📈 Higher power (5–15% improvement)
    - 🎯 Better Type I error control
    - 🔄 More stable decisions
    - 📊 Narrower p-value intervals
    """)


# ============================================================================
# HELPER FUNCTIONS — VISUALISATIONS
# ============================================================================

def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Division that returns fallback instead of inf/NaN."""
    if denominator == 0 or not np.isfinite(denominator):
        return fallback
    result = numerator / denominator
    return result if np.isfinite(result) else fallback


def plot_power_comparison(df: pd.DataFrame) -> go.Figure:
    """Power comparison: original vs modified across four conditioning variables."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Power by Effect Size",
            "Power by Indeterminacy Level (δ)",
            "Power by Sample Size",
            "Power by Distribution",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
    )

    power_df = df[df["effect_size"] > 0].copy()
    if power_df.empty:
        fig.add_annotation(text="No effect_size > 0 selected", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5, font_size=16)
        return fig

    COLORS = {"original": "#2196F3", "modified": "#FF5722"}

    for variant in ["original", "modified"]:
        vd = power_df[power_df["variant"] == variant]
        dash = "solid" if variant == "modified" else "dash"

        for (row, col), (group_col, x_title) in zip(
            [(1,1),(1,2),(2,1)],
            [("effect_size","Effect Size"),
             ("delta","Indeterminacy Level (δ)"),
             ("n","Sample Size (n)")],
        ):
            means = vd.groupby(group_col)["power"].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=means[group_col], y=means["power"],
                mode="lines+markers", name=variant.title(),
                line=dict(width=2, dash=dash, color=COLORS[variant]),
                marker=dict(size=8), showlegend=(row == 1 and col == 1),
            ), row=row, col=col)

        # Distribution bar chart
        dmeans = vd.groupby("distribution")["power"].mean().reset_index()
        fig.add_trace(go.Bar(
            x=dmeans["distribution"], y=dmeans["power"],
            name=variant.title(), marker_color=COLORS[variant],
            showlegend=False,
        ), row=2, col=2)

    fig.update_layout(
        title="📈 Power Analysis: Original vs Modified Tests",
        height=750, showlegend=True,
        legend=dict(x=0.02, y=0.98), template="plotly_white",
        barmode="group",
    )
    for (r,c), txt in zip([(1,1),(1,2),(2,1),(2,2)],
                           ["Effect Size","Indeterminacy Level (δ)","Sample Size (n)","Distribution"]):
        fig.update_xaxes(title_text=txt, row=r, col=c)
        fig.update_yaxes(title_text="Power", row=r, col=c, range=[0, 1.05])
    return fig


def plot_type1_comparison(df: pd.DataFrame) -> go.Figure:
    """Type I error by indeterminacy level and sample size, with α reference line."""
    type1_df = df[df["effect_size"] == 0].copy()

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Type I Error by Indeterminacy Level",
                        "Type I Error by Sample Size"),
        specs=[[{"type":"bar"},{"type":"bar"}]])

    if type1_df.empty:
        fig.add_annotation(text="No effect_size = 0 selected — Type I error not measured",
                           showarrow=False, xref="paper", yref="paper",
                           x=0.5, y=0.5, font_size=14)
        return fig

    COLORS = {"original": "#2196F3", "modified": "#FF5722"}
    alpha_val = st.session_state.get("alpha", 0.05)

    for (row, col), group_col in [((1,1),"delta"), ((1,2),"n")]:
        means = type1_df.groupby([group_col,"variant"])["type1_error"].mean().reset_index()
        for variant in ["original","modified"]:
            vd = means[means["variant"]==variant]
            fig.add_trace(go.Bar(
                x=vd[group_col], y=vd["type1_error"],
                name=variant.title(), marker_color=COLORS[variant],
                text=[f"{v:.3f}" for v in vd["type1_error"]],
                textposition="outside",
                showlegend=(col==1),
            ), row=row, col=col)
        fig.add_hline(y=alpha_val, line_dash="dash", line_color="red",
                      annotation_text=f"α = {alpha_val}", row=row, col=col)

    fig.update_layout(title="🎯 Type I Error Control", height=480,
                      showlegend=True, template="plotly_white", barmode="group")
    fig.update_xaxes(title_text="Indeterminacy Level (δ)", row=1, col=1)
    fig.update_xaxes(title_text="Sample Size (n)", row=1, col=2)
    fig.update_yaxes(title_text="Type I Error Rate", row=1, col=1)
    fig.update_yaxes(title_text="Type I Error Rate", row=1, col=2)
    return fig


def plot_relative_efficiency_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap of relative efficiency (modified / original).
    FIX: filter variant == 'modified' BEFORE groupby so that NaN original rows
    do not enter the aggregation (even though pandas mean() skips NaN, explicit
    filtering is clearer and future-proof).
    """
    eff_df = df[(df["effect_size"] > 0) & (df["variant"] == "modified")].copy()

    if eff_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    hm = eff_df.groupby(["delta","n"])["relative_efficiency"].mean().reset_index()
    pivot = hm.pivot(index="delta", columns="n", values="relative_efficiency")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale="RdYlGn", zmid=1.0,
        text=np.round(pivot.values, 2), texttemplate="%{text}",
        textfont={"size": 12},
        colorbar_title="Relative<br>Efficiency",
    ))
    fig.update_layout(
        title="⚡ Relative Efficiency: Modified vs Original",
        xaxis_title="Sample Size (n)",
        yaxis_title="Indeterminacy Level (δ)",
        height=420, template="plotly_white",
    )
    return fig


def plot_decision_stability_comparison(df: pd.DataFrame) -> go.Figure:
    """Decision stability by indeterminacy level and effect size."""
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Stability by Indeterminacy Level",
                        "Stability by Effect Size"),
        specs=[[{"type":"bar"},{"type":"scatter"}]])

    COLORS = {"original": "#2196F3", "modified": "#FF5722"}

    for variant in ["original","modified"]:
        vd = df[df["variant"]==variant]

        indet_s = vd.groupby("delta")["decision_stability"].mean().reset_index()
        fig.add_trace(go.Bar(
            x=indet_s["delta"], y=indet_s["decision_stability"],
            name=variant.title(), marker_color=COLORS[variant],
            text=[f"{v:.3f}" for v in indet_s["decision_stability"]],
            textposition="outside",
        ), row=1, col=1)

        eff_s = vd.groupby("effect_size")["decision_stability"].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=eff_s["effect_size"], y=eff_s["decision_stability"],
            mode="lines+markers", name=variant.title(),
            line=dict(width=2, color=COLORS[variant]),
            marker=dict(size=8), showlegend=False,
        ), row=1, col=2)

    fig.update_layout(title="🔄 Decision Stability", height=420,
                      showlegend=True, template="plotly_white", barmode="group")
    fig.update_xaxes(title_text="Indeterminacy Level (δ)", row=1, col=1)
    fig.update_xaxes(title_text="Effect Size", row=1, col=2)
    fig.update_yaxes(title_text="Decision Stability", row=1, col=1, range=[0,1.1])
    fig.update_yaxes(title_text="Decision Stability", row=1, col=2, range=[0,1.1])
    return fig


def plot_interval_width_comparison(df: pd.DataFrame) -> go.Figure:
    """P-value interval width: mean by indeterminacy + distribution box."""
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Mean Interval Width by Indeterminacy",
                        "Interval Width Distribution"),
        specs=[[{"type":"bar"},{"type":"box"}]])

    COLORS = {"original": "#2196F3", "modified": "#FF5722"}

    for variant in ["original","modified"]:
        vd = df[df["variant"]==variant]

        wb = vd.groupby("delta")["interval_width_mean"].mean().reset_index()
        fig.add_trace(go.Bar(
            x=wb["delta"], y=wb["interval_width_mean"],
            name=variant.title(), marker_color=COLORS[variant],
            text=[f"{v:.3f}" for v in wb["interval_width_mean"]],
            textposition="outside",
        ), row=1, col=1)

        fig.add_trace(go.Box(
            y=vd["interval_width_mean"], name=variant.title(),
            marker_color=COLORS[variant], boxmean="sd", showlegend=False,
        ), row=1, col=2)

    fig.update_layout(title="📏 P-value Interval Width Comparison",
                      height=460, showlegend=True,
                      template="plotly_white", barmode="group")
    fig.update_xaxes(title_text="Indeterminacy Level (δ)", row=1, col=1)
    fig.update_yaxes(title_text="Mean Interval Width", row=1, col=1)
    fig.update_yaxes(title_text="Interval Width", row=1, col=2)
    return fig


def plot_summary_metrics(df: pd.DataFrame) -> go.Figure:
    """Overall performance summary bar chart."""
    # Compute means carefully:
    # - power: only H1 rows  (effect > 0)
    # - type1_error: only H0 rows (effect == 0); NaN if none
    # - decision_stability, interval_width: all rows
    rows = []
    for variant in ["original", "modified"]:
        vd = df[df["variant"] == variant]
        h1 = vd[vd["effect_size"] > 0]
        h0 = vd[vd["effect_size"] == 0]
        rows.append({
            "variant":          variant,
            "power":            h1["power"].mean() if not h1.empty else 0.0,
            "type1_error":      h0["type1_error"].mean() if not h0.empty else np.nan,
            "decision_stability": vd["decision_stability"].mean(),
            "interval_width":   vd["interval_width_mean"].mean(),
        })
    summary = pd.DataFrame(rows)

    metrics = ["power", "type1_error", "decision_stability", "interval_width"]
    labels  = ["Power", "Type I Error", "Decision Stability", "Interval Width"]
    COLORS  = ["#2196F3", "#FF5722"]

    fig = go.Figure()
    for i, variant in enumerate(["original", "modified"]):
        vrow = summary[summary["variant"] == variant].iloc[0]
        fig.add_trace(go.Bar(
            name=variant.title(), x=labels,
            y=[vrow[m] for m in metrics],
            marker_color=COLORS[i],
            text=[f"{vrow[m]:.3f}" if np.isfinite(vrow[m]) else "N/A" for m in metrics],
            textposition="auto",
        ))

    fig.update_layout(
        title="📊 Overall Performance Summary",
        yaxis_title="Metric Value", barmode="group",
        template="plotly_white", height=460,
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Grouped summary table.
    FIX: uses ddof=0 (population std) so single-group std is 0.0 not NaN.
    Also drops type1_error std for H1-only conditions via fillna(0).
    """
    agg = (
        df.groupby([
            "variant",
            "delta",
            "n",
            "distribution",
            "effect_size",
        ])
        .agg(

            power_mean=(
                "power",
                lambda x: np.nanmean(x)
            ),

            power_std=(
                "power",
                lambda x: np.nanstd(x, ddof=0)
            ),

            type1_error_mean=(
                "type1_error",
                lambda x: np.nanmean(x)
            ),

            type1_error_std=(
                "type1_error",
                lambda x: np.nanstd(x, ddof=0)
            ),

            decision_stability_mean=(
                "decision_stability",
                "mean",
            ),

            decision_stability_std=(
                "decision_stability",
                lambda x: np.std(x, ddof=0)
            ),

            interval_width_mean=(
                "interval_width_mean",
                "mean",
            ),

            interval_width_std=(
                "interval_width_mean",
                lambda x: np.std(x, ddof=0)
            ),

            relative_efficiency_mean=(
                "relative_efficiency",
                lambda x: np.nanmean(x)
            ),

        )
        .round(4)
        .reset_index()
    )

    # Replace NaN std with 0 where only one value existed
    std_cols = [c for c in agg.columns if c.endswith("_std")]
    agg[std_cols] = agg[std_cols].fillna(0.0)

    return agg


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

col_config, col_results = st.columns([3, 9])

with col_config:
    st.markdown("### 🎮 Simulation Control")

    run_sim = st.button(
        "🚀 Run Simulation",
        type="primary",
        use_container_width=True,
        disabled=not all_params_set,
    )

with col_results:
    # FIX (run_simulation flag): wrap execution in try/finally so the flag
    # is always cleared even if an exception occurs mid-simulation,
    # preventing an infinite re-run loop on every Streamlit rerender.
    if run_sim:
        st.session_state["run_simulation"] = True

    should_run = st.session_state.get("run_simulation", False)

    if should_run:
        try:
            sim_engine = MonteCarloSimulation(
                n_simulations=n_sims,
                random_seed=st.session_state.get("random_seed", 42),
            )

            progress_bar = st.progress(0)
            status_text  = st.empty()

            def cb_progress(current: int, total: int) -> None:
                pct = min(1.0, current / max(1, total))
                progress_bar.progress(pct)
                status_text.text(
                    f"Running simulation: condition {current} of {total}..."
                )

            with st.spinner("Executing Monte Carlo simulations..."):
                res_df = sim_engine.run(
                    test_type,
                    sample_sizes,
                    indet_levels,
                    distributions,
                    effect_sizes,
                    alpha=st.session_state.get("alpha", 0.05),
                    progress_callback=cb_progress,
                )

            # FIX: explicitly drive progress bar to 100% after loop ends
            # (simulation.py calls callback(current, total) at START of each
            # condition, so the final condition never calls (total, total))
            progress_bar.progress(1.0)
            status_text.text("Simulation complete.")

            # IMPROVEMENT: empty-df guard
            if res_df.empty:
                st.warning("⚠️ Simulation returned no results. "
                           "Check that test functions return valid output.")
            else:
                st.success("✅ Simulation Complete!")
                st.session_state[f"sim_results_{test_type}"] = res_df
                st.session_state["last_simulation_df"] = res_df
                st.session_state["last_test_type"]     = test_type

        except Exception as e:
            st.error(f"❌ Simulation failed: {e}")
        finally:
            # FIX: always clear the flag so we don't loop forever
            st.session_state["run_simulation"] = False

    # ── Display results ──────────────────────────────────────────────────────
    if f"sim_results_{test_type}" in st.session_state:
        df = st.session_state[f"sim_results_{test_type}"]

        # IMPROVEMENT: guard against completely empty DataFrame reaching charts
        if df.empty:
            st.info("No simulation results to display.")
            st.stop()

        tabs = st.tabs([
            "📊 Power Analysis",
            "🎯 Type I Error",
            "⚡ Efficiency & Stability",
            "📏 Interval Widths",
            "📋 Summary Tables",
            "📥 Export Data",
        ])

        with tabs[0]:
            st.plotly_chart(plot_power_comparison(df), use_container_width=True)

            # FIX: guard against division by zero when original power = 0
            h1 = df[df["effect_size"] > 0]
            if not h1.empty and set(["original","modified"]) <= set(h1["variant"].unique()):
                orig_p = h1[h1["variant"]=="original"]["power"].mean()
                mod_p  = h1[h1["variant"]=="modified"]["power"].mean()
                if orig_p > 0:
                    improvement = _safe_div(mod_p - orig_p, orig_p) * 100
                    st.metric("Average Power Improvement",
                              f"+{improvement:.1f}%", delta="modified vs original")
                else:
                    st.info("Original power is ~0% — relative improvement is undefined.")

        with tabs[1]:
            st.plotly_chart(plot_type1_comparison(df), use_container_width=True)

            h0 = df[df["effect_size"] == 0]
            if not h0.empty and "modified" in h0["variant"].unique():
                t1_mod = h0[h0["variant"]=="modified"]["type1_error"].mean()
                alpha_val = st.session_state.get("alpha", 0.05)
                if np.isfinite(t1_mod):
                    diff = (t1_mod - alpha_val) * 100
                    st.metric("Modified Test Type I Error", f"{t1_mod:.3f}",
                              delta=f"{diff:+.1f}% from α={alpha_val}", delta_color="off")
            else:
                # FIX: was silently hidden; now show explicit message
                st.info("ℹ️ Type I error is only measured when effect_size = 0 is included. "
                        "Add 0.0 to Effect Sizes to see this metric.")

        with tabs[2]:
            # FIX: give each chart its own full-width row instead of cramming
            # two complex charts side-by-side in a half-column context
            st.plotly_chart(plot_relative_efficiency_heatmap(df),
                            use_container_width=True)
            st.plotly_chart(plot_decision_stability_comparison(df),
                            use_container_width=True)

            h1_mod = df[(df["effect_size"] > 0) & (df["variant"]=="modified")]
            if not h1_mod.empty:
                avg_eff = h1_mod["relative_efficiency"].mean()
                if np.isfinite(avg_eff):
                    st.metric("Average Relative Efficiency", f"{avg_eff:.2f}",
                              delta="modified/original ratio")

        with tabs[3]:
            st.plotly_chart(plot_interval_width_comparison(df),
                            use_container_width=True)

            if set(["original","modified"]) <= set(df["variant"].unique()):
                orig_w = df[df["variant"]=="original"]["interval_width_mean"].mean()
                mod_w  = df[df["variant"]=="modified"]["interval_width_mean"].mean()
                reduction = _safe_div(mod_w - orig_w, orig_w) * 100
                st.metric("Average Interval Width Change", f"{reduction:+.1f}%",
                          delta="negative = narrower (better)")

        with tabs[4]:
            summary_df = create_summary_table(df)
            st.dataframe(summary_df, use_container_width=True)
            
            # Add copyable CSV block
            with st.expander("📋 Copy Summary Table Data"):
                st.caption("Click the copy icon in the top right corner of the block below to copy the data to your clipboard.")
                st.code(summary_df.to_csv(index=False), language="csv")
                
            st.plotly_chart(plot_summary_metrics(df), use_container_width=True)

        with tabs[5]:
            col1, col2 = st.columns(2)
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            with col1:
                st.download_button(
                    "📥 Download Full Results (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"sim_results_{test_type}_{ts}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col2:
                st.download_button(
                    "📊 Download Summary Table (CSV)",
                    data=create_summary_table(df).to_csv(index=False).encode("utf-8"),
                    file_name=f"sim_summary_{test_type}_{ts}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            st.markdown("### Data Preview (first 100 rows)")
            st.dataframe(df.head(100), use_container_width=True)

    else:
        st.info("👈 Configure simulation parameters and click 'Run Simulation' to see results")

# ============================================================================
# FOOTER — FIX: use f-string so {n_sims} renders as its actual value
# ============================================================================

st.markdown("---")
st.markdown(
    f"""
<div style="text-align:center;padding:1rem;color:#666;">
  <p>© 2024–2026 Akua Agyapomah Oteng | University of Mines and Technology, Tarkwa, Ghana</p>
  <p>Monte Carlo simulations evaluate performance across sample sizes, indeterminacy levels,
     distributions, and effect sizes.</p>
  <p style="font-size:0.9em;">Each condition simulated with <strong>{n_sims}</strong>
     iterations for statistical stability.</p>
</div>
""",
    unsafe_allow_html=True,
)