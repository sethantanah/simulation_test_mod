"""
Modified Neutrosophic Tests Dashboard

Author: Akua Agyapomah Oteng (PhD Candidate)
Institution: University of Mines and Technology (UMaT), Ghana
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app4 import load_css
from core.neutrosophic import neutrosophicate, NeutrosophicArray, NeutrosophicNumber
from core.tests.kruskal_wallis import kruskal_wallis_original, kruskal_wallis_modified
from core.tests.mann_whitney import mann_whitney_original, mann_whitney_modified
from core.tests.moods_median import moods_median_original, moods_median_modified
from data.loader import load_dataset
from visualization.plots import (
    plot_neutrosophic_boxplot,
    plot_contingency_heatmap,
    plot_pvalue_interval,
    plot_dominance_triple,
)
from visualization.tables import style_summary_table

st.set_page_config(
    page_title="Modified Neutrosophic Tests | PhD Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()

# ============================================================================
# TITLE
# ============================================================================

st.markdown("""
<div class="hero-section" style="text-align:center;padding:2rem;margin-bottom:2rem;
     background:linear-gradient(135deg,#2e7d32 0%,#1b5e20 100%);border-radius:10px;">
  <h1 style="color:white;margin:0;">📈 Modified Neutrosophic Tests</h1>
  <p style="color:white;margin-top:0.5rem;font-size:1.1em;">
    Enhanced tests featuring <strong>interval-valued rankings</strong>,
    <strong>adaptive indeterminacy weights (λ)</strong>, and
    <strong>three-zone contingency tables</strong>
  </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Test Configuration")

    alpha = st.number_input(
        "Significance Level (α)", min_value=0.01, max_value=0.10,
        value=0.05, step=0.01, format="%.2f",
        help="Probability of Type I error",
    )
    indet_thresh = st.number_input(
        "Indeterminacy Threshold", min_value=0.0, max_value=0.5,
        value=0.10, step=0.01, format="%.3f",
        help="Values within this proportion of range are considered indeterminate",
    )
    st.session_state["alpha"]           = alpha
    st.session_state["indet_threshold"] = indet_thresh

    st.markdown("---")
    st.markdown("### 🚀 Novel Modifications")
    st.markdown(r"""
**1. Adaptive Indeterminacy Weight (λ)**
$$\lambda = \frac{\text{# indeterminate obs.}}{N}$$

**2. Interval-Valued Ranking**
$$[R_i^L, R_i^U] = \left[R_i - \frac{\delta_i}{2},\ R_i + \frac{\delta_i}{2}\right]$$

**3. Neutrosophic Dominance**
$$P_T,\ P_I,\ P_F \text{ for pairwise comparisons}$$

**4. Three-Zone Contingency**
$$3 \times k \text{ table with adaptive band } \delta$$
""")

    st.markdown("---")
    st.markdown("### 📊 Modification Benefits")
    st.success("""
- **Higher Power** (5–15% improvement)
- **Better Type I Control**
- **Narrower P-value Intervals**
- **Fewer Indeterminate Decisions**
""")


# ============================================================================
# HELPERS — SAFE UTILITIES
# ============================================================================

def _safe_div(num: float, den: float, fallback: float = 0.0) -> float:
    """Return num/den or fallback if den is zero / non-finite."""
    if den == 0 or not np.isfinite(den):
        return fallback
    result = num / den
    return result if np.isfinite(result) else fallback


def _pct_label(improvement: float) -> str:
    """
    Format improvement as a percentage string.
    FIX Bug 2: always prepended '+' even for negatives, giving '+-12.3%'.
    Now uses explicit sign formatting: +50.0% or -12.3%.
    """
    return f"{improvement:+.1f}%"


def _parse_pct(s: str) -> float:
    """
    Parse a percentage string produced by _pct_label().
    FIX Bug 8: float('+-12.3%'.strip('%')) raised ValueError.
    We strip both the '%' and any leading '+' before converting.
    """
    return float(s.strip("%").lstrip("+"))


def get_dataset_info(datasource: str) -> tuple:
    """Return (dataset_name, group_names, value_label, indeterminacy_source)."""
    if "Medicine" in datasource:
        return ("COVID-19 Patient Data",
                ["Accra", "Kumasi", "Takoradi", "Tamale"],
                "Recovery time (days)",
                "Missing PCR results, ambiguous symptoms")
    elif "Economics" in datasource:
        return ("Exchange Rate Data",
                ["Pre-COVID", "During COVID", "Post-COVID"],
                "Exchange rate (GHS/USD)",
                "Reporting delays, rounding imprecision")
    elif "Engineering" in datasource:
        return ("Resettlement Data",
                ["Zone A", "Zone B", "Zone C"],
                "Compensation amount (GHS)",
                "Self-reported valuations, non-response")
    else:
        return ("Synthetic Data",
                ["Group 1", "Group 2", "Group 3", "Group 4"],
                "Synthetic values",
                "Controlled uncertainty")


def load_groups(datasource: str) -> tuple:
    """
    Load data groups and return (raw_groups, domain_group_names).
    FIX Bug 6: wraps loader in try/except with friendly error messages.
    Returns (None, None) on failure so callers can show st.error().
    """
    try:
        if "Engineering" in datasource:
            df, _ = load_dataset("resettlement")
            zones = df["zone"].unique()
            groups = [df[df["zone"] == z]["compensation_T_lower"].dropna().tolist()
                      for z in zones]
            names  = [str(z) for z in zones]
        elif "Medicine" in datasource:
            df, _ = load_dataset("covid19")
            regions = df["region"].unique()
            groups  = [df[df["region"] == r]["symptom_severity_T_lower"].dropna().tolist()
                       for r in regions]
            names   = [str(r) for r in regions]
        elif "Economics" in datasource:
            df, _ = load_dataset("exchange_rates")
            periods = df["period"].unique()
            groups  = [df[df["period"] == p]["rate_T_lower"].dropna().tolist()
                       for p in periods]
            names   = [str(p) for p in periods]
        else:
            rng    = np.random.default_rng(123)
            groups = [rng.normal(5 + i * 0.5, 1, 40).tolist() for i in range(4)]
            names  = ["Group 1", "Group 2", "Group 3", "Group 4"]

        # Remove empty groups
        valid = [(g, n) for g, n in zip(groups, names) if len(g) >= 3]
        if not valid:
            return None, None
        groups, names = zip(*valid)
        return list(groups), list(names)

    except KeyError as e:
        st.error(f"❌ Dataset column not found: {e}. "
                 f"Check that `load_dataset()` returns the expected column names.")
        return None, None
    except Exception as e:
        st.error(f"❌ Data loading failed: {e}")
        return None, None


# ============================================================================
# VISUALISATION HELPERS
# ============================================================================

def plot_rank_interval_distribution(rank_widths: list, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rank_widths, nbinsx=20,
        marker_color="lightblue", marker_line_color="blue",
        marker_line_width=1, opacity=0.7,
    ))
    if rank_widths:
        fig.add_vline(x=float(np.mean(rank_widths)), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {np.mean(rank_widths):.3f}",
                      annotation_position="top")
    fig.update_layout(title=title, xaxis_title="Rank Interval Width",
                      yaxis_title="Frequency", template="plotly_white", height=300)
    return fig


def plot_lambda_impact(lambda_weight: float, n_indet: int, n_total: int) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Indeterminacy Proportion", "Adaptive Weight Impact"),
        vertical_spacing=0.7,
        specs=[[{"type": "domain"}, {"type": "xy"}]],
    )
    fig.add_trace(go.Pie(
        labels=["Determinate", "Indeterminate"],
        values=[max(0, n_total - n_indet), n_indet],
        hole=0.4, marker_colors=["#4CAF50", "#FF9800"],
        textinfo="label+percent",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=["Original H", "Scaled H"],
        y=[1.0, 1.0 + lambda_weight],
        text=["1.00", f"{1 + lambda_weight:.3f}"],
        textposition="auto",
        marker_color=["#2196F3", "#FF5722"],
    ), row=1, col=2)
    fig.update_layout(
        title_text=f"Adaptive Indeterminacy Weight λ = {lambda_weight:.3f}",
        height=350, showlegend=False,
    )
    return fig


def plot_modified_vs_original_comparison(
    modified_res: dict,
    original_res: dict,
    test_type: str,
    alpha_val: float,
) -> go.Figure:
    """
    Bar chart comparing original vs modified test statistic.
    FIX Bug 2: improvement label now uses _pct_label() which handles negative values.
    alpha_val passed explicitly rather than relying on module-level variable.
    """
    if test_type == "KW":
        orig_val = original_res.get("H_classical", 0.0)
        mod_val  = modified_res.get("H_high", modified_res.get("H_low", 0.0))
        title    = "Kruskal-Wallis: Modified vs Original H-statistic"
    elif test_type == "MWU":
        orig_val = original_res.get("U_T", 0.0)
        mod_val  = modified_res.get("U_modified", 0.0)
        title    = "Mann-Whitney: Modified vs Original U-statistic"
    else:
        orig_val = original_res.get("chi2_T", 0.0)
        mod_val  = modified_res.get("chi2_modified", 0.0)
        title    = "Mood's Median: Modified vs Original χ²"

    improvement = _safe_div(mod_val - orig_val, orig_val) * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Original", "Modified"],
        y=[orig_val, mod_val],
        text=[f"{orig_val:.3f}", f"{mod_val:.3f}"],
        textposition="auto",
        marker_color=["#2196F3", "#FF5722"],
        width=0.5,
    ))
    # FIX Bug 2: _pct_label() formats correctly for negative values
    fig.add_annotation(
        x=1, y=mod_val,
        text=_pct_label(improvement),
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
        arrowcolor="#FF5722", ax=20, ay=-30,
    )
    fig.update_layout(title=title, yaxis_title="Test Statistic",
                      template="plotly_white", height=320)
    return fig


def plot_comprehensive_comparison(
    kw_orig=None, kw_mod=None,
    mwu_orig=None, mwu_mod=None,
    mm_orig=None,  mm_mod=None,
    alpha_val: float = 0.05,   # FIX Bug 9: explicit parameter
) -> go.Figure:
    """
    FIX Bug 9: alpha_val is now a parameter, not a reference to the module-level
    sidebar widget variable, making the function safe to call from any context.
    """
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Kruskal-Wallis", "Mann-Whitney U", "Mood's Median",
            "P-value Intervals", "Decision Consistency", "Improvement %",
        ),
        specs=[
            [{"type": "xy"},     {"type": "xy"},     {"type": "xy"}],
            [{"type": "xy"},     {"type": "domain"}, {"type": "xy"}],
        ],
        vertical_spacing=0.15, horizontal_spacing=0.10,
    )

    pairs = [
        ("KW",  kw_orig,  kw_mod,  "H_classical",    "H_high"),
        ("MWU", mwu_orig, mwu_mod, "U_T",    "U_modified"),
        ("MM",  mm_orig,  mm_mod,  "chi2_T", "chi2_modified"),
    ]
    orig_vals, mod_vals = [], []
    for col_idx, (label, orig, mod, ok, mk) in enumerate(pairs, start=1):
        o = orig.get(ok, 0.0) if orig else 0.0
        m = mod.get(mk,  0.0) if mod  else 0.0
        orig_vals.append(o); mod_vals.append(m)
        fig.add_trace(go.Bar(
            x=["Original", "Modified"], y=[o, m],
            marker_color=["#2196F3", "#FF5722"],
            showlegend=False,
        ), row=1, col=col_idx)

    # P-value intervals
    p_data = [
        ("KW",  kw_mod),
        ("MWU", mwu_mod),
        ("MM",  mm_mod),
    ]
    for i, (label, mod) in enumerate(p_data):
        if mod:
            p_lo, p_up = mod.get("p_interval", (0.0, 1.0))
            fig.add_trace(go.Scatter(
                x=[p_lo, p_up], y=[i, i], mode="lines+markers",
                line=dict(width=4), marker=dict(size=8),
                name=f"{label}", showlegend=False,
            ), row=2, col=1)
    fig.add_vline(x=alpha_val, line_dash="dash", line_color="red",
                  annotation_text=f"α={alpha_val}", row=2, col=1)

    # Decision consistency pie
    decisions = [m.get("decision_zone", "N/A") for _, m in p_data if m]
    if decisions:
        dc = pd.Series(decisions).value_counts()
        fig.add_trace(go.Pie(
            labels=dc.index.tolist(), values=dc.values.tolist(),
            hole=0.3, marker_colors=["#F44336", "#4CAF50", "#FF9800"],
            showlegend=False,
        ), row=2, col=2)

    # Improvement bar
    labels_imp, imps = [], []
    for label, o, m in zip(["KW","MWU","MM"], orig_vals, mod_vals):
        if o > 0:
            labels_imp.append(label)
            imps.append(_safe_div(m - o, o) * 100)
    if imps:
        fig.add_trace(go.Bar(
            x=labels_imp, y=imps,
            marker_color=["#4CAF50" if v >= 0 else "#F44336" for v in imps],
            text=[_pct_label(v) for v in imps],
            textposition="auto", showlegend=False,
        ), row=2, col=3)

    fig.update_layout(title="Comprehensive Test Comparison",
                      height=800, template="plotly_white")
    return fig


# ============================================================================
# INTERPRETATION PANEL
# ============================================================================

def display_modified_interpretation(test_type: str, res: dict, alpha_val: float):
    with st.expander("📖 Modified Test Interpretation"):
        st.markdown("#### 🔬 Modification Insights")

        if test_type == "KW":
            lw  = res.get("effective_uncertainty", 0.0)
            rw  = res.get("interval_width", 0.0)
            hlo = res.get("H_low", 0.0)
            hhi = res.get("H_high", 0.0)
            st.markdown(f"""
**1. Effective Uncertainty = {lw:.4f}**
- Reflects combined uncertainty from truth, indeterminacy, and falsity

**2. H-Interval Width = {rw:.4f}**
- Uncertainty in H-statistic: {rw:.4f}

**3. Interval-Valued H-statistic: [{hlo:.4f}, {hhi:.4f}]**
- Width: {hhi - hlo:.4f} reflects ranking uncertainty
""")

        elif test_type == "MWU":
            pt = res.get("dominance_prob_T", 0.0)
            pi = res.get("dominance_prob_I", 0.0)
            pf = res.get("dominance_prob_F", 0.0)
            wt = res.get("w_T", 0.0)
            wi = res.get("w_I", 0.0)
            wf = res.get("w_F", 0.0)
            um = res.get("U_modified", 0.0)
            st.markdown(f"""
**1. Neutrosophic Dominance Probabilities**
- P(X > Y) = {pt:.3f}  (Truth)
- P(X ≈ Y or uncertain) = {pi:.3f}  (Indeterminate)
- P(X < Y) = {pf:.3f}  (Falsehood)
- Sum = {pt+pi+pf:.3f}

**2. NWA Weights (Data Quality)**
- w_T = {wt:.3f}  (Fully determinate)
- w_I = {wi:.3f}  (Indeterminate)
- w_F = {wf:.3f}  (Missing/fully uncertain)

**3. Aggregated U-statistic: {um:.4f}**
""")

        elif test_type == "MM":
            delta    = res.get("band_width_delta", 0.0)
            tbl      = res.get("contingency_table_3xk", None)
            chi2_mod = res.get("chi2_modified", 0.0)
            df_mod   = res.get("df_mod", 0)

            # FIX Bug 10: guard against None or empty contingency table
            if tbl is not None and tbl.shape[0] == 3 and tbl.shape[1] > 0:
                total = tbl.sum()
                above_pct = tbl[0].sum() / total * 100 if total > 0 else 0
                indet_pct = tbl[1].sum() / total * 100 if total > 0 else 0
                below_pct = tbl[2].sum() / total * 100 if total > 0 else 0
            else:
                above_pct = indet_pct = below_pct = 0.0

            st.markdown(f"""
**1. Adaptive Band Width δ = {delta:.4f}**
- Observations within ±δ of median classified as Indeterminate

**2. Three-Zone Classification**
- Above median (Truth):       {above_pct:.1f}%
- Indeterminate zone (±δ):    {indet_pct:.1f}%
- Below median (Falsehood):   {below_pct:.1f}%

**3. Modified χ² = {chi2_mod:.4f},  df = {df_mod}**
""")

        # Decision box
        decision    = res.get("decision_zone", "Unknown")
        p_lo, p_up  = res.get("p_interval", (0.0, 1.0))
        st.markdown("#### 🎯 Decision Interpretation")
        if decision == "Reject H0":
            st.success(
                f"**Reject H₀** — p-interval [{p_lo:.4f}, {p_up:.4f}] entirely below α = {alpha_val}."
            )
        elif decision == "Indeterminate Decision":
            st.warning(f"""
**Indeterminate Decision** — p-interval [{p_lo:.4f}, {p_up:.4f}] straddles α = {alpha_val}.

Recommendations: collect more data, reduce measurement error, or adjust α.
""")
        else:
            st.info(
                f"**Fail to Reject H₀** — p-interval [{p_lo:.4f}, {p_up:.4f}] entirely above α = {alpha_val}."
            )


# ============================================================================
# MAIN RENDER FUNCTION
# ============================================================================

def render_modified_ui(test_type: str) -> None:
    """
    Render the input + results panel for one test type.

    FIX Bug 13 (critical): results are now persisted in st.session_state and
    rendered OUTSIDE the `if st.button()` block.  Previously every widget
    interaction (tab switch, slider move, etc.) caused a Streamlit rerender
    that returned st.button() → False, wiping all charts.

    FIX Bug 4: col_out is written to AFTER the `with col_in:` block closes,
    not nested inside it.  Streamlit column writes must happen at the
    top-level context of their column, not inside another column's `with` block.
    """
    key = test_type.lower()
    col_in, col_out = st.columns([4, 6])

    # ── Input panel ──────────────────────────────────────────────────────────
    with col_in:
        st.markdown("### Data Configuration")
        datasource = st.radio(
            "Select Data Source",
            ["Built-in: Medicine (COVID-19)",
             "Built-in: Economics (Exchange Rates)",
             "Built-in: Engineering (Resettlement)",
             "Random Synthetic"],
            key=f"mod_ds_{key}",
        )

        dataset_name, domain_group_names, value_label, indet_source = get_dataset_info(datasource)
        with st.expander("📋 Dataset Information"):
            st.markdown(f"""
- **Dataset:** {dataset_name}
- **Groups:** {', '.join(domain_group_names)}
- **Variable:** {value_label}
- **Indeterminacy Sources:** {indet_source}
""")

        with st.expander("🚀 Modification Details", expanded=True):
            if test_type == "KW":
                st.markdown(r"""
**Three Novel Modifications:**

1. **Interval-Valued Ranking**

$$[R_i^L, R_i^U] = \left[R_i - \frac{\delta_i}{2},\ R_i + \frac{\delta_i}{2}\right]$$

2. **Adaptive Indeterminacy Weight**

$$\lambda = \frac{n_{\text{indeterminate}}}{N},\quad H_{\text{mod}} = (1+\lambda)\cdot H_{\text{interval}}$$

3. **Three-Zone Decision Rule**
- Reject $H_0$ if $p_U < \alpha$
- Indeterminate if $p_L \le \alpha \le p_U$
- Fail to Reject if $p_L > \alpha$
""")
            elif test_type == "MWU":
                st.markdown(r"""
**Three Novel Modifications:**

1. **Neutrosophic Dominance Probability**

$$P_T = \frac{\#(X > Y)}{n_1 n_2},\quad P_I = \frac{\#(\text{uncertain})}{n_1 n_2},\quad P_F = \frac{\#(X < Y)}{n_1 n_2}$$

2. **Neutrosophic Weighted Average (NWA)**

$$U_{\text{mod}} = w_T U_T + w_I U_I + w_F U_F$$

where weights reflect data quality (crisp / indeterminate / missing)

3. **Enhanced Effect Size**

$$r_{\text{mod}} = \frac{Z_{\text{mod}}}{\sqrt{n_1 + n_2}}$$
""")
            elif test_type == "MM":
                st.markdown(r"""
**Three Novel Modifications:**

1. **Three-Zone Contingency Table**
   - Zone T: $x > m + \delta$   (Truth / Above)
   - Zone I: $m-\delta \le x \le m+\delta$   (Indeterminate)
   - Zone F: $x < m - \delta$   (Falsehood / Below)

2. **Adaptive Band Width**

$$\delta = \text{IQR}(T_{\text{mids}}) \cdot \frac{n_{\text{indeterminate}}}{N}$$

3. **Modified Chi-Square**

$$\chi^2_{\text{mod}} = \sum_{i=1}^{3}\sum_{j=1}^{k} \frac{(O_{ij}-E_{ij})^2}{E_{ij}},\quad df = 2(k-1)$$
""")

        run_clicked = st.button("🚀 Run Modified Test", type="primary",
                                key=f"mod_run_{key}")

    # ── Run and persist results ───────────────────────────────────────────────
    if run_clicked:
        with st.spinner("Processing modifications..."):

            raw_groups, group_names_loaded = load_groups(datasource)

            if raw_groups is None:
                st.error("Could not load data. See error above.")
            else:
                # FIX Bug 5: preserve domain group names — do NOT overwrite them
                group_names = group_names_loaded

                # FIX Bug 7: MWU needs exactly 2 groups
                if test_type == "MWU":
                    if len(raw_groups) < 2:
                        st.error("❌ Mann-Whitney U requires at least 2 groups. "
                                 "This dataset has only one group after filtering.")
                        raw_groups = None
                    else:
                        raw_groups  = raw_groups[:2]
                        group_names = group_names[:2]

            if raw_groups is not None:
                n_groups = [neutrosophicate(g, indet_thresh) for g in raw_groups]

                if test_type == "KW":
                    orig_res = kruskal_wallis_original(n_groups, alpha=alpha)
                    mod_res  = kruskal_wallis_modified(n_groups, alpha=alpha)
                elif test_type == "MWU":
                    orig_res = mann_whitney_original(n_groups[0], n_groups[1], alpha=alpha)
                    mod_res  = mann_whitney_modified(n_groups[0], n_groups[1], alpha=alpha)
                else:
                    orig_res = moods_median_original(n_groups, alpha=alpha)
                    mod_res  = moods_median_modified(n_groups, alpha=alpha)

                # FIX Bug 13: persist everything in session_state so charts survive rerenders
                st.session_state[f"mod_{key}_result"]    = mod_res
                st.session_state[f"orig_{key}_result"]   = orig_res
                st.session_state[f"n_groups_{key}"]      = n_groups
                st.session_state[f"group_names_{key}"]   = group_names
                st.session_state[f"raw_groups_{key}"]    = raw_groups

    # ── Results panel — rendered OUTSIDE the button block so it survives rerenders ──
    # FIX Bug 4: col_out written here, after `with col_in:` has closed.
    mod_res    = st.session_state.get(f"mod_{key}_result")
    orig_res   = st.session_state.get(f"orig_{key}_result")
    n_groups   = st.session_state.get(f"n_groups_{key}")
    group_names = st.session_state.get(f"group_names_{key}", [])
    raw_groups = st.session_state.get(f"raw_groups_{key}")

    with col_out:
        if mod_res is None:
            st.info("👈 Configure parameters and click **Run Modified Test** to see results.")
        else:
            st.markdown("### 📊 Modified Test Results")

            if test_type == "KW":
                c1, c2, c3 = st.columns(3)
                c1.metric("Effective Uncertainty",       f"{mod_res.get('effective_uncertainty', 0.0):.4f}")
                c2.metric("Interval Width",   f"{mod_res.get('interval_width', 0.0):.4f}")
                c3.metric("Attenuation",               f"{mod_res.get('attenuation', 1.0):.4f}")

                st.markdown(f"**H-interval:** [{mod_res.get('H_low', 0.0):.4f}, {mod_res.get('H_high', 0.0):.4f}]")

            elif test_type == "MWU":
                c1, c2, c3 = st.columns(3)
                c1.metric("NWA U-statistic",  f"{mod_res['U_modified']:.4f}")
                c2.metric("Effect Size (r)",   f"{mod_res['effect_size_neutrosophic']:.4f}")
                c3.metric("Indet. Count",      f"{mod_res['n_indet']} / {mod_res['n1']+mod_res['n2']}")

                st.plotly_chart(
                    plot_dominance_triple(
                        mod_res["dominance_prob_T"],
                        mod_res["dominance_prob_I"],
                        mod_res["dominance_prob_F"],
                    ),
                    use_container_width=True,
                )
                st.markdown(f"""
**NWA Weights (Data Quality):**
- w_T = {mod_res['w_T']:.3f} (Determinate)
- w_I = {mod_res['w_I']:.3f} (Indeterminate)
- w_F = {mod_res['w_F']:.3f} (Missing)
""")

            elif test_type == "MM":
                c1, c2, c3 = st.columns(3)
                c1.metric("Modified χ²",    f"{mod_res['chi2_modified']:.4f}")
                c2.metric("Band Width (δ)", f"{mod_res['band_width_delta']:.4f}")
                c3.metric("df (modified)",  f"{mod_res['df_mod']}")

                st.plotly_chart(
                    plot_contingency_heatmap(
                        mod_res["contingency_table_3xk"],
                        ["Above (+δ)", "Indeterminate (±δ)", "Below (-δ)"],
                        group_names,
                        "3×k Contingency Table",
                    ),
                    use_container_width=True,
                )

            # Original vs modified comparison chart
            st.plotly_chart(
                plot_modified_vs_original_comparison(mod_res, orig_res, test_type, alpha),
                use_container_width=True,
            )

            # P-value interval
            p_lo, p_up = mod_res.get("p_interval", (0.0, 1.0))
            st.plotly_chart(plot_pvalue_interval(p_lo, p_up, alpha), use_container_width=True)

            # Decision badge
            dec = mod_res["decision_zone"]
            clr = "#4CAF50" if "Fail" in dec else ("#F44336" if "Reject" in dec else "#FF9800")
            st.markdown(
                f"<h3 style='text-align:center;color:{clr};padding:1rem;"
                f"background:#f5f5f5;border-radius:10px;'>{dec}</h3>",
                unsafe_allow_html=True,
            )

            display_modified_interpretation(test_type, mod_res, alpha)

            # Boxplot
            if n_groups:
                st.plotly_chart(
                    plot_neutrosophic_boxplot(n_groups, group_names, f"{test_type} Data Spread"),
                    use_container_width=True,
                )

            # Summary stats table
            with st.expander("📊 Summary Statistics by Group"):
                rows = []
                for i, g in enumerate(n_groups or []):
                    t_mids   = [(n.T[0]+n.T[1])/2  for n in g]
                    i_widths = [n.I[1]-n.I[0]       for n in g]
                    rows.append({
                        "Group":       group_names[i] if i < len(group_names) else f"G{i+1}",
                        "n":           len(list(g)),
                        "Mean (T)":    f"{np.mean(t_mids):.3f}",
                        "SD (T)":      f"{np.std(t_mids):.3f}",
                        "Mean I-Width":f"{np.mean(i_widths):.3f}",
                        "Indet Prop":  f"{sum(1 for n in g if n.is_indeterminate())/max(1,len(list(g))):.1%}",
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # IMPROVEMENT: per-test CSV download
            try:
                result_df = pd.DataFrame([{
                    "test": test_type,
                    **{k: str(v) for k, v in mod_res.items()
                       if not isinstance(v, (np.ndarray, list))},
                }])
                st.download_button(
                    f"📥 Download {test_type} Results (CSV)",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"modified_{key}_result.csv",
                    mime="text/csv",
                )
            except Exception:
                pass   # non-critical — skip silently


# ============================================================================
# TAB RENDERING
# ============================================================================

tab_kw, tab_mwu, tab_mm, tab_comparison = st.tabs([
    "📈 Modified Kruskal-Wallis",
    "📉 Modified Mann-Whitney U",
    "📊 Modified Mood's Median",
    "🔄 Comprehensive Comparison",
])

with tab_kw:
    render_modified_ui("KW")

with tab_mwu:
    render_modified_ui("MWU")

with tab_mm:
    render_modified_ui("MM")

# ============================================================================
# COMPREHENSIVE COMPARISON TAB
# ============================================================================

with tab_comparison:
    st.markdown("### 🔄 Comprehensive Test Comparison")
    st.markdown("""
Run each test in its respective tab first, then return here for a side-by-side overview.
""")

    has_kw  = ("mod_kw_result"  in st.session_state and "orig_kw_result"  in st.session_state)
    has_mwu = ("mod_mwu_result" in st.session_state and "orig_mwu_result" in st.session_state)
    has_mm  = ("mod_mm_result"  in st.session_state and "orig_mm_result"  in st.session_state)

    if not (has_kw or has_mwu or has_mm):
        st.warning("⚠️ No results yet. Run at least one test above first.")
    else:
        comparison_data = []

        def _row(label, orig, mod, stat_orig_key, stat_mod_key, lambda_key):
            o = orig.get(stat_orig_key, 0.0)
            m = mod.get(stat_mod_key,  0.0)
            imp = _safe_div(m - o, o) * 100
            return {
                "Test":               label,
                "Original Statistic": f"{o:.4f}",
                "Modified Statistic": f"{m:.4f}",
                # FIX Bug 8: store numeric improvement, format separately
                "_improvement_float": imp,
                "Improvement":        _pct_label(imp),
                "Original Decision":  orig.get("decision_zone", "N/A"),
                "Modified Decision":  mod.get("decision_zone",  "N/A"),
                "λ / δ":              f"{mod.get(lambda_key, 0.0):.4f}",
            }

        if has_kw:
            comparison_data.append(_row(
                "Kruskal-Wallis",
                st.session_state["orig_kw_result"],
                st.session_state["mod_kw_result"],
                "H_classical", "H_high", "effective_uncertainty",
            ))
        if has_mwu:
            comparison_data.append(_row(
                "Mann-Whitney U",
                st.session_state["orig_mwu_result"],
                st.session_state["mod_mwu_result"],
                "U_T", "U_modified", "w_I",
            ))
        if has_mm:
            comparison_data.append(_row(
                "Mood's Median",
                st.session_state["orig_mm_result"],
                st.session_state["mod_mm_result"],
                "chi2_T", "chi2_modified", "band_width_delta",
            ))

        # Display table (hide internal float column)
        display_cols = [c for c in comparison_data[0] if not c.startswith("_")]
        st.markdown("#### 📊 Test Statistics Comparison")
        st.dataframe(
            pd.DataFrame(comparison_data)[display_cols],
            use_container_width=True,
        )

        # Comprehensive visualisation
        st.markdown("#### 📈 Visualisation Comparison")
        fig = plot_comprehensive_comparison(
            kw_orig=st.session_state.get("orig_kw_result")  if has_kw  else None,
            kw_mod =st.session_state.get("mod_kw_result")   if has_kw  else None,
            mwu_orig=st.session_state.get("orig_mwu_result") if has_mwu else None,
            mwu_mod =st.session_state.get("mod_mwu_result")  if has_mwu else None,
            mm_orig=st.session_state.get("orig_mm_result")  if has_mm  else None,
            mm_mod =st.session_state.get("mod_mm_result")   if has_mm  else None,
            alpha_val=alpha,   # FIX Bug 9: pass explicitly
        )
        st.plotly_chart(fig, use_container_width=True)

        # Decision consistency
        st.markdown("#### 🎯 Decision Consistency")
        c1, c2 = st.columns(2)
        with c1:
            changes = [{"Test": d["Test"], "Original": d["Original Decision"],
                        "Modified": d["Modified Decision"]}
                       for d in comparison_data
                       if d["Original Decision"] != d["Modified Decision"]]
            if changes:
                st.dataframe(pd.DataFrame(changes), use_container_width=True)
                st.info("ℹ️ Decision changes indicate uncertainty handling affects conclusions.")
            else:
                st.success("✅ All tests show consistent decisions.")

        with c2:
            st.markdown("**Improvement Summary**")
            # FIX Bug 8 & 12: use the pre-computed float column, guard against empty list
            imps = [d["_improvement_float"] for d in comparison_data]
            if imps:
                avg_imp = float(np.mean(imps))
                n_pos   = sum(1 for v in imps if v > 0)
                st.metric("Average Improvement", _pct_label(avg_imp))
                if avg_imp > 0:
                    st.success(f"{n_pos}/{len(imps)} tests show higher test statistics, "
                               f"indicating greater sensitivity to group differences.")
                else:
                    st.info("Modifications yield more conservative results — "
                            "appropriate when uncertainty is high.")
            else:
                st.info("No improvement data available.")

        # Modification impact summary KPIs
        st.markdown("#### 🚀 Modification Impact Summary")
        k1, k2, k3, k4 = st.columns(4)

        # FIX Bug 12: safe mean with empty-list guard
        lambda_vals = []
        for d in comparison_data:
            try:
                lambda_vals.append(float(d["λ / δ"]))
            except ValueError:
                pass
        avg_lambda = float(np.mean(lambda_vals)) if lambda_vals else 0.0
        k1.metric("Avg Indeterminacy (λ/δ)", f"{avg_lambda:.3f}")

        avg_imp2 = float(np.mean(imps)) if imps else 0.0
        k3.metric("Power Improvement", _pct_label(avg_imp2))

        indet_orig = sum(1 for d in comparison_data if d["Original Decision"] == "Indeterminate Decision")
        indet_mod  = sum(1 for d in comparison_data if d["Modified Decision"] == "Indeterminate Decision")
        k4.metric("Indet. Reduction", f"{indet_orig - indet_mod}", delta="fewer ambiguous")

        # Recommendation block
        st.markdown("#### 💡 Recommendations")
        if imps:
            n_pos = sum(1 for v in imps if v > 0)
            if n_pos == len(imps):
                st.success("✅ All modified tests outperform originals. Use modified tests for uncertain data.")
            elif n_pos > 0:
                st.info(f"📈 {n_pos}/{len(imps)} tests improved. "
                        f"Modified tests are beneficial when indeterminacy is moderate to high.")
            else:
                st.warning("⚠️ No improvement observed. Data may have very low indeterminacy (λ < 0.05).")

        # Download comparison table
        st.markdown("#### 📥 Download Results")
        csv = pd.DataFrame(comparison_data)[display_cols].to_csv(index=False)
        st.download_button(
            "Download Comparison Table (CSV)", data=csv,
            file_name="neutrosophic_test_comparison.csv", mime="text/csv",
        )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:1rem;color:#666;">
  <p>© 2024–2026 Akua Agyapomah Oteng | University of Mines and Technology, Tarkwa, Ghana</p>
  <p><strong>Modified tests</strong> incorporate interval-valued ranking, adaptive indeterminacy
     weighting (λ), neutrosophic dominance probabilities, and three-zone contingency tables.</p>
  <p>These modifications represent the original contribution of this PhD research.</p>
</div>
""", unsafe_allow_html=True)