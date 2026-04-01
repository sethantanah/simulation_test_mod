"""
Neutrosophic Non-Parametric Tests: Theory, Implementation, and Simulation Framework

This module provides a comprehensive implementation of original and modified 
neutrosophic non-parametric tests for the PhD research:

"MODIFICATION OF NEUTROSOPHIC NON-PARAMETRIC TESTS AND THEIR APPLICATION TO REAL-LIFE DATA"

Author: Akua Agyapomah Oteng
Institution: University of Mines and Technology (UMaT), Ghana
"""

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import load_css

st.set_page_config(
    page_title="Neutrosophic Non-Parametric Tests | PhD Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()


# ============================================================================
# THEORETICAL BACKGROUND SECTION
# ============================================================================

st.markdown("""
<div class="hero-section" style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);">
    <h1 style="color: white; font-size: 2.8em; margin-bottom: 0;">🔬 Modification of Neutrosophic Non-Parametric Tests</h1>
    <h3 style="color: #E3F2FD; font-weight: 300; margin-top: 1rem;">Theory, Methodology, and Applications</h3>
    <p style="color: white; margin-top: 0.5rem; font-size: 1.1em;">
        Akua Agyapomah Oteng — PhD Candidate<br>
        University of Mines and Technology (UMaT), Department of Mathematical Sciences
    </p>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# SECTION 1: THEORETICAL FOUNDATIONS
# ----------------------------------------------------------------------------

with st.container():
    st.markdown("## 📚 1. Theoretical Foundations")
    
    # Neutrosophic Numbers
    with st.expander("🔢 Neutrosophic Numbers: Definition and Properties", expanded=True):
        st.markdown(r"""
        ### Definition

        A **neutrosophic number** is a mathematical object that simultaneously represents three distinct components:

        $$
        N(T, I, F) = ([T_L, T_U], [I_L, I_U], [F_L, F_U])
        $$

        where:
        - **$T = [T_L, T_U]$**: Truth component — degree of membership (0 to 1)
        - **$I = [I_L, I_U]$**: Indeterminacy component — degree of uncertainty
        - **$F = [F_L, F_U]$**: Falsehood component — degree of non-membership (0 to 1)

        ### Properties

        1. **Normalization Constraint**: $T_L + I_L + F_L \le 3$ and $T_U + I_U + F_U \le 3$  
        2. **Crisp Special Case**: When $I = [0,0]$, the neutrosophic number reduces to a fuzzy number  
        3. **Classical Special Case**: When $I = [0,0]$ and $T = [x,x]$, $F = [1-x,1-x]$, we recover classical crisp values  

        ### Neutrosophication of Crisp Data

        For a crisp value $x$ in the range $[\min, \max]$:

        $$
        \begin{aligned}
        T &= \left[\frac{x - \min}{\max - \min},\ \frac{x - \min}{\max - \min}\right] \\
        [10pt]
        I &= \begin{cases}
        [0, \delta] & \text{if } x \text{ is near boundary} \\
        [0,0] & \text{otherwise}
        \end{cases} \\
        [10pt]
        F &= \left[1 - \frac{x - \min}{\max - \min},\ 1 - \frac{x - \min}{\max - \min}\right]
        \end{aligned}
        $$

        where $\delta$ is the **indeterminacy band width**.
        """)
        
        # Example visualization
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Example: Neutrosophic Number for x = 0.3**")
            st.latex(r"N = ([0.3, 0.3], [0, 0.1], [0.7, 0.7])")
            st.markdown("This indicates: 30% truth, 10% indeterminacy, 70% falsehood")
        
        with col2:
            st.markdown("**Example: Missing/Indeterminate Data**")
            st.latex(r"N_{\text{missing}} = ([0, 0], [0, 1], [0, 0])")
            st.markdown("Fully indeterminate — no information about truth or falsehood")
    
    # Neutrosophic Hypothesis Testing Framework
    with st.expander("🎯 Neutrosophic Hypothesis Testing Framework", expanded=True):
        st.markdown(r"""
        ### Three-Outcome Decision Framework

        Unlike classical hypothesis testing which yields only two outcomes (Reject/Fail to Reject), neutrosophic testing introduces a **third outcome**:

        $$
        \text{Decision} =
        \begin{cases}
        \text{Reject } H_0 & \text{if } p_U < \alpha \\
        \text{Indeterminate} & \text{if } p_L \le \alpha \le p_U \\
        \text{Fail to Reject } H_0 & \text{if } p_L > \alpha
        \end{cases}
        $$

        where $p_L$ and $p_U$ are the lower and upper bounds of the neutrosophic p-value interval $[p_L, p_U]$.

        ### Interpretation

        - **Reject $H_0$**: Strong evidence against null hypothesis (all components agree)  
        - **Indeterminate**: Inconclusive — uncertainty prevents clear decision  
        - **Fail to Reject $H_0$**: Insufficient evidence (all components agree)  

        This framework explicitly acknowledges when data uncertainty leads to ambiguous conclusions.
        """)

# ----------------------------------------------------------------------------
# SECTION 2: ORIGINAL NEUTROSOPHIC TESTS
# ----------------------------------------------------------------------------

st.markdown("---")
st.markdown("## 📊 2. Original Neutrosophic Non-Parametric Tests")

# Test 1: Kruskal-Wallis
with st.expander("📈 Neutrosophic Kruskal-Wallis Test (Original Formulation)", expanded=False):
    st.markdown("""
    ### Test Purpose
    Compare $k \\ge 2$ independent groups to determine if they come from the same distribution.
    
    ### Mathematical Formulation
    
    For each component $c \\in \\{T, I, F\\}$:
    
    $$\\boxed{H_c = \\frac{12}{N_c(N_c+1)}\\sum_{i=1}^k \\frac{R_{ci}^2}{n_i} - 3(N_c+1)}$$
    
    where:
    - $R_{ci} = \\sum_{j=1}^{n_i} r_{cij}$ = sum of ranks for group $i$ based on component $c$
    - $n_i$ = sample size of group $i$
    - $N_c = \\sum_{i=1}^k n_i$ = total sample size for component $c$
    
    ### Neutrosophic Test Statistic
    
    $$H_N = \\left( [H_L, H_U], H_I, [H_L, H_U] \\right)$$
    
    where:
    - $H_L = \\min(H_T, H_I, H_F)$
    - $H_U = \\max(H_T, H_I, H_F)$
    
    ### P-value Interval
    
    Under $H_0$, $H_c \\sim \\chi^2(k-1)$:
    
    $$p_c = P(\\chi^2_{k-1} > H_c)$$
    
    $$[p_L, p_U] = [\\min(p_T, p_I, p_F),\\ \\max(p_T, p_I, p_F)]$$
    
    ### Decision Rule
    
    Apply the three-outcome decision framework using $[p_L, p_U]$ and significance level $\\alpha$.
    """)
    
    st.code("""
    # Example usage
    from core.tests.kruskal_wallis import kruskal_wallis_original
    
    result = kruskal_wallis_original(groups, alpha=0.05)
    print(f"H-statistic: {result['H_N']}")
    print(f"p-value interval: {result['p_interval']}")
    print(f"Decision: {result['overall_decision']}")
    """, language="python")

# Test 2: Mann-Whitney U
with st.expander("📉 Neutrosophic Mann-Whitney U Test (Original Formulation)", expanded=False):
    st.markdown("""
    ### Test Purpose
    Compare two independent groups to determine if one stochastically dominates the other.
    
    ### Mathematical Formulation
    
    For each component $c \\in \\{T, I, F\\}$:
    
    $$\\boxed{U_c = \\min(U_{c1}, U_{c2})}$$
    
    where:
    - $U_{c1} = n_1 n_2 + \\frac{n_1(n_1+1)}{2} - R_{c1}$
    - $U_{c2} = n_1 n_2 - U_{c1}$
    - $R_{c1}$ = sum of ranks for group 1 based on component $c$
    
    ### Normal Approximation
    
    For $n_1, n_2 \\ge 8$:
    
    $$Z_c = \\frac{U_c - \\mu_{U}}{\\sigma_{U}}, \\quad \\mu_U = \\frac{n_1 n_2}{2}, \\quad \\sigma_U = \\sqrt{\\frac{n_1 n_2(n_1+n_2+1)}{12}}$$
    
    $$p_c = 2 \\cdot P(Z > |Z_c|)$$
    
    ### Neutrosophic Test Statistic
    
    $$U_N = \\left( [U_L, U_U], U_I, [U_L, U_U] \\right)$$
    
    where $U_L = \\min(U_T, U_I, U_F)$, $U_U = \\max(U_T, U_I, U_F)$
    """)
    
    st.code("""
    # Example usage
    from core.tests.mann_whitney import mann_whitney_original
    
    result = mann_whitney_original(group1, group2, alpha=0.05)
    print(f"U-statistic: {result['U_N']}")
    print(f"p-value interval: {result['p_interval']}")
    """, language="python")

# Test 3: Mood's Median
with st.expander("📊 Neutrosophic Mood's Median Test (Original Formulation)", expanded=False):
    st.markdown("""
    ### Test Purpose
    Compare medians of $k \\ge 2$ independent groups using a contingency table approach.
    
    ### Mathematical Formulation
    
    For each component $c \\in \\{T, I, F\\}$:
    
    1. Compute grand median $m_c$ of all $N_c$ observations
    2. Construct $2 \\times k$ contingency table:
       - Row 1: Observations above $m_c$
       - Row 2: Observations at or below $m_c$
    
    $$\\boxed{\\chi^2_c = \\sum_{i=1}^2 \\sum_{j=1}^k \\frac{(O_{ij} - E_{ij})^2}{E_{ij}}}$$
    
    where $E_{ij} = \\frac{(\\text{row } i \\text{ total}) \\times (\\text{col } j \\text{ total})}{N_c}$
    
    ### Neutrosophic Test Statistic
    
    $$\\chi^2_N = \\left( [\\chi^2_L, \\chi^2_U], \\chi^2_I, [\\chi^2_L, \\chi^2_U] \\right)$$
    
    where $\\chi^2_L = \\min(\\chi^2_T, \\chi^2_I, \\chi^2_F)$, $\\chi^2_U = \\max(\\chi^2_T, \\chi^2_I, \\chi^2_F)$
    
    ### Degrees of Freedom
    
    $$df = k - 1$$
    
    ### P-value Interval
    
    $$p_c = P(\\chi^2_{df} > \\chi^2_c)$$
    $$[p_L, p_U] = [\\min(p_T, p_I, p_F),\\ \\max(p_T, p_I, p_F)]$$
    """)

# ----------------------------------------------------------------------------
# SECTION 3: MODIFIED TESTS (PhD CONTRIBUTION)
# ----------------------------------------------------------------------------

st.markdown("---")
st.markdown("## 🚀 3. Modified Neutrosophic Tests (Original Contributions)")

# Modification 1: Enhanced Kruskal-Wallis
with st.expander("🎯 Modified Neutrosophic Kruskal-Wallis Test", expanded=True):
    st.markdown(r"""
    ### Three Novel Modifications

    #### Modification 1: Interval-Valued Neutrosophic Ranking

    Traditional ranking assigns a single rank $R_i$ to each observation. Our modification creates a **rank interval** based on indeterminacy width:

    $$
    [R_i^L, R_i^U] = \left[R_i - \frac{\delta_i}{2},\ R_i + \frac{\delta_i}{2}\right]
    $$

    where $\delta_i = I_{iU} - I_{iL}$ is the width of the indeterminacy interval for observation $i$.

    This ensures that observations with higher uncertainty contribute less precision to the ranking structure.

    #### Modification 2: Adaptive Indeterminacy Weighting ($\lambda$)

    We introduce an adaptive weight that scales the test statistic based on overall data indeterminacy:

    $$
    \lambda = \frac{\text{\# indeterminate observations}}{N}
    $$

    where an observation is considered indeterminate if $\delta_i > 0.01$.

    The modified H-statistic becomes:

    $$
    H_{\text{mod}} = (1 + \lambda) \cdot H_{\text{interval}}
    $$

    where $H_{\text{interval}}$ is computed using the rank intervals $[R_i^L, R_i^U]$.

    #### Modification 3: Enhanced Three-Zone Decision

    We maintain the three-outcome decision framework but with improved sensitivity through the adaptive weighting mechanism.

    ### Combined Formulation

    $$
    H_{\text{mod}} =
    (1 + \lambda)\cdot
    \frac{12}{N(N+1)}
    \sum_{i=1}^{k}
    \frac{[R_i^L, R_i^U]^2}{n_i}
    - 3(N+1)
    $$

    where $[R_i^L, R_i^U]^2$ is interval arithmetic:
    - Lower bound: $(R_i^L)^2$
    - Upper bound: $(R_i^U)^2$

    ### Theoretical Properties

    1. **Classical Reduction**: When all $\delta_i = 0$ (crisp data), $H_{\text{mod}}$ reduces to the classical Kruskal-Wallis H-statistic  
    2. **Monotonicity**: $H_{\text{mod}}$ increases with $\lambda$, reflecting greater influence of indeterminacy  
    3. **Boundedness**: $H_{\text{mod}} \in [0, N(1+\lambda)]$ for proper normalization
    """)
        
    st.info("""
    **💡 Intuition**: The modified test upweights the test statistic when data contains more determinate observations, 
    making it more sensitive to true differences while properly accounting for uncertainty through interval-valued ranks.
    """)

# Modification 2: Enhanced Mann-Whitney
with st.expander("⚖️ Modified Neutrosophic Mann-Whitney U Test", expanded=False):
    st.markdown(r"""
    ### Three Novel Modifications

    #### Modification 1: Neutrosophic Dominance Probability

    Instead of simply taking $\min(U_1, U_2)$, we compute dominance probabilities:

    For all pairs $(x_i, y_j)$ where $x_i \in \text{Group1}$, $y_j \in \text{Group2}$:

    $$
    P_T = \frac{\#\{(i,j): x_i > y_j \text{ and neither is indeterminate}\}}{n_1 n_2}
    $$

    $$
    P_I = \frac{\#\{(i,j): x_i \text{ or } y_j \text{ is indeterminate or } x_i = y_j\}}{n_1 n_2}
    $$

    $$
    P_F = \frac{\#\{(i,j): x_i < y_j \text{ and neither is indeterminate}\}}{n_1 n_2}
    $$

    This provides a comprehensive picture of group differences beyond simple stochastic dominance.

    #### Modification 2: Neutrosophic Weighted Average (NWA) U-Statistic

    We aggregate component U-statistics using **data-quality weights**:

    $$
    U_{\text{mod}} = w_T \cdot U_T + w_I \cdot U_I + w_F \cdot U_F
    $$

    where:

    $$
    w_T = \frac{\text{\# fully determinate observations}}{N_{\text{total}}}
    $$

    $$
    w_I = \frac{\text{\# indeterminate observations}}{N_{\text{total}}}
    $$

    $$
    w_F = \frac{\text{\# missing observations}}{N_{\text{total}}}
    $$

    These weights reflect the **quality** of the data, not the outcome of the test.

    #### Modification 3: Enhanced Effect Size

    $$
    r_{\text{mod}} = \frac{Z_{\text{mod}}}{\sqrt{n_1 + n_2}}
    $$

    where:

    $$
    Z_{\text{mod}} = \frac{U_{\text{mod}} - \mu_U}{\sigma_U}
    $$

    ### Combined Formulation

    $$
    U_{\text{mod}} =
    \frac{
    n_{\text{crisp}} \cdot U_T +
    n_{\text{indet}} \cdot U_I +
    n_{\text{missing}} \cdot U_F
    }{
    n_{\text{crisp}} + n_{\text{indet}} + n_{\text{missing}}
    }
    $$
    """, unsafe_allow_html=True)
    st.warning("""
    **⚠️ Important**: The NWA weights are based on **data quality**, not test outcomes. 
    This avoids circularity and ensures the test statistic reflects genuine data characteristics.
    """)

# Modification 3: Enhanced Mood's Median
with st.expander("📐 Modified Neutrosophic Mood's Median Test", expanded=False):
    st.markdown(r"""
    ### Three Novel Modifications

    #### Modification 1: Three-Zone Contingency Table ($3 \times k$)

    Instead of binary classification (above/below median), we introduce an **indeterminate zone**:

    Zone definitions:

    $$
    \text{Zone T (Truth)}: \quad x > m + \delta
    $$

    $$
    \text{Zone I (Indeterminate)}: \quad m - \delta \le x \le m + \delta
    $$

    $$
    \text{Zone F (Falsehood)}: \quad x < m - \delta
    $$

    where $\delta$ is an adaptive band width.

    #### Modification 2: Adaptive Band Width ($\delta$)

    $$
    \delta = \text{IQR}(T_{\text{mids}}) \times \frac{\text{\# indeterminate observations}}{N}
    $$

    This ensures:
    - Larger IQR → wider bands (more variability)
    - More indeterminacy → wider bands (more uncertainty)
    - Minimum $\delta = \text{IQR} \times 0.01$ to prevent degeneracy

    #### Modification 3: Modified Chi-Square Statistic

    $$
    \chi^2_{\text{mod}} =
    \sum_{i=1}^{3} \sum_{j=1}^{k}
    \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
    $$

    with degrees of freedom:

    $$
    df_{\text{mod}} = 2(k - 1)
    $$

    The additional $df$ accounts for the extra row in the contingency table.

    ### Combined Formulation

    The modified test uses:
    1. **$3 \times k$ contingency table** instead of $2 \times k$
    2. **Adaptive band width** $\delta$ based on data characteristics
    3. **Increased degrees of freedom** ($2(k-1)$ vs $k-1$)

    ### Theoretical Properties

    1. **Degeneracy Prevention**: $\delta_{\min}$ ensures the test works even with fully crisp data  
    2. **Scale Invariance**: $\delta$ scales with IQR, making it invariant to linear transformations  
    3. **Consistency**: When $\delta \to 0$, the test approximates the original Mood's median test
    """)

# ----------------------------------------------------------------------------
# SECTION 4: SIMULATION METHODOLOGY
# ----------------------------------------------------------------------------

st.markdown("---")
st.markdown("## 🔬 4. Simulation Methodology (Objective 3)")

with st.expander("📊 Experimental Design", expanded=True):
    st.markdown("""
    ### Monte Carlo Simulation Framework
    
    We conduct comprehensive simulations to compare original and modified tests across controlled conditions.
    
    #### Design Parameters
    
    | Parameter | Levels | Rationale |
    |-----------|--------|-----------|
    | **Sample Size ($n$)** | 20, 50, 100, 200 | Small to large samples |
    | **Indeterminacy Proportion ($\\delta$)** | 0.0, 0.1, 0.25, 0.4 | None to high uncertainty |
    | **Distributions** | Normal, Skewed, Heavy-tailed | Robustness testing |
    | **Effect Size ($d$)** | 0, 0.2, 0.5, 0.8, 1.0 | Null to large effects |
    | **Simulations per condition** | 1,000 | Stable estimates |
    
    #### Performance Metrics
    
    $$\\boxed{\\text{Power} = \\frac{\\# \\text{Reject } H_0 \\text{ when } H_1 \\text{ true}}{N_{\\text{sim}}}}$$
    
    $$\\boxed{\\text{Type I Error} = \\frac{\\# \\text{Reject } H_0 \\text{ when } H_0 \\text{ true}}{N_{\\text{sim}}}}$$
    
    $$\\boxed{\\text{Decision Stability} = 1 - \\frac{\\# \\text{Indeterminate}}{N_{\\text{sim}}}}$$
    
    $$\\boxed{\\text{Relative Efficiency (RE)} = \\frac{\\text{Power}_{\\text{modified}}}{\\text{Power}_{\\text{original}}}}$$
    
    #### Indeterminacy Induction
    
    We add controlled indeterminacy using four patterns:
    
    1. **Uniform**: $I = [0, 1]$ for selected observations (full uncertainty)
    2. **Partial**: $I = [0, 0.5]$ (moderate uncertainty)
    3. **Mixed**: $I$ varies randomly $\\in [0.2, 0.8]$
    4. **Data-dependent**: $I$ proportional to distance from center
    
    ### Simulation Algorithm
    
    ```python
    for each condition (n, δ, distribution, effect_size):
        results = []
        for simulation in range(1000):
            # Generate data with specified effect size
            data = generate_data(n, distribution, effect_size)
            
            # Convert to neutrosophic with controlled indeterminacy
            neutro_data = neutrosophicate(data, δ)
            
            # Apply original test
            orig_result = original_test(neutro_data)
            
            # Apply modified test
            mod_result = modified_test(neutro_data)
            
            # Store decisions and p-value intervals
            results.append([orig_result, mod_result])
        
        # Compute performance metrics
        power_original = proportion_rejections(orig_results, H1=True)
        power_modified = proportion_rejections(mod_results, H1=True)
        type1_original = proportion_rejections(orig_results, H0=True)
        type1_modified = proportion_rejections(mod_results, H0=True)
        """)

# Visualization of expected results
with st.expander("📈 Expected Results Visualization", expanded=False):
    st.markdown("""

Anticipated Performance Characteristics
Power Comparison by Indeterminacy Level
""")

# --------------------------------------------------------------------------
# SECTION 5: REAL-WORLD APPLICATIONS
# --------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 🌍 5. Real-World Applications (Objective 2)")

with st.expander("🏥 Medical Application: COVID-19 Data", expanded=False):
    st.markdown("""
    **Dataset:** Ghana COVID-19 Case Data  
    **Source:** Ghana Health Service (2020-2023)

    **Indeterminacy Sources:**
    - Missing PCR test results (15-20% of cases)
    - Ambiguous symptom reports
    - Incomplete contact tracing data

    **Research Questions:**
    - Are there significant differences in recovery times across regions?
    - Does vaccination status affect symptom severity?
    - How does indeterminacy in testing affect these conclusions?

    **Expected Contribution:**  
    Demonstrate that modified tests can handle real-world medical data with inherent uncertainty.
    """)

with st.expander("💰 Economic Application: Exchange Rate Analysis", expanded=False):
    st.markdown("""
    **Dataset:** Ghana Cedi (GHS) vs USD Exchange Rates  
    **Source:** Bank of Ghana (2015-2024)

    **Indeterminacy Sources:**
    - Reporting delays (2-5 day gaps)
    - Rounding imprecision in reported rates
    - Market volatility during economic shocks

    **Research Questions:**
    - Do exchange rate distributions differ across pre- and post-COVID periods?
    - Is there seasonal variation in exchange rate volatility?
    - How does reporting uncertainty affect trend detection?

    **Expected Contribution:**  
    Show how neutrosophic tests can detect subtle patterns masked by data uncertainty.
    """)

with st.expander("🏗️ Engineering Application: Resettlement Data", expanded=False):
    st.markdown("""
    **Dataset:** Tarkwa Mining Resettlement Survey  
    **Source:** University of Mines and Technology (UMaT)

    **Indeterminacy Sources:**
    - Self-reported land valuations
    - Survey non-response (10-15%)
    - Disputed compensation claims

    **Research Questions:**
    - Are compensation amounts fairly distributed across affected communities?
    - Does satisfaction differ by demographic factors?
    - How does non-response bias affect conclusions?

    **Expected Contribution:**  
    Provide actionable insights for policy decisions under uncertainty.
    """)

# --------------------------------------------------------------------------
# SECTION 6: SOFTWARE IMPLEMENTATION
# --------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 💻 6. Software Implementation")

with st.expander("🐍 Python Package Structure", expanded=False):
    st.markdown("""
    ### Package Architecture
    ```
    neutrosophic_tests/
    ├── core/
    │   ├── neutrosophic.py
    │   ├── neutrosophication.py
    │   └── utils.py
    ├── tests/
    │   ├── kruskal_wallis.py
    │   ├── mann_whitney.py
    │   └── moods_median.py
    ├── simulation/
    │   └── monte_carlo.py
    └── applications/
        ├── covid_analysis.py
        ├── exchange_rates.py
        └── resettlement.py
    ```

    ### Key Class Example
    ```python
    class NeutrosophicNumber:
        def __init__(self, T, I, F):
            self.T = T
            self.I = I
            self.F = F

        def score(self):
            return (self.T[0] + self.T[1]) / 2 - (self.F[0] + self.F[1]) / 2

        def defuzzify(self):
            t_m = sum(self.T) / 2
            i_m = sum(self.I) / 2
            f_m = sum(self.F) / 2
            return (2 + t_m - i_m - f_m) / 3
    ```
    """)

# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2>🔬 Research Navigator</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📍 Research Progress")

    milestones = [
        "✅ Literature Review",
        "✅ Theoretical Framework",
        "✅ Test Modifications",
        "🔄 Simulation Studies",
        "⏳ Case Study 1: COVID-19",
        "⏳ Case Study 2: Exchange Rates",
        "⏳ Case Study 3: Resettlement",
        "⏳ Thesis Writing"
    ]

    for milestone in milestones:
        if "✅" in milestone:
            st.markdown(f"<span style='color: #2ecc71;'>✓</span> {milestone}", unsafe_allow_html=True)
        elif "🔄" in milestone:
            st.markdown(f"<span style='color: #f39c12;'>⟳</span> {milestone}", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color: #95a5a6;'>○</span> {milestone}", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🔗 Quick Links")
    st.markdown("""
    - Theoretical Foundations  
    - Original Tests  
    - Modified Tests  
    - Simulation Methodology  
    - Real-World Applications  
    - Software Implementation  
    """)

    st.markdown("---")

    st.markdown("### 📧 Contact")
    st.markdown("""
    Akua Agyapomah Oteng  
    PhD Candidate  
    University of Mines and Technology, Tarkwa  

    akua.oteng@umat.edu.gh
    """)

    st.markdown("---")

    st.markdown("""
    <div style="text-align: center; font-size: 0.8em; color: #95a5a6;">
        Version 2.0 | March 2026
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>© 2024-2026 Akua Agyapomah Oteng</p>
    <p>University of Mines and Technology, Tarkwa, Ghana</p>
</div>
""", unsafe_allow_html=True)
