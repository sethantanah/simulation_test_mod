import streamlit as st

st.set_page_config(page_title="Introduction", page_icon="🔬", layout="wide")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import load_css
load_css()

# Hero Banner
st.markdown('''
<div class="hero-section" style="text-align: center; padding: 4rem 2rem;">
    <h1 style="color: white; font-size: 3em; margin-bottom: 0;">MODIFICATION OF NEUTROSOPHIC NON-PARAMETRIC TESTS AND THEIR APPLICATION TO REAL-LIFE DATA</h1>
    <h3 style="color: #E3F2FD; font-weight: 300; margin-top: 1rem;">Akua Agyapomah Oteng — PhD Candidate</h3>
    <p style="color: white; margin-top: 0.5rem;">University of Mines and Technology (UMaT)<br>Department of Mathematical Sciences</p>
</div>
''', unsafe_allow_html=True)

# Context Cards
st.markdown("### Research Context")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('''
    <div class="metric-card">
        <h4 style="color: #1565C0; margin-top: 0;">The Problem</h4>
        <p>Traditional non-parametric tests implicitly assume precise, crisp data. When real-world data contain uncertainty, indeterminacy, and missing values, these conventional tools often fail or discard valuable information leading to biased statistical conclusions.</p>
    </div>
    ''', unsafe_allow_html=True)
    
with col2:
    st.markdown('''
    <div class="metric-card">
        <h4 style="color: #1565C0; margin-top: 0;">The Solution</h4>
        <p>Neutrosophic statistics extends classical theory by explicitly modelling truth (T), indeterminacy (I), and falsehood (F). By treating indeterminacy as an independent component, we can perform robust statistical testing without discarding uncertain data points.</p>
    </div>
    ''', unsafe_allow_html=True)
    
with col3:
    st.markdown('''
    <div class="metric-card">
        <h4 style="color: #1565C0; margin-top: 0;">This Study</h4>
        <p>We propose <strong>novel modifications</strong> to the neutrosophic Kruskal-Wallis, Mann-Whitney U, and Mood's Median tests, incorporating interval-valued rankings, dominance probabilities, and adaptive indeterminacy bounds to significantly improve statistical power.</p>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")

with st.expander("📚 Neutrosophic Theory Primer", expanded=True):
    st.markdown("""
    #### What is a Neutrosophic Number?
    A neutrosophic number is represented as a triple **$N(T, I, F)$** where:
    - **T** is the degree of Truth / Membership
    - **I** is the degree of Indeterminacy / Uncertainty
    - **F** is the degree of Falsehood / Non-membership
    
    In this research, each component is treated as an interval $[L, U]$, e.g., $T = [T_L, T_U]$.
    """)
    st.latex(r"N([T_L, T_U], [I_L, I_U], [F_L, F_U])")
    
    st.markdown("""
    #### Core Test Equations:
    **Kruskal-Wallis (Original):**
    """)
    st.latex(r"H_N = \frac{12}{N_N(N_T+1)} \sum \left( \frac{R_{Ti}^2}{n_{Ti}}, \frac{R_{Ii}^2}{n_{Ii}}, \frac{R_{Fi}^2}{n_{Fi}} \right) - 3(N_T+1)")
    

st.markdown("### 🗺️ Study Roadmap")
# A simple mermaid flowchart for the roadmap
st.markdown("""
```mermaid
graph LR
    A[Data with Indeterminacy] --> B[Neutrosophication]
    B --> C[Statistical Testing]
    C --> D[Original Tests]
    C --> E[Modified Tests]
    D -.-> F[Comparison & Simulation]
    E -.-> F
    F --> G[Real-Life Applications]
```
""")

st.markdown("### 📖 References")
st.markdown("""
1. **Smarandache, F. (1998).** *Neutrosophy: Neutrosophic Probability, Set, and Logic*. ProQuest Information & Learning.
2. **Sherwani, R. A. K., et al. (2021).** Neutrosophic Kruskal-Wallis test: Application to COVID-19 data. *Journal of Mathematics*.
3. **Aslam, M. (2020).** A generic approach for the application of neutrosophic statistics. *Symmetry*.
""")

"""
Run with:
    streamlit run app.py
"""
