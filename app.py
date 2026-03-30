import streamlit as st
import os

st.set_page_config(
    page_title="NeutroStat — Neutrosophic Statistical Analysis Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

# Session State Initialization
if 'alpha' not in st.session_state: st.session_state.alpha = 0.05
if 'indet_threshold' not in st.session_state: st.session_state.indet_threshold = 0.10
if 'n_sims' not in st.session_state: st.session_state.n_sims = 1000
if 'random_seed' not in st.session_state: st.session_state.random_seed = 42

with st.sidebar:
    st.markdown('''
        <div class="sidebar-brand">
            <h2>NeutroStat 🔬</h2>
            <p style="margin: 0;">PhD Research Tool | UMaT | 2025</p>
        </div>
    ''', unsafe_allow_html=True)
    
    with st.expander("⚙️ Global Settings", expanded=True):
        st.session_state.alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)
        st.session_state.indet_threshold = st.slider("Indeterminacy threshold (δ)", 0.0, 0.5, 0.10, 0.01)
        st.session_state.n_sims = st.selectbox("Monte Carlo Simulations", [100, 500, 1000, 5000], index=2)
        st.session_state.random_seed = st.number_input("Random Seed", value=42, step=1)

# Main Page Welcome
st.markdown('''
<div class="hero-section">
    <h1 style="color: white; font-size: 3em; margin-bottom: 0;">Welcome to NeutroStat 🔬</h1>
    <h3 style="color: #E3F2FD; font-weight: 300;">This tool implements the complete research study:</h3>
    <h2 style="color: white; font-style: italic;">"Modification of Neutrosophic Non-Parametric Tests and Their Application to Real-Life Data"</h2>
</div>
''', unsafe_allow_html=True)

st.markdown("""
### Getting Started
Navigate using the sidebar to explore theory, run tests, simulate performance, and export your research report.

#### Key Features:
1. **Interactive Theory:** Explore Neutrosophic Numbers and interval arithmetic.
2. **Standard & Modified Tests:** Run Kruskal-Wallis, Mann-Whitney U, and Mood's Median tests under neutrosophic uncertainty.
3. **Monte Carlo Engine:** Simulate statistical power, type I error, and decision stability across conditions.
4. **Real-Life Applications:** Apply tests to medicine, economics, and engineering datasets.
5. **Research PDF Export:** Fully auto-generate an academic report.
""")

"""
Run with:
    streamlit run app.py
    
Development mode:
    streamlit run app.py --server.runOnSave true
    
Requirements:
    pip install -r requirements.txt
"""
