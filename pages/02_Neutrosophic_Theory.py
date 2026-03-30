import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import load_css
from core.neutrosophic import NeutrosophicNumber, neutrosophicate
import plotly.graph_objects as go

st.set_page_config(page_title="Neutrosophic Theory", layout="wide")
load_css()

st.title("🧩 Neutrosophic Theory Explorer")
st.markdown("Learn how neutrosophic numbers work through interactive examples.")

st.header("1. Neutrosophic Number Builder")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Define Interval Bounds")
    t_val = st.slider("Truth (T) Interval", 0.0, 1.0, (0.7, 0.9), 0.05)
    i_val = st.slider("Indeterminacy (I) Interval", 0.0, 1.0, (0.1, 0.3), 0.05)
    f_val = st.slider("Falsehood (F) Interval", 0.0, 1.0, (0.0, 0.2), 0.05)
    
    N = NeutrosophicNumber(t_val, i_val, f_val)

with col2:
    st.markdown("#### Properties")
    st.markdown(f"**Number Output:** `<span class='neutro-number'>{N}</span>`", unsafe_allow_html=True)
    
    st.metric("Score Function (T_mean - F_mean)", f"{N.score():.3f}")
    st.metric("Accuracy (T_mean + F_mean)", f"{N.accuracy():.3f}")
    st.metric("Defuzzified Crisp Value", f"{N.defuzzify():.3f}")
    
    # Radar chart for midpoints
    fig = go.Figure(go.Scatterpolar(
        r=[(t_val[0]+t_val[1])/2, (i_val[0]+i_val[1])/2, (f_val[0]+f_val[1])/2],
        theta=['Truth', 'Indeterminacy', 'Falsehood'],
        fill='toself',
        marker_color='#1565C0'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=300, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.header("2. Arithmetic Explorer")
colA, colB = st.columns(2)

with colA:
    st.markdown("#### Number A")
    A_t = st.slider("A: Truth", 0.0, 5.0, (1.0, 2.0), key="A_T")
    A_i = st.slider("A: Indet", 0.0, 5.0, (0.0, 0.5), key="A_I")
    A_f = st.slider("A: False", 0.0, 5.0, (0.0, 0.0), key="A_F")
    A = NeutrosophicNumber(A_t, A_i, A_f)
    
with colB:
    st.markdown("#### Number B")
    B_t = st.slider("B: Truth", 0.0, 5.0, (2.0, 3.0), key="B_T")
    B_i = st.slider("B: Indet", 0.0, 5.0, (0.1, 0.6), key="B_I")
    B_f = st.slider("B: False", 0.0, 5.0, (0.0, 0.0), key="B_F")
    B = NeutrosophicNumber(B_t, B_i, B_f)
    
st.markdown("#### Results")
r1, r2, r3, r4 = st.columns(4)
r1.markdown(f"**A + B**<br>`{A + B}`", unsafe_allow_html=True)
r2.markdown(f"**A - B**<br>`{A - B}`", unsafe_allow_html=True)
r3.markdown(f"**A × B**<br>`{A * B}`", unsafe_allow_html=True)
r4.markdown(f"**A ÷ B**<br>`{A / B}`", unsafe_allow_html=True)

st.divider()

st.header("3. Neutrosophication Simulator")
st.markdown("Convert a crisp dataset into neutrosophic data.")
crisp_input = st.text_area("Input Crisp Data (comma separated, use 'nan' for missing)", "12, 15, 14, nan, 22, 18, 11")
threshold = st.slider("Boundary Indeterminacy Threshold (δ)", 0.0, 0.5, 0.1)

if st.button("Neutrosophicate"):
    try:
        data = [float(x.strip()) if x.strip() != 'nan' else np.nan for x in crisp_input.split(',')]
        n_array = neutrosophicate(data, threshold)
        
        df_out = []
        for i, (orig, neut) in enumerate(zip(data, n_array.data)):
            # Color logic
            is_miss = pd.isna(orig)
            is_indt = neut.is_indeterminate(0.01)
            
            cClass = "missing-cell" if is_miss else ("indeterminate-cell" if is_indt else "determinate-cell")
            
            df_out.append({
                "Index": i,
                "Original": orig,
                "Neutrosophic N(T, I, F)": str(neut),
                "Class": cClass
            })
            
        df = pd.DataFrame(df_out)
        
        # Display colored table
        html = "<table><tr><th>Index</th><th>Original</th><th>Neutrosophic Form</th></tr>"
        for _, row in df.iterrows():
            html += f"<tr class='{row['Class']}'><td>{row['Index']}</td><td>{row['Original']}</td><td><code>{row['Neutrosophic N(T, I, F)']}</code></td></tr>"
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error parsing data: {e}")
