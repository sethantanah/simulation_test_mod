import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import load_css
from reporting.pdf_export import generate_pdf_report

st.set_page_config(page_title="Export Research Report", layout="wide")
load_css()

st.title("📄 Export Academic PDF Report")
st.markdown("Automatically generate a comprehensive, publication-ready research report embedding all statistical analyses, derivations, and simulation tracking charts.")

col_conf, col_prev = st.columns([4, 6])

with col_conf:
    st.markdown("### Report Configuration")
    
    title = st.text_input("Report Title", "Modification of Neutrosophic Non-Parametric Tests and Their Application to Real-Life Data")
    author = st.text_input("Author Name", "Akua Agyapomah Oteng (PhD Candidate)")
    
    st.markdown("#### Include Sections")
    s1 = st.checkbox("1. Executive Summary", True)
    s2 = st.checkbox("2. Introduction & Background", True)
    s3 = st.checkbox("3. Mathematical Framework", True)
    s4 = st.checkbox("4. Proposed Modifications", True)
    s5 = st.checkbox("5. Simulation Study Results", True)
    s6 = st.checkbox("6. Real-Life Data Applications", True)
    s7 = st.checkbox("7. Comparative Analysis", True)
    s8 = st.checkbox("8. Conclusions & Recommendations", True)
    
    st.markdown("#### Render Options")
    st.radio("Chart Resolution", ["150 DPI (Fast)", "300 DPI (High Quality)"])
    
    generate_btn = st.button("Generate Final Report", type="primary")

with col_prev:
    st.markdown("### Generation Status")
    
    if generate_btn:
        config = {
            'title': title, 'author': author,
            'sections': {
                'Executive Summary': s1, 'Introduction & Background': s2,
                'Mathematical Framework': s3, 'Proposed Modifications': s4,
                'Simulation Study Results': s5, 'Real-Life Data Applications': s6,
                'Comparative Analysis': s7, 'Conclusions & Recommendations': s8
            }
        }
        
        output_filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "NeutroStat_Research_Report.pdf")
        
        prog = st.progress(0.0)
        status_txt = st.empty()
        status_txt.text("Initializing ReportLab engine...")
        
        def pb(current, total):
            prog.progress(current / total)
            status_txt.text(f"Building section elements: {int((current/total)*100)}%...")
            
        try:
            generate_pdf_report(config, output_filepath, pb)
            status_txt.success("PDF Generation Complete!")
            
            with open(output_filepath, "rb") as pdf_file:
                PDFbyte = pdf_file.read()

            st.download_button(
                label="📥 Download Research Report PDF",
                data=PDFbyte,
                file_name="NeutroStat_Research_Report_2025.pdf",
                mime='application/octet-stream',
                type="primary",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
            st.warning("Please ensure all dependencies including `reportlab` are installed.")
    else:
        st.info("Configure the report settings on the left and click Generate to build the PDF.")
