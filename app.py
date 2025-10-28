"""
MedAssist Copilot - Streamlit Application
AI-powered radiology report generator with voice input
"""

import streamlit as st
import sys
from pathlib import Path
import time
from PIL import Image
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import ReportGenerationPipeline
from src.rag import RAGSystem
from src.audio_processor import AudioProcessor
from src.report_manager import ReportManager
import config

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme - Dark mode with neon green accents */
    .main {
        background-color: #0E1117;
    }

    /* Headers */
    h1 {
        color: #39FF14;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 300;
    }

    h2, h3 {
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 300;
    }

    /* Report sections */
    .report-section {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #39FF14;
        margin: 10px 0;
    }

    .report-header {
        color: #39FF14;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1E1E1E;
    }

    /* Success message */
    .success-box {
        background-color: #1E3A1E;
        border-left: 4px solid #39FF14;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }

    /* Metrics */
    .metric-container {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }

    .metric-value {
        font-size: 28px;
        color: #39FF14;
        font-weight: bold;
    }

    .metric-label {
        font-size: 14px;
        color: #888888;
    }

    /* Button styling */
    .stButton>button {
        background-color: #39FF14;
        color: #000000;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }

    .stButton>button:hover {
        background-color: #2DE00F;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = None
if 'current_report' not in st.session_state:
    st.session_state.current_report = None
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
if 'vision_caption' not in st.session_state:
    st.session_state.vision_caption = None


@st.cache_resource
def load_pipeline():
    """Load the report generation pipeline (cached)"""
    try:
        pipeline = ReportGenerationPipeline(use_rag=True)
        return pipeline
    except Exception as e:
        st.error(f"Failed to load pipeline: {str(e)}")
        return None


@st.cache_resource
def load_rag_system():
    """Load the RAG system (cached)"""
    try:
        rag = RAGSystem()
        return rag
    except Exception as e:
        st.error(f"Failed to load RAG system: {str(e)}")
        return None


def main():
    """Main application"""

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"{config.APP_ICON} {config.APP_TITLE}")
        st.markdown("*AI-powered radiology report generator with voice input*")

    with col2:
        st.markdown("### ")
        if st.button("🔄 Reset", key="reset_button"):
            st.session_state.current_report = None
            st.session_state.report_generated = False
            st.session_state.vision_caption = None
            st.rerun()

    # Sidebar - Patient Information & Settings
    with st.sidebar:
        st.header("📋 Patient Information")

        patient_id = st.text_input(
            "Patient ID",
            value="P001",
            help="Enter patient identifier"
        )

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=65,
                help="Patient age in years"
            )

        with col2:
            gender = st.selectbox(
                "Gender",
                options=["M", "F", "Other"],
                help="Patient gender"
            )

        exam_date = st.date_input("Exam Date")

        indication = st.text_area(
            "Clinical Indication",
            placeholder="e.g., Fever and cough",
            help="Reason for the examination"
        )

        st.markdown("---")

        # Settings
        st.header("⚙️ Settings")

        use_rag = st.checkbox(
            "Enable RAG",
            value=True,
            help="Use prior reports for context"
        )

        detailed_vision = st.checkbox(
            "Detailed Vision Analysis",
            value=False,
            help="Generate multiple vision descriptions"
        )

        st.markdown("---")

        # Prior Reports (RAG)
        if use_rag and patient_id:
            st.header("📚 Prior Reports")

            try:
                if st.session_state.rag_system is None:
                    with st.spinner("Loading RAG system..."):
                        st.session_state.rag_system = load_rag_system()

                if st.session_state.rag_system:
                    prior_reports = st.session_state.rag_system.search_by_patient(patient_id)

                    if prior_reports:
                        st.success(f"Found {len(prior_reports)} prior report(s)")

                        for i, report in enumerate(prior_reports[:3], 1):
                            with st.expander(f"Report {i} - {report['date']}"):
                                st.markdown(f"**Impression:**")
                                st.write(report['report']['impression'][:200] + "...")
                    else:
                        st.info("No prior reports found for this patient")
            except Exception as e:
                st.warning(f"Could not load prior reports: {str(e)}")

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📸 Image Upload", "📄 Report", "📊 Evaluation"])

    # Tab 1: Image Upload
    with tab1:
        st.header("Upload Chest X-Ray")

        uploaded_file = st.file_uploader(
            "Choose an X-ray image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a chest X-ray image (JPG, JPEG, or PNG)"
        )

        if uploaded_file is not None:
            # Display image
            col1, col2 = st.columns([2, 1])

            with col1:
                image = Image.open(uploaded_file)
                st.image(
                    image,
                    caption="Uploaded X-Ray",
                    use_container_width=True
                )

            with col2:
                st.markdown("### Image Information")
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Size:** {image.size[0]} x {image.size[1]}")
                st.write(f"**Format:** {image.format}")
                st.write(f"**Mode:** {image.mode}")

                file_size = len(uploaded_file.getvalue()) / 1024
                st.write(f"**File Size:** {file_size:.2f} KB")

            st.markdown("---")

            # Generate Report Button
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if st.button("🤖 Generate Report", type="primary", use_container_width=True):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    try:
                        # Load pipeline if not already loaded
                        if st.session_state.pipeline is None:
                            with st.spinner("🔄 Loading AI models (this may take a moment)..."):
                                st.session_state.pipeline = load_pipeline()

                        if st.session_state.pipeline:
                            # Generate report
                            with st.spinner("🔬 Analyzing X-ray and generating report..."):
                                progress_bar = st.progress(0)

                                # Simulate progress
                                progress_bar.progress(25, text="Analyzing image...")

                                result = st.session_state.pipeline.generate_report(
                                    image=tmp_path,
                                    patient_id=patient_id,
                                    age=age,
                                    gender=gender,
                                    detailed_vision=detailed_vision
                                )

                                progress_bar.progress(100, text="Complete!")
                                time.sleep(0.5)
                                progress_bar.empty()

                                if result['success']:
                                    st.session_state.current_report = result
                                    st.session_state.report_generated = True
                                    st.session_state.vision_caption = result.get('vision_caption', '')

                                    st.success("✅ Report generated successfully!")
                                    st.balloons()
                                else:
                                    st.error(f"❌ Failed to generate report: {result.get('error', 'Unknown error')}")
                        else:
                            st.error("❌ Pipeline not loaded. Please check your configuration.")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

    # Tab 2: Report Display
    with tab2:
        if st.session_state.report_generated and st.session_state.current_report:
            report = st.session_state.current_report

            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(
                    f'<div class="metric-container">'
                    f'<div class="metric-value">{report.get("total_time", 0):.1f}s</div>'
                    f'<div class="metric-label">Total Time</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f'<div class="metric-container">'
                    f'<div class="metric-value">{report.get("vision_time", 0):.1f}s</div>'
                    f'<div class="metric-label">Vision Analysis</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with col3:
                st.markdown(
                    f'<div class="metric-container">'
                    f'<div class="metric-value">{report.get("llm_time", 0):.1f}s</div>'
                    f'<div class="metric-label">LLM Generation</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with col4:
                st.markdown(
                    f'<div class="metric-container">'
                    f'<div class="metric-value">{report.get("tokens_used", 0)}</div>'
                    f'<div class="metric-label">Tokens Used</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")

            # Vision Caption
            if st.session_state.vision_caption:
                with st.expander("🔍 Vision Analysis", expanded=False):
                    st.write(st.session_state.vision_caption)

            st.markdown("---")

            # Report Display
            st.header("📄 Radiology Report")

            # Parse and display sections
            sections = report.get('report_sections', {})

            # Findings
            st.markdown("### **FINDINGS:**")
            findings = st.text_area(
                "Findings",
                value=sections.get('findings', ''),
                height=150,
                label_visibility="collapsed",
                key="findings_edit"
            )

            st.markdown("### **IMPRESSION:**")
            impression = st.text_area(
                "Impression",
                value=sections.get('impression', ''),
                height=100,
                label_visibility="collapsed",
                key="impression_edit"
            )

            st.markdown("### **RECOMMENDATIONS:**")
            recommendations = st.text_area(
                "Recommendations",
                value=sections.get('recommendations', ''),
                height=100,
                label_visibility="collapsed",
                key="recommendations_edit"
            )

            st.markdown("---")

            # Action buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("💾 Save Report", use_container_width=True):
                    # Save report functionality
                    if st.session_state.pipeline:
                        saved_path = st.session_state.pipeline.save_report(report)
                        if saved_path:
                            st.success(f"✅ Report saved to: {saved_path}")
                        else:
                            st.error("❌ Failed to save report")

            with col2:
                if st.button("📄 Export PDF", use_container_width=True):
                    st.info("PDF export feature coming soon!")

            with col3:
                if st.button("🎤 Voice Note", use_container_width=True):
                    st.info("Voice note feature: Record audio and add to report")

        else:
            st.info("👈 Upload an X-ray image and generate a report to view it here")

    # Tab 3: Evaluation
    with tab3:
        st.header("📊 Evaluation & Metrics")

        if st.session_state.report_generated:
            st.success("Report generated! Metrics available above.")

            # Additional evaluation info
            st.markdown("### Performance Breakdown")

            if st.session_state.current_report:
                report = st.session_state.current_report

                # Create performance chart data
                import pandas as pd

                perf_data = {
                    'Component': ['Vision Analysis', 'RAG Retrieval', 'LLM Generation'],
                    'Time (seconds)': [
                        report.get('vision_time', 0),
                        report.get('rag_time', 0),
                        report.get('llm_time', 0)
                    ]
                }

                df = pd.DataFrame(perf_data)
                st.bar_chart(df.set_index('Component'))

                st.markdown("### Report Statistics")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Prior Reports Retrieved", report.get('prior_reports_count', 0))
                    st.metric("Total Processing Time", f"{report.get('total_time', 0):.2f}s")

                with col2:
                    st.metric("Tokens Used", report.get('tokens_used', 0))
                    st.metric("Vision Caption Length", len(st.session_state.vision_caption or ""))

        else:
            st.info("Generate a report to see evaluation metrics")

            # Show system info
            st.markdown("### System Information")
            st.write(f"**Vision Model:** {config.VISION_MODEL_NAME}")
            st.write(f"**LLM Model:** {config.LLM_MODEL_NAME}")
            st.write(f"**Embedding Model:** {config.EMBEDDING_MODEL_NAME}")
            st.write(f"**RAG Enabled:** {config.ENABLE_RAG}")

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #888888; font-size: 12px;">'
        f'MedAssist Copilot v{config.__version__ if hasattr(config, "__version__") else "0.3.0"} | '
        'Educational/Research Use Only | Not for Clinical Diagnosis'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
