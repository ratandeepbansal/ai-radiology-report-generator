"""
Configuration file for MedAssist Copilot
Contains model paths, API keys, prompts, and system parameters
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== API Configuration ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")

# ==================== Model Configuration ====================
# Vision Model Configuration
VISION_BACKEND = "gpt4"  # Options: "blip" (fast/generic), "gpt4" (medical-grade), "auto"
VISION_MODEL_NAME = "Salesforce/blip-image-captioning-base"  # Used when VISION_BACKEND is "blip"
VISION_MODEL_DEVICE = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"

# GPT-4 Vision Configuration (Medical-Grade Analysis - Recommended)
GPT4_VISION_MODEL = "gpt-4o-mini"  # GPT-4o-mini with vision capabilities (use "gpt-4o" if available)
GPT4_VISION_TEMPERATURE = 0.3  # Lower for consistent medical analysis
GPT4_VISION_MAX_TOKENS = 1500

# LLM Configuration
LLM_MODEL_NAME = "gpt-4o-mini"  # or "gpt-4" for better quality. Cannot use 5-mini as the style is different.
LLM_MAX_TOKENS = 1000
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Alternative biomedical model: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# Whisper Model Configuration
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large

# ==================== Data Paths ====================
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
MODELS_DIR = "models"

# Patient reports database
PATIENT_REPORTS_PATH = os.path.join(REPORTS_DIR, "patient_reports.json")

# ==================== Image Processing Parameters ====================
IMAGE_SIZE = (384, 384)  # Default size for vision models
IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".dicom"]
MAX_IMAGE_SIZE_MB = 10

# ==================== RAG Configuration ====================
VECTOR_DB_TYPE = "chromadb"  # Options: chromadb, faiss
COLLECTION_NAME = "patient_reports"
RAG_TOP_K = 3  # Number of relevant reports to retrieve
SIMILARITY_THRESHOLD = 0.5

# ==================== Prompt Templates ====================

SYSTEM_PROMPT = """You are an expert radiologist assistant with years of experience in interpreting chest X-rays.
Your role is to generate structured, professional radiology reports based on X-ray findings."""

REPORT_GENERATION_PROMPT = """
Task: Generate a professional, structured radiology report based on AI-detected findings from chest X-ray analysis.

AI Vision Analysis:
{vision_caption}

Prior Report Context (for reference):
{rag_context}

Patient Information:
- Patient ID: {patient_id}
- Age: {age}
- Gender: {gender}
- Date: {date}

Instructions:
1. Review the AI-detected findings carefully
2. Synthesize the information into a professional radiology report
3. Use appropriate medical terminology
4. Structure your report as follows:

**FINDINGS:**
[Provide a systematic description of the chest X-ray including:
 - Heart size and contour
 - Lung fields (both right and left)
 - Pleural spaces
 - Mediastinum
 - Bones and soft tissues
 - Any detected abnormalities with their locations and characteristics]

**IMPRESSION:**
[Provide clinical interpretation:
 - Primary diagnosis or most likely diagnosis
 - Differential diagnoses if applicable
 - Severity assessment
 - Comparison to prior studies if relevant context provided]

**RECOMMENDATIONS:**
[Suggest appropriate follow-up:
 - Additional imaging if needed
 - Clinical correlation recommendations
 - Follow-up timeframe if applicable
 - Any urgent findings requiring immediate attention]

Guidelines:
- Be professional, concise, and clinically accurate
- If findings indicate a normal study, state clearly "No acute cardiopulmonary abnormality"
- For abnormal findings, be specific about location, size, and characteristics
- Consider patient age and gender when relevant
- Reference AI confidence levels when discussing uncertain findings
"""

NORMAL_FINDINGS_PROMPT = """
Task: Generate a structured radiology report for a NORMAL chest X-ray.

Vision Model Findings:
{vision_caption}

Patient Information:
- Patient ID: {patient_id}
- Date: {date}

Generate a report indicating normal findings in a professional format.
"""

COMPARISON_PROMPT = """
Task: Generate a comparative radiology report analyzing changes between two X-rays.

Current X-ray Findings:
{current_findings}

Previous X-ray Findings:
{previous_findings}

Previous Report:
{previous_report}

Patient Information:
- Patient ID: {patient_id}
- Current Date: {current_date}
- Previous Date: {previous_date}

Please generate a comparative report highlighting:
1. Stable findings
2. New or worsening findings
3. Improved findings
4. Clinical significance of changes
"""

# ==================== Voice Processing Configuration ====================
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = "wav"
MAX_RECORDING_DURATION = 300  # seconds (5 minutes)

# Voice commands
VOICE_COMMANDS = {
    "add to findings": "findings",
    "add to impression": "impression",
    "add to recommendations": "recommendations",
    "new report": "new_report",
    "save report": "save_report"
}

# ==================== UI Configuration ====================
APP_TITLE = "MedAssist Copilot"
APP_ICON = "🏥"
THEME_COLOR = "#39FF14"  # Neon green

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": APP_TITLE,
    "page_icon": APP_ICON,
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# ==================== Performance Configuration ====================
BATCH_SIZE = 4
NUM_WORKERS = 2
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)

# ==================== Logging Configuration ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "medassist.log"

# ==================== Evaluation Configuration ====================
EVALUATION_METRICS = ["bleu", "rouge", "latency"]
BLEU_MAX_N = 4  # Maximum n-gram for BLEU score
ROUGE_METRICS = ["rouge-1", "rouge-2", "rouge-l"]

# ==================== Security & Privacy ====================
ALLOWED_FILE_TYPES = ["jpg", "jpeg", "png"]
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB in bytes

# Data anonymization
ANONYMIZE_REPORTS = True
MASK_PATIENT_ID = False

# ==================== Error Messages ====================
ERROR_MESSAGES = {
    "no_api_key": "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.",
    "invalid_image": "Invalid image format. Please upload a JPG, JPEG, or PNG file.",
    "file_too_large": f"File size exceeds {MAX_IMAGE_SIZE_MB}MB limit.",
    "model_load_failed": "Failed to load model. Please check your configuration.",
    "generation_failed": "Failed to generate report. Please try again.",
    "audio_processing_failed": "Failed to process audio input. Please try again."
}

# ==================== Feature Flags ====================
ENABLE_RAG = True
ENABLE_VOICE_INPUT = True
ENABLE_REPORT_EXPORT = True
ENABLE_COMPARISON_MODE = False  # Set to True when implementing comparison feature
ENABLE_METRICS_TRACKING = True

# ==================== Development Settings ====================
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
VERBOSE_LOGGING = DEBUG_MODE

# ==================== Validation ====================
def validate_config():
    """Validate critical configuration settings"""
    errors = []

    if not OPENAI_API_KEY and ENABLE_RAG:
        errors.append("OpenAI API key is required when RAG is enabled")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR, exist_ok=True)

    return errors

if __name__ == "__main__":
    # Test configuration
    print("MedAssist Copilot Configuration")
    print("=" * 50)
    print(f"Vision Backend: {VISION_BACKEND}")
    if VISION_BACKEND == "gpt4":
        print(f"GPT-4 Vision Model: {GPT4_VISION_MODEL}")
    else:
        print(f"Vision Model: {VISION_MODEL_NAME}")
        print(f"Device: {VISION_MODEL_DEVICE}")
    print(f"LLM Model: {LLM_MODEL_NAME}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print(f"Whisper Model: {WHISPER_MODEL_SIZE}")
    print(f"RAG Enabled: {ENABLE_RAG}")
    print(f"Voice Input Enabled: {ENABLE_VOICE_INPUT}")
    print("=" * 50)

    # Validate configuration
    errors = validate_config()
    if errors:
        print("\nConfiguration Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nConfiguration validated successfully!")
