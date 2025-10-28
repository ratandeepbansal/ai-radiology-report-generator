"""
MedAssist Copilot - Source Package
AI-powered radiology report generator
"""

__version__ = "0.2.0"
__author__ = "MedAssist Team"

from .data_loader import XRayDataLoader
from .report_manager import ReportManager
from .vision import VisionAnalyzer
from .llm_processor import LLMProcessor
from .pipeline import ReportGenerationPipeline

__all__ = [
    "XRayDataLoader",
    "ReportManager",
    "VisionAnalyzer",
    "LLMProcessor",
    "ReportGenerationPipeline",
]
