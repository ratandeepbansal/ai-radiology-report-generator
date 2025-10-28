"""
MedAssist Copilot - Source Package
AI-powered radiology report generator
"""

__version__ = "0.1.0"
__author__ = "MedAssist Team"

from .data_loader import XRayDataLoader
from .report_manager import ReportManager

__all__ = [
    "XRayDataLoader",
    "ReportManager",
]
