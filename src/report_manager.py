"""
Report Manager Module for MedAssist Copilot
Handles loading, querying, and managing patient radiology reports
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportManager:
    """
    Manager for patient radiology reports
    Provides functionality to load, query, save, and manage reports
    """

    def __init__(self, reports_path: str = "data/reports/patient_reports.json"):
        """
        Initialize the report manager

        Args:
            reports_path: Path to the JSON file containing patient reports
        """
        self.reports_path = Path(reports_path)
        self.reports: List[Dict[str, Any]] = []
        self.reports_by_id: Dict[str, List[Dict[str, Any]]] = {}

        # Load reports if file exists
        if self.reports_path.exists():
            self.load_reports()
        else:
            logger.warning(f"Reports file not found: {self.reports_path}")
            logger.info("Starting with empty report database")

    def load_reports(self) -> bool:
        """
        Load reports from JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.reports_path, 'r', encoding='utf-8') as f:
                self.reports = json.load(f)

            # Index reports by patient_id
            self._index_reports()

            logger.info(f"Successfully loaded {len(self.reports)} reports from {self.reports_path}")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in reports file: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error loading reports: {str(e)}")
            return False

    def _index_reports(self):
        """Create an index of reports by patient_id for faster lookup"""
        self.reports_by_id = {}

        for report in self.reports:
            patient_id = report.get('patient_id')
            if patient_id:
                if patient_id not in self.reports_by_id:
                    self.reports_by_id[patient_id] = []
                self.reports_by_id[patient_id].append(report)

        # Sort reports by date for each patient
        for patient_id in self.reports_by_id:
            self.reports_by_id[patient_id].sort(
                key=lambda x: x.get('date', ''),
                reverse=True
            )

    def save_reports(self, output_path: Optional[str] = None) -> bool:
        """
        Save reports to JSON file

        Args:
            output_path: Optional custom output path (defaults to self.reports_path)

        Returns:
            True if successful, False otherwise
        """
        save_path = Path(output_path) if output_path else self.reports_path

        try:
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.reports, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully saved {len(self.reports)} reports to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving reports: {str(e)}")
            return False

    def get_all_reports(self) -> List[Dict[str, Any]]:
        """
        Get all reports

        Returns:
            List of all report dictionaries
        """
        return self.reports.copy()

    def get_report_by_patient_id(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Get all reports for a specific patient

        Args:
            patient_id: Patient identifier

        Returns:
            List of reports for the patient (sorted by date, most recent first)
        """
        return self.reports_by_id.get(patient_id, []).copy()

    def get_latest_report(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent report for a patient

        Args:
            patient_id: Patient identifier

        Returns:
            Most recent report dictionary or None if not found
        """
        patient_reports = self.reports_by_id.get(patient_id, [])
        return patient_reports[0] if patient_reports else None

    def add_report(self, report: Dict[str, Any]) -> bool:
        """
        Add a new report to the database

        Args:
            report: Report dictionary with required fields

        Returns:
            True if successful, False otherwise
        """
        # Validate required fields
        required_fields = ['patient_id', 'date', 'report']
        if not all(field in report for field in required_fields):
            logger.error(f"Report missing required fields: {required_fields}")
            return False

        # Add report
        self.reports.append(report)

        # Update index
        patient_id = report['patient_id']
        if patient_id not in self.reports_by_id:
            self.reports_by_id[patient_id] = []
        self.reports_by_id[patient_id].append(report)
        self.reports_by_id[patient_id].sort(key=lambda x: x.get('date', ''), reverse=True)

        logger.info(f"Added report for patient {patient_id}")
        return True

    def search_reports(
        self,
        keyword: Optional[str] = None,
        patient_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        age_min: Optional[int] = None,
        age_max: Optional[int] = None,
        gender: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search reports with various filters

        Args:
            keyword: Search in findings, impression, and recommendations
            patient_id: Filter by patient ID
            date_from: Filter by date (YYYY-MM-DD)
            date_to: Filter by date (YYYY-MM-DD)
            age_min: Minimum age filter
            age_max: Maximum age filter
            gender: Gender filter ('M' or 'F')

        Returns:
            List of matching reports
        """
        results = self.reports.copy()

        # Filter by patient_id
        if patient_id:
            results = [r for r in results if r.get('patient_id') == patient_id]

        # Filter by keyword
        if keyword:
            keyword_lower = keyword.lower()
            results = [
                r for r in results
                if any(
                    keyword_lower in str(r.get('report', {}).get(field, '')).lower()
                    for field in ['findings', 'impression', 'recommendations']
                )
            ]

        # Filter by date range
        if date_from:
            results = [r for r in results if r.get('date', '') >= date_from]
        if date_to:
            results = [r for r in results if r.get('date', '') <= date_to]

        # Filter by age
        if age_min is not None:
            results = [
                r for r in results
                if r.get('metadata', {}).get('age', 0) >= age_min
            ]
        if age_max is not None:
            results = [
                r for r in results
                if r.get('metadata', {}).get('age', 999) <= age_max
            ]

        # Filter by gender
        if gender:
            results = [
                r for r in results
                if r.get('metadata', {}).get('gender') == gender
            ]

        logger.info(f"Search returned {len(results)} results")
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the report database

        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_reports': len(self.reports),
            'unique_patients': len(self.reports_by_id),
            'gender_distribution': {'M': 0, 'F': 0, 'Unknown': 0},
            'age_stats': {},
            'date_range': {'earliest': None, 'latest': None}
        }

        ages = []
        dates = []

        for report in self.reports:
            # Gender distribution
            gender = report.get('metadata', {}).get('gender', 'Unknown')
            if gender in stats['gender_distribution']:
                stats['gender_distribution'][gender] += 1
            else:
                stats['gender_distribution']['Unknown'] += 1

            # Collect ages
            age = report.get('metadata', {}).get('age')
            if age:
                ages.append(age)

            # Collect dates
            date = report.get('date')
            if date:
                dates.append(date)

        # Age statistics
        if ages:
            stats['age_stats'] = {
                'min': min(ages),
                'max': max(ages),
                'average': round(sum(ages) / len(ages), 1)
            }

        # Date range
        if dates:
            stats['date_range'] = {
                'earliest': min(dates),
                'latest': max(dates)
            }

        return stats

    def format_report(self, report: Dict[str, Any], include_metadata: bool = True) -> str:
        """
        Format a report for display

        Args:
            report: Report dictionary
            include_metadata: Whether to include patient metadata

        Returns:
            Formatted report string
        """
        output = []

        # Header
        output.append("=" * 70)
        output.append(f"Patient ID: {report.get('patient_id', 'Unknown')}")
        output.append(f"Date: {report.get('date', 'Unknown')}")

        if include_metadata and 'metadata' in report:
            meta = report['metadata']
            output.append(f"Age: {meta.get('age', 'Unknown')} | Gender: {meta.get('gender', 'Unknown')}")
            if 'exam_type' in meta:
                output.append(f"Exam Type: {meta['exam_type']}")
            if 'indication' in meta:
                output.append(f"Indication: {meta['indication']}")

        output.append("=" * 70)

        # Report content
        if 'report' in report:
            rep = report['report']

            output.append("\nFINDINGS:")
            output.append(rep.get('findings', 'N/A'))

            output.append("\nIMPRESSION:")
            output.append(rep.get('impression', 'N/A'))

            output.append("\nRECOMMENDATIONS:")
            output.append(rep.get('recommendations', 'N/A'))

        output.append("\n" + "=" * 70)

        return "\n".join(output)

    def export_patient_history(
        self,
        patient_id: str,
        output_file: Optional[str] = None
    ) -> Optional[str]:
        """
        Export all reports for a patient to a text file

        Args:
            patient_id: Patient identifier
            output_file: Optional output file path

        Returns:
            Output file path if successful, None otherwise
        """
        reports = self.get_report_by_patient_id(patient_id)

        if not reports:
            logger.warning(f"No reports found for patient {patient_id}")
            return None

        # Generate output filename if not provided
        if not output_file:
            output_file = f"data/reports/patient_{patient_id}_history.txt"

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Patient History Report for {patient_id}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Reports: {len(reports)}\n")
                f.write("=" * 70 + "\n\n")

                for i, report in enumerate(reports, 1):
                    f.write(f"REPORT {i} of {len(reports)}\n")
                    f.write(self.format_report(report))
                    f.write("\n\n")

            logger.info(f"Exported patient history to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Error exporting patient history: {str(e)}")
            return None

    def validate_report(self, report: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a report structure

        Args:
            report: Report dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required fields
        if 'patient_id' not in report:
            errors.append("Missing required field: patient_id")
        if 'date' not in report:
            errors.append("Missing required field: date")
        if 'report' not in report:
            errors.append("Missing required field: report")
        else:
            # Check report sub-fields
            report_obj = report['report']
            if 'findings' not in report_obj:
                errors.append("Missing report sub-field: findings")
            if 'impression' not in report_obj:
                errors.append("Missing report sub-field: impression")
            if 'recommendations' not in report_obj:
                errors.append("Missing report sub-field: recommendations")

        # Validate date format
        if 'date' in report:
            try:
                datetime.strptime(report['date'], '%Y-%m-%d')
            except ValueError:
                errors.append("Invalid date format (should be YYYY-MM-DD)")

        return (len(errors) == 0, errors)


# Utility functions

def create_empty_report(patient_id: str) -> Dict[str, Any]:
    """
    Create an empty report template

    Args:
        patient_id: Patient identifier

    Returns:
        Empty report dictionary
    """
    return {
        "patient_id": patient_id,
        "date": datetime.now().strftime('%Y-%m-%d'),
        "report": {
            "findings": "",
            "impression": "",
            "recommendations": ""
        },
        "metadata": {
            "age": None,
            "gender": None,
            "exam_type": "Chest X-Ray PA/Lateral",
            "indication": ""
        }
    }


# Main execution for testing
if __name__ == "__main__":
    print("=" * 70)
    print("MedAssist Copilot - Report Manager Test")
    print("=" * 70)

    # Initialize manager
    manager = ReportManager()

    print(f"\n✅ Report manager initialized")
    print(f"   Reports file: {manager.reports_path}")

    # Get statistics
    print("\n📊 Report Database Statistics:")
    stats = manager.get_statistics()
    print(f"   - Total reports: {stats['total_reports']}")
    print(f"   - Unique patients: {stats['unique_patients']}")
    print(f"   - Gender distribution: {stats['gender_distribution']}")
    if stats['age_stats']:
        print(f"   - Age range: {stats['age_stats']['min']}-{stats['age_stats']['max']} (avg: {stats['age_stats']['average']})")
    if stats['date_range']['earliest']:
        print(f"   - Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")

    # Test patient lookup
    if manager.reports:
        print("\n🔍 Testing patient lookup...")
        test_patient = manager.reports[0]['patient_id']
        reports = manager.get_report_by_patient_id(test_patient)
        print(f"   Found {len(reports)} report(s) for patient {test_patient}")

        # Display first report
        if reports:
            print("\n📄 Sample Report:")
            print(manager.format_report(reports[0]))

    # Test search
    print("\n🔎 Testing keyword search (pneumonia)...")
    results = manager.search_reports(keyword="pneumonia")
    print(f"   Found {len(results)} reports matching 'pneumonia'")

    # Test age filter
    print("\n🔎 Testing age filter (age > 60)...")
    results = manager.search_reports(age_min=60)
    print(f"   Found {len(results)} reports for patients aged 60+")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
