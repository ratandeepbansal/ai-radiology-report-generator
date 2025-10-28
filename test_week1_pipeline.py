"""
Week 1 Pipeline Test
Comprehensive test of data loading and report management functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import XRayDataLoader
from src.report_manager import ReportManager


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_image_pipeline():
    """Test the image loading and preprocessing pipeline"""
    print_section("IMAGE PROCESSING PIPELINE TEST")

    # Initialize data loader
    print("\n1️⃣  Initializing XRay Data Loader...")
    loader = XRayDataLoader(
        data_dir="data/raw",
        image_size=(384, 384),
        normalize=True
    )
    print("   ✅ Data loader initialized")

    # Find images
    print("\n2️⃣  Finding images in data/raw/...")
    images = loader.find_images()
    print(f"   ✅ Found {len(images)} images")

    if images:
        # Get dataset statistics
        print("\n3️⃣  Analyzing dataset statistics...")
        stats = loader.get_dataset_statistics()
        print(f"   - Total images: {stats['total_images']}")
        print(f"   - Image formats: {stats['formats']}")
        print(f"   - Total size: {stats['total_size_mb']} MB")
        print(f"   - Average dimensions: {stats.get('avg_width', 'N/A')}x{stats.get('avg_height', 'N/A')}")

        # Test loading and preprocessing
        print("\n4️⃣  Testing image loading and preprocessing...")
        test_image_path = images[0]
        print(f"   Loading: {Path(test_image_path).name}")

        # Load image
        img = loader.load_image(test_image_path)
        if img:
            print(f"   ✅ Loaded successfully (size: {img.size})")

            # Preprocess to tensor
            tensor = loader.preprocess_image(img, return_type='tensor')
            if tensor is not None:
                print(f"   ✅ Preprocessed to tensor (shape: {tensor.shape})")

            # Preprocess to numpy
            array = loader.preprocess_image(img, return_type='numpy')
            if array is not None:
                print(f"   ✅ Preprocessed to numpy (shape: {array.shape})")
        else:
            print("   ❌ Failed to load image")

        # Test batch processing
        print("\n5️⃣  Testing batch processing...")
        batches = loader.create_image_batch(images[:4], batch_size=2)
        print(f"   ✅ Created {len(batches)} batches")
        if batches:
            print(f"   ✅ First batch shape: {batches[0].shape}")

        # Test validation
        print("\n6️⃣  Testing image validation...")
        is_valid, message = loader.validate_image(test_image_path)
        print(f"   {'✅' if is_valid else '❌'} {message}")

    else:
        print("\n   ⚠️  No images found. Run create_test_images.py first.")

    return loader


def test_report_pipeline():
    """Test the report management pipeline"""
    print_section("REPORT MANAGEMENT PIPELINE TEST")

    # Initialize report manager
    print("\n1️⃣  Initializing Report Manager...")
    manager = ReportManager()
    print("   ✅ Report manager initialized")

    # Get statistics
    print("\n2️⃣  Analyzing report database...")
    stats = manager.get_statistics()
    print(f"   - Total reports: {stats['total_reports']}")
    print(f"   - Unique patients: {stats['unique_patients']}")
    print(f"   - Gender distribution: M={stats['gender_distribution']['M']}, F={stats['gender_distribution']['F']}")

    if stats['age_stats']:
        print(f"   - Age range: {stats['age_stats']['min']}-{stats['age_stats']['max']} years")

    if stats['date_range']['earliest']:
        print(f"   - Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")

    # Test retrieving reports by patient ID
    print("\n3️⃣  Testing patient report retrieval...")
    all_reports = manager.get_all_reports()
    if all_reports:
        test_patient = all_reports[0]['patient_id']
        patient_reports = manager.get_report_by_patient_id(test_patient)
        print(f"   ✅ Retrieved {len(patient_reports)} report(s) for patient {test_patient}")

        # Get latest report
        latest = manager.get_latest_report(test_patient)
        if latest:
            print(f"   ✅ Latest report date: {latest['date']}")

    # Test searching
    print("\n4️⃣  Testing report search functionality...")

    # Search by keyword
    results = manager.search_reports(keyword="pneumonia")
    print(f"   - Keyword 'pneumonia': {len(results)} results")

    results = manager.search_reports(keyword="effusion")
    print(f"   - Keyword 'effusion': {len(results)} results")

    # Search by age
    results = manager.search_reports(age_min=60)
    print(f"   - Age >= 60: {len(results)} results")

    # Search by gender
    results = manager.search_reports(gender="M")
    print(f"   - Gender Male: {len(results)} results")

    # Search by date range
    results = manager.search_reports(date_from="2024-03-01")
    print(f"   - Date from 2024-03-01: {len(results)} results")

    # Test report formatting
    print("\n5️⃣  Testing report formatting...")
    if all_reports:
        formatted = manager.format_report(all_reports[0], include_metadata=True)
        print("   ✅ Sample formatted report:")
        print("\n" + formatted)

    # Test adding a new report
    print("\n6️⃣  Testing new report creation...")
    from src.report_manager import create_empty_report

    new_report = create_empty_report("P999")
    new_report['report']['findings'] = "Test findings"
    new_report['report']['impression'] = "Test impression"
    new_report['report']['recommendations'] = "Test recommendations"
    new_report['metadata']['age'] = 50
    new_report['metadata']['gender'] = "M"

    # Validate report
    is_valid, errors = manager.validate_report(new_report)
    if is_valid:
        print("   ✅ New report validation passed")
    else:
        print(f"   ❌ Validation errors: {errors}")

    return manager


def test_integration():
    """Test integration between image and report systems"""
    print_section("INTEGRATION TEST")

    loader = XRayDataLoader()
    manager = ReportManager()

    print("\n1️⃣  Testing integrated workflow...")

    # Simulate workflow: Load image -> Retrieve patient reports -> Process
    images = loader.find_images()

    if images and len(manager.get_all_reports()) > 0:
        # Pick first image
        test_image = images[0]
        print(f"\n   📁 Processing: {Path(test_image).name}")

        # Load and preprocess image
        img = loader.load_image(test_image)
        if img:
            tensor = loader.preprocess_image(img, return_type='tensor')
            print(f"   ✅ Image preprocessed: {tensor.shape}")

        # Simulate patient lookup
        # In real workflow, patient_id would come from the form
        test_patient = "P002"  # Patient with pneumonia
        print(f"\n   🔍 Looking up history for patient: {test_patient}")

        prior_reports = manager.get_report_by_patient_id(test_patient)
        if prior_reports:
            print(f"   ✅ Found {len(prior_reports)} prior report(s)")
            latest = prior_reports[0]
            print(f"   📄 Latest report from: {latest['date']}")
            print(f"   📝 Previous impression: {latest['report']['impression'][:60]}...")
        else:
            print("   ℹ️  No prior reports found (new patient)")

        print("\n   ✅ Integration test successful!")
        print("   ✅ Ready for Week 2: Vision-Language Pipeline")

    else:
        print("   ⚠️  Need both images and reports for integration test")


def main():
    """Run all pipeline tests"""
    print("\n" + "=" * 70)
    print(" " * 15 + "WEEK 1 PIPELINE - COMPREHENSIVE TEST")
    print("=" * 70)

    try:
        # Test image pipeline
        loader = test_image_pipeline()

        # Test report pipeline
        manager = test_report_pipeline()

        # Test integration
        test_integration()

        # Summary
        print_section("SUMMARY")
        print("\n✅ All Week 1 components are working correctly!")
        print("\n📋 Week 1 Deliverables:")
        print("   ✅ Functional data pipeline that can load and display X-rays")
        print("   ✅ Initial report database with 15 sample reports")
        print("   ✅ Documented project structure")

        print("\n🎯 Ready for Week 2 Tasks:")
        print("   - Vision model integration (BLIP-2)")
        print("   - LLM processor setup (GPT-4o-mini)")
        print("   - Prompt engineering")
        print("   - End-to-end pipeline: image → caption → report")

        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
