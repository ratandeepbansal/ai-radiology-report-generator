"""
Download CheXzero Model Weights
Helper script to download pre-trained CheXzero model weights from Google Drive

Author: MedAssist Copilot Team
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm


# Google Drive file IDs for CheXzero weights
# These are the pre-trained model checkpoints from the official repository
MODEL_WEIGHTS = {
    "best_64_5e-05_original_20000_0.864.pt": {
        "id": "1EY75OY5B6qVDGTtM8LYL0IvLEqZF1-X0",
        "size_mb": 338,
        "description": "Main CheXzero model checkpoint (Epoch 64, AUC 0.864)"
    },
    # Add more checkpoints if needed from:
    # https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno
}


def download_file_from_google_drive(file_id: str, destination: str, description: str = ""):
    """
    Download a file from Google Drive

    Args:
        file_id: Google Drive file ID
        destination: Local path to save the file
        description: Description for progress bar
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None

    # Check for virus scan warning
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Get file size
    total_size = int(response.headers.get('content-length', 0))

    # Download with progress bar
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"✅ Downloaded: {destination}")


def main():
    """Main download function"""
    print("=" * 70)
    print("CheXzero Model Weights Downloader")
    print("=" * 70)

    # Check if directory exists
    weights_dir = Path("models/CheXzero/checkpoints/chexzero_weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Download directory: {weights_dir}")
    print(f"\n📊 Models to download:")

    total_size = sum(info['size_mb'] for info in MODEL_WEIGHTS.values())
    print(f"   Total size: ~{total_size} MB\n")

    for filename, info in MODEL_WEIGHTS.items():
        print(f"   • {filename}")
        print(f"     Size: ~{info['size_mb']} MB")
        print(f"     Description: {info['description']}\n")

    # Ask for confirmation
    response = input("Continue with download? (y/n): ")
    if response.lower() != 'y':
        print("❌ Download cancelled")
        return

    print("\n🔄 Starting downloads...\n")

    # Download each model
    for filename, info in MODEL_WEIGHTS.items():
        destination = weights_dir / filename

        # Check if already exists
        if destination.exists():
            print(f"⏭️  Skipping {filename} (already exists)")
            continue

        try:
            download_file_from_google_drive(
                file_id=info['id'],
                destination=str(destination),
                description=filename
            )
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            print(f"   Please download manually from:")
            print(f"   https://drive.google.com/file/d/{info['id']}/view")
            continue

    print("\n" + "=" * 70)
    print("✅ Download complete!")
    print("=" * 70)

    # Verify downloads
    downloaded_files = list(weights_dir.glob("*.pt"))
    print(f"\n📦 Downloaded {len(downloaded_files)} model checkpoint(s)")

    if downloaded_files:
        print("\n🎉 CheXzero is ready to use!")
        print("\nNext steps:")
        print("   1. Run: python test_chexzero_analyzer.py")
        print("   2. Or use CheXzero backend in the pipeline:")
        print("      pipeline = ReportGenerationPipeline(vision_backend='chexzero')")
    else:
        print("\n⚠️  No model weights found. Manual download required.")
        print("\nManual Download Instructions:")
        print("   1. Visit: https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno")
        print("   2. Download all .pt files")
        print(f"   3. Save them to: {weights_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
