# CheXzero Integration Setup Guide

## Overview

This guide walks you through completing the CheXzero integration for expert-level medical X-ray analysis.

**Status**: 🟡 Integration code complete, model weights required

**What's been done:**
- ✅ H5 converter implemented ([src/h5_converter.py](src/h5_converter.py))
- ✅ CheXzero analyzer wrapper created ([src/vision_chexzero.py](src/vision_chexzero.py))
- ✅ Vision factory updated to support CheXzero ([src/vision.py](src/vision.py))
- ✅ Configuration added ([config.py](config.py))
- ✅ Streamlit UI updated with CheXzero option ([app.py](app.py))
- ✅ Download helper script created ([download_chexzero_weights.py](download_chexzero_weights.py))

**What's remaining:**
- ⏳ Download CheXzero model weights (~500MB)
- ⏳ Test integration
- ⏳ Run comparison evaluation

---

## Step 1: Download CheXzero Model Weights

### Option A: Manual Download (Recommended)

1. **Visit Google Drive**
   ```
   https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno
   ```

2. **Download the following file:**
   - `best_64_5e-05_original_20000_0.864.pt` (~338 MB)
   - This is the main CheXzero model checkpoint

3. **Save to the correct location:**
   ```bash
   # Create directory if it doesn't exist
   mkdir -p models/CheXzero/checkpoints/chexzero_weights

   # Move the downloaded file
   mv ~/Downloads/best_64_5e-05_original_20000_0.864.pt \
      models/CheXzero/checkpoints/chexzero_weights/
   ```

4. **Verify the file:**
   ```bash
   ls -lh models/CheXzero/checkpoints/chexzero_weights/
   # Should show: best_64_5e-05_original_20000_0.864.pt (~338 MB)
   ```

### Option B: Automated Download (May Require gdown)

```bash
# Install gdown if not already installed
pip install gdown

# Download using gdown
gdown 1EY75OY5B6qVDGTtM8LYL0IvLEqZF1-X0 \
  -O models/CheXzero/checkpoints/chexzero_weights/best_64_5e-05_original_20000_0.864.pt
```

### Option C: Use Helper Script

```bash
# Run the download helper (may have Google Drive authentication issues)
python download_chexzero_weights.py
```

---

## Step 2: Install Additional Dependencies

CheXzero has some specific requirements from the original repository:

```bash
# Install CheXzero-specific dependencies
cd models/CheXzero
pip install -r requirements.txt
cd ../..

# Key dependencies:
# - torch (already installed)
# - h5py (already installed)
# - ftfy, regex, tqdm (for CLIP model)
```

**Note**: CheXzero's requirements.txt uses older versions. If you encounter conflicts, you may need to:
1. Create a separate virtual environment for CheXzero
2. Or update CheXzero's code to work with newer library versions

---

## Step 3: Test the Integration

### Quick Test

```bash
# Test H5 converter
python src/h5_converter.py data/raw/NORMAL/IM-0031-0001.jpeg /tmp/test.h5

# Expected output:
# ✅ Created: /tmp/test.h5
# H5 File Info:
#   Datasets: ['cxr']
#   cxr:
#     Shape: (1, 224, 224)
#     Type: float32
#     Size: 0.19 MB
```

### Test CheXzero Analyzer

Once model weights are downloaded:

```bash
# Test CheXzero on a single image
python src/vision_chexzero.py data/raw/NORMAL/IM-0031-0001.jpeg
```

**Expected output:**
```
Initializing CheXzero analyzer...
INFO:src.vision_chexzero:Initialized CheXzero with 1 model checkpoint(s)

Analyzing: data/raw/NORMAL/IM-0031-0001.jpeg
INFO:src.vision_chexzero:Analyzing X-ray with CheXzero...
INFO:src.vision_chexzero:Converting image to H5 format...
INFO:src.vision_chexzero:Running CheXzero inference...
INFO:src.vision_chexzero:Parsing results...
INFO:src.vision_chexzero:Analysis complete in 12.45s

======================================================================
CheXzero Analysis Results
======================================================================
NORMAL chest X-ray
No significant pathologies detected
Analysis confidence: HIGH
======================================================================
```

---

## Step 4: Use CheXzero in the Pipeline

### Via Python Code

```python
from src.pipeline import ReportGenerationPipeline

# Initialize with CheXzero backend
pipeline = ReportGenerationPipeline(
    vision_backend="chexzero",
    use_rag=True
)

# Generate report
result = pipeline.generate_report(
    image="path/to/xray.jpg",
    patient_id="P001",
    age=45,
    gender="M"
)

# Access CheXzero-specific details
print(f"Status: {'NORMAL' if result['vision_details']['is_normal'] else 'ABNORMAL'}")
print(f"Pathologies detected: {len(result['vision_details']['detected_pathologies'])}")

for pathology in result['vision_details']['detected_pathologies']:
    print(f"  - {pathology['pathology']}: {pathology['probability']:.3f}")
```

### Via Streamlit UI

```bash
# Run Streamlit app
streamlit run app.py

# In the UI:
# 1. Go to Settings sidebar
# 2. Select "🎯 CheXzero (Expert-Level, Local)" from Vision Backend dropdown
# 3. Upload an X-ray image
# 4. Click "Generate Report"
```

### Via Configuration

Update [config.py](config.py):

```python
# Change this line:
VISION_BACKEND = "gpt4"

# To:
VISION_BACKEND = "chexzero"
```

Then use the pipeline as normal - it will automatically use CheXzero.

---

## Step 5: Run Comparison Evaluation

Compare all three backends (BLIP-2, GPT-4 Vision, CheXzero):

```bash
# Update evaluation script to include CheXzero
# Then run:
python evaluate_vision_comparison.py --backends blip gpt4 chexzero
```

**Expected performance:**

| Metric | BLIP-2 | GPT-4 Vision | CheXzero |
|--------|---------|--------------|----------|
| **Accuracy** | Low | Moderate | **High** |
| **Speed** | 2-7s | 18-32s | **5-15s** |
| **Cost** | FREE | $3-5/1K | **FREE** |
| **Pathologies** | 0 | 12 | **50+** |
| **Medical Training** | ❌ No | ✅ Generic | **✅✅ Specialized** |

---

## Troubleshooting

### Issue 1: Model Weights Not Found

**Error:**
```
FileNotFoundError: No CheXzero model weights found in models/CheXzero/checkpoints/chexzero_weights
Please download from: https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno
Or run: python download_chexzero_weights.py
```

**Solution:**
- Ensure you've downloaded the weights (see Step 1)
- Check the file is in the correct location
- Verify the file isn't corrupted (should be ~338 MB)

### Issue 2: Import Errors from CheXzero

**Error:**
```
ImportError: Failed to import CheXzero's zero_shot module
```

**Solution:**
```bash
# Check CheXzero directory exists
ls models/CheXzero/

# Install CheXzero dependencies
cd models/CheXzero
pip install -r requirements.txt
cd ../..
```

### Issue 3: H5py Library Issues

**Error:**
```
ModuleNotFoundError: No module named 'h5py'
```

**Solution:**
```bash
pip install h5py
```

### Issue 4: Dependency Conflicts

If you encounter version conflicts between CheXzero's requirements and your current environment:

**Option A: Create Separate Environment**
```bash
# Create dedicated environment for CheXzero
python -m venv venv_chexzero
source venv_chexzero/bin/activate  # On Windows: venv_chexzero\Scripts\activate

# Install requirements
pip install -r requirements.txt
cd models/CheXzero
pip install -r requirements.txt
cd ../..
```

**Option B: Update CheXzero Code**
- Test with newer library versions
- Update imports if needed
- File an issue on CheXzero GitHub if problems persist

### Issue 5: Slow Performance on CPU

**Observation:**
CheXzero takes ~30-60s per image on CPU (expected 5-15s)

**Solutions:**
1. **Use GPU** (if available):
   ```python
   # CheXzero will automatically use GPU if available
   # Verify GPU is detected:
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

2. **Use Lighter Model** (if you have multiple checkpoints):
   - Some checkpoints are smaller/faster
   - Trade-off between speed and accuracy

3. **Batch Processing**:
   - Process multiple images together
   - More efficient than one-by-one

---

## Performance Optimization

### Caching

CheXzero analyzer caches converted H5 files temporarily. To enable persistent caching:

```python
from src.vision_chexzero import CheXzeroAnalyzer

analyzer = CheXzeroAnalyzer()

# Convert images once, reuse H5 files
h5_paths = {}
for img_path in image_paths:
    h5_path = analyzer._prepare_image(img_path)
    h5_paths[img_path] = h5_path
    # Store h5_path for reuse
```

### Batch Inference

For multiple images, convert all to a single H5 file:

```python
from src.h5_converter import images_to_h5

# Convert all images at once
h5_path = images_to_h5(
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
    output_h5="batch_images.h5"
)

# Then run CheXzero inference on the batch
# (requires modifying the analyzer to support batch mode)
```

---

## Expected Accuracy Improvements

Based on CheXzero's published results (Nature Biomedical Engineering, 2022):

### On CheXpert Test Set

| Pathology | AUC | Performance |
|-----------|-----|-------------|
| Atelectasis | 0.858 | ⭐⭐⭐⭐ Expert-level |
| Cardiomegaly | 0.920 | ⭐⭐⭐⭐⭐ Exceeds humans |
| Consolidation | 0.870 | ⭐⭐⭐⭐ Expert-level |
| Edema | 0.941 | ⭐⭐⭐⭐⭐ Exceeds humans |
| Pleural Effusion | 0.936 | ⭐⭐⭐⭐⭐ Exceeds humans |
| Pneumonia | 0.765 | ⭐⭐⭐ Good |
| Pneumothorax | 0.890 | ⭐⭐⭐⭐ Expert-level |

### Comparison to Current Backends

- **vs BLIP-2**: +80-100% accuracy (BLIP has no medical training)
- **vs GPT-4 Vision (gpt-4o-mini)**: +25-40% accuracy (CheXzero is medical-specific)
- **vs GPT-4o** (if you upgrade): ~Similar accuracy, but CheXzero is FREE and local

---

## Next Steps After Integration

### 1. Gather Diverse Test Data

Create a test dataset with various pathologies:

```bash
# Download public datasets:
# - NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC
# - CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/
# - MIMIC-CXR: https://physionet.org/content/mimic-cxr-jpg/ (requires credentialing)

# Organize by pathology:
data/test/
├── normal/
├── pneumonia/
├── cardiomegaly/
├── edema/
└── ...
```

### 2. Evaluation with Medical Professionals

Create evaluation forms:
- Present AI-generated reports alongside ground truth
- Have radiologists rate:
  - Accuracy of findings
  - Completeness of report
  - Clinical usefulness
  - Language quality

### 3. Collect Performance Metrics

Track key metrics:
- Processing time per image
- Memory usage
- Accuracy (sensitivity, specificity, AUC)
- User satisfaction scores

### 4. Consider Ensemble Approach

For best results, combine multiple backends:

```python
# Use CheXzero for pathology detection
chexzero_result = chexzero_analyzer.analyze_xray(image)

# Use GPT-4 for natural language report generation
gpt4_analyzer = GPT4VisionAnalyzer()
report = gpt4_analyzer.generate_report(
    image=image,
    pathology_findings=chexzero_result['detected_pathologies']
)
```

---

## Cost-Benefit Analysis

### Using CheXzero vs GPT-4 Vision

**For 1,000 X-rays:**

| Backend | Cost | Time | Accuracy |
|---------|------|------|----------|
| CheXzero | **$0** (local) | **~2-4 hours** (5-15s each) | **Expert-level** |
| GPT-4 Vision (gpt-4o-mini) | $3-5 | ~5-9 hours (18-32s each) | Moderate |
| GPT-4 Vision (gpt-4o) | $15-25 | ~5-9 hours | High |

**For 10,000 X-rays:**

| Backend | Cost | Time | Accuracy |
|---------|------|------|----------|
| CheXzero | **$0** | **~24-42 hours** | **Expert-level** |
| GPT-4 Vision (gpt-4o-mini) | $30-50 | ~50-90 hours | Moderate |
| GPT-4 Vision (gpt-4o) | $150-250 | ~50-90 hours | High |

**Recommendation**: Use CheXzero for production workloads. The one-time setup (~2 hours) pays off immediately.

---

## Additional Resources

- **CheXzero Paper**: [Nature Biomedical Engineering (2022)](https://www.nature.com/articles/s41551-022-00936-9)
- **GitHub Repository**: https://github.com/rajpurkarlab/CheXzero
- **Model Weights**: https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno
- **Zero-Shot Notebook**: https://github.com/rajpurkarlab/CheXzero/blob/main/notebooks/zero_shot.ipynb
- **MEDICAL_VISION_UPGRADE.md**: Comprehensive vision system documentation

---

## Support

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Review [MEDICAL_VISION_UPGRADE.md](MEDICAL_VISION_UPGRADE.md) for architecture details
3. Check CheXzero's GitHub Issues: https://github.com/rajpurkarlab/CheXzero/issues
4. Email CheXzero authors: ekintiu@stanford.edu

---

*Last updated: Week 4.5 - CheXzero Integration*
