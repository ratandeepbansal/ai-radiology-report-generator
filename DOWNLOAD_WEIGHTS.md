# Download CheXzero Model Weights - Manual Instructions

## Quick Steps

The automatic download failed, but manual download is simple:

### Option 1: Direct Browser Download (Recommended)

1. **Click this link**: https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno

2. **Find the file**: `best_64_5e-05_original_20000_0.864.pt` (338 MB)

3. **Download it**:
   - Click on the file
   - Click the download button (⬇️ icon)
   - Wait for download to complete

4. **Move to correct location**:
   ```bash
   # From your Downloads folder
   mv ~/Downloads/best_64_5e-05_original_20000_0.864.pt \
      models/CheXzero/checkpoints/chexzero_weights/

   # Verify it's there
   ls -lh models/CheXzero/checkpoints/chexzero_weights/
   # Should show: best_64_5e-05_original_20000_0.864.pt (338 MB)
   ```

### Option 2: Use wget with Cookies (Advanced)

If you have gdown issues but want CLI download:

```bash
# Install wget if needed
brew install wget  # macOS
# or: sudo apt-get install wget  # Linux

# Download with wget (may require authentication)
cd models/CheXzero/checkpoints/chexzero_weights/
wget --load-cookies /tmp/cookies.txt \
  "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1EY75OY5B6qVDGTtM8LYL0IvLEqZF1-X0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1EY75OY5B6qVDGTtM8LYL0IvLEqZF1-X0" \
  -O best_64_5e-05_original_20000_0.864.pt && rm -rf /tmp/cookies.txt
```

### Option 3: Alternative Model Sources

If Google Drive access is blocked:

1. **Check CheXzero GitHub Releases**: https://github.com/rajpurkarlab/CheXzero/releases
   (May have alternative hosting)

2. **Academic Institution Access**:
   If you're at a university, your institution may have local copies

3. **Request from Authors**:
   Email: ekintiu@stanford.edu
   Subject: "CheXzero Model Weights Request"

## Verification

After downloading, verify the file:

```bash
# Check file exists and size
ls -lh models/CheXzero/checkpoints/chexzero_weights/best_64_5e-05_original_20000_0.864.pt

# Expected output:
# -rw-r--r--  1 user  staff   338M  [date]  best_64_5e-05_original_20000_0.864.pt

# Check it's not corrupted (optional)
python -c "import torch; print('File OK' if torch.load('models/CheXzero/checkpoints/chexzero_weights/best_64_5e-05_original_20000_0.864.pt', map_location='cpu') else 'File corrupted')"
```

## What's This File?

This is the **pre-trained CheXzero model checkpoint**:
- **Trained on**: MIMIC-CXR dataset (377,000+ chest X-rays)
- **Published**: Nature Biomedical Engineering (2022)
- **Performance**: Matches expert radiologist accuracy
- **Size**: 338 MB (CLIP-based vision-language model)
- **Detects**: 50+ chest pathologies with confidence scores

## After Download

Once you have the file in place, continue with:

```bash
# Test the integration
python src/vision_chexzero.py data/raw/NORMAL/IM-0031-0001.jpeg

# Or skip directly to using it in Streamlit
streamlit run app.py
# Select "CheXzero" from Vision Backend dropdown
```

## Troubleshooting

**File won't download?**
- Try incognito/private browsing mode
- Clear browser cache
- Try different browser (Chrome, Firefox, Safari)
- Check internet connection

**Download keeps failing?**
- Download in parts using a download manager
- Try at different times (less traffic)
- Use university/institutional network if available

**File size wrong?**
- Should be exactly 338 MB (354,484,994 bytes)
- If different, re-download (file may be corrupted)

## Need Help?

If you still can't download:
1. Check [CHEXZERO_SETUP.md](CHEXZERO_SETUP.md) troubleshooting section
2. File an issue with details about the error
3. Email CheXzero authors for alternative access

---

*For more information, see [CHEXZERO_SETUP.md](CHEXZERO_SETUP.md)*
