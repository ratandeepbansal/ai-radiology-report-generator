# Quick Start - Week 2 Testing

## ✅ What's Been Built

Week 2 is complete! Here's what you can now do:

1. **Analyze X-ray images** with BLIP-2 vision model
2. **Generate radiology reports** with GPT LLM
3. **Complete pipeline**: Image → Vision → RAG → Report
4. **Batch process** multiple X-rays
5. **Track performance** metrics

---

## 🚀 Quick Testing Guide

### Option 1: Test Vision Only (No API key needed)

```bash
# Activate environment
source venv/bin/activate

# Test vision model on your X-rays
python src/vision.py
```

**Expected output:**
- Model loads in ~2 minutes (first time)
- Generates captions for 4 sample X-rays
- Shows processing statistics

---

### Option 2: Test Complete Pipeline (Requires API key)

#### Step 1: Get OpenAI API Key

1. Visit: https://platform.openai.com/api-keys
2. Sign up / Log in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)
5. Add $5-10 credits to your account

#### Step 2: Set Up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your key
nano .env
# or
code .env
```

Add this line:
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

#### Step 3: Test LLM Processor

```bash
# Test LLM with sample caption
python src/llm_processor.py
```

#### Step 4: Test Complete Pipeline

```bash
# Run end-to-end test
python src/pipeline.py

# Or run comprehensive test suite
python test_week2_pipeline.py
```

**Expected output:**
- Loads vision model
- Analyzes X-ray image
- Retrieves prior reports (RAG)
- Generates complete radiology report
- Shows performance metrics
- Saves report to file

---

## 📊 What You'll See

### Vision Output:
```
Image 1: IM-0031-0001.jpeg
Caption: an x-ray image of the chest
Processing time: 2.42s
```

### Complete Report Example:
```
GENERATED RADIOLOGY REPORT
==========================================
Patient ID: P001
Date: 2025-10-28

**FINDINGS:**
The lungs are clear bilaterally without focal consolidation,
pleural effusion, or pneumothorax. Cardiac silhouette is
normal in size and contour...

**IMPRESSION:**
No acute cardiopulmonary abnormality.

**RECOMMENDATIONS:**
Routine clinical follow-up as indicated.

Performance:
  Total time: 8.45s
  Vision: 2.31s
  LLM: 4.12s
  Tokens: 487
```

---

## 🎯 Testing Checklist

- [ ] Vision model loads successfully
- [ ] Can generate captions for X-rays
- [ ] OpenAI API key is configured
- [ ] LLM generates reports from captions
- [ ] Complete pipeline works end-to-end
- [ ] Reports are saved to data/reports/generated/
- [ ] Performance metrics are displayed

---

## 🐛 Troubleshooting

### Vision Model Issues

**Problem**: Model download fails
```bash
# Check internet connection
# Try again - downloads can be large (1-2 GB)
python src/vision.py
```

**Problem**: Out of memory
```bash
# Close other applications
# Vision model needs ~2GB RAM
```

### API Key Issues

**Problem**: "API key not found"
```bash
# Check .env file exists
ls -la .env

# Check key is set correctly
cat .env | grep OPENAI_API_KEY
```

**Problem**: "Invalid API key"
- Verify key starts with `sk-`
- Check no extra spaces in .env file
- Make sure you copied the entire key

**Problem**: "Rate limit" or "Insufficient quota"
- Add credits to your OpenAI account
- Wait a moment and try again

---

## 📝 Cost Estimate

- **Vision Model**: FREE (runs locally)
- **LLM (GPT-5-mini)**: ~$0.001-0.002 per report
- **Testing (10 reports)**: ~$0.01-0.02

💡 $5 of credits = ~2,500-5,000 test reports

---

## 🎨 Try These Examples

### Generate report for normal X-ray:
```bash
python -c "
from src.pipeline import ReportGenerationPipeline
p = ReportGenerationPipeline()
r = p.generate_report('data/raw/NORMAL/IM-0001-0001.jpeg', patient_id='P001')
p.print_report(r)
"
```

### Generate report for pneumonia case:
```bash
python -c "
from src.pipeline import ReportGenerationPipeline
p = ReportGenerationPipeline()
r = p.generate_report('data/raw/PNEUMONIA/person100_bacteria_482.jpeg', patient_id='P002')
p.print_report(r)
"
```

---

## 📚 What's Next?

After testing Week 2, you can:

1. **Review** generated reports in `data/reports/generated/`
2. **Experiment** with different images
3. **Adjust** prompts in `config.py`
4. **Proceed to Week 3**: RAG enhancement + Voice input

---

## 💬 Need Help?

- Check [WEEK2_SUMMARY.md](WEEK2_SUMMARY.md) for detailed documentation
- Review [config.py](config.py) for all settings
- Check logs for error messages
- Test components individually before full pipeline

---

**Ready to test? Start with:**
```bash
source venv/bin/activate
python test_week2_pipeline.py
```

Good luck! 🚀
