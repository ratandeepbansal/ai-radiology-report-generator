# Running MedAssist Copilot 🚀

Quick guide to run the Streamlit application

---

## ✅ Prerequisites

Before running the app, make sure you have:

1. ✅ **Week 1-3 setup complete**
   - Virtual environment created
   - Dependencies installed
   - OpenAI API key configured (in `.env`)

2. ✅ **All required packages**
   ```bash
   pip install streamlit pandas
   ```

3. ✅ **Test X-ray images**
   - Located in `data/raw/NORMAL/` or `data/raw/PNEUMONIA/`
   - Or use your own X-ray images

---

## 🚀 Quick Start

### Option 1: Run Locally

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run Streamlit app
streamlit run app.py
```

The app will open in your browser at: **http://localhost:8501**

### Option 2: Run with Custom Port

```bash
streamlit run app.py --server.port 8502
```

### Option 3: Run in Background

```bash
nohup streamlit run app.py &
```

---

## 🎨 Using the Application

### Step 1: Patient Information (Sidebar)
1. Enter **Patient ID** (e.g., P001)
2. Enter **Age** and **Gender**
3. Select **Exam Date**
4. Add **Clinical Indication** (optional)

### Step 2: Upload X-Ray
1. Go to **"📸 Image Upload"** tab
2. Click **"Choose an X-ray image"**
3. Select a chest X-ray (JPG/PNG)
4. Image will display with information

### Step 3: Generate Report
1. Click **"🤖 Generate Report"** button
2. Wait for AI processing (~10-15 seconds)
   - Vision analysis: ~2-3s
   - RAG retrieval: ~0.1s
   - LLM generation: ~4-5s
3. Report appears automatically

### Step 4: Review & Edit Report
1. Go to **"📄 Report"** tab
2. View generated report sections:
   - **Findings**
   - **Impression**
   - **Recommendations**
3. Edit any section directly in text boxes
4. Click **"💾 Save Report"** to save

### Step 5: View Metrics
1. Go to **"📊 Evaluation"** tab
2. See performance breakdown
3. View processing times
4. Check token usage

---

## 🎯 Features

### ✅ Available Now
- ✅ X-ray image upload and display
- ✅ Patient information form
- ✅ AI-powered report generation
- ✅ Vision analysis (BLIP-2)
- ✅ LLM report writing (GPT)
- ✅ RAG prior report retrieval
- ✅ Report editing
- ✅ Report saving
- ✅ Performance metrics
- ✅ Dark mode UI with neon green accents

### 🔄 Coming Soon (if time permits)
- Voice input integration
- PDF export
- Multi-image comparison
- Report history
- Advanced search

---

## ⚙️ Configuration

### API Key Setup
Make sure your `.env` file has:
```env
OPENAI_API_KEY=sk-proj-your-key-here
```

### Model Settings
Edit `config.py` to change:
- Vision model (default: BLIP-2)
- LLM model (default: gpt-5-mini)
- RAG settings
- UI preferences

---

## 🐛 Troubleshooting

### Issue: "Failed to load pipeline"
**Solution**: Check your OpenAI API key
```bash
# Verify .env file
cat .env | grep OPENAI_API_KEY

# Make sure it starts with sk-
```

### Issue: "No module named 'streamlit'"
**Solution**: Install Streamlit
```bash
pip install streamlit
```

### Issue: App is slow
**Causes**:
- First run loads models (~2 minutes)
- CPU processing (consider GPU)
- Large images (resize before upload)

**Solutions**:
- Models are cached after first load
- Use smaller images (< 2MB)
- Be patient on first generation

### Issue: "RAG system failed"
**Solution**: Install ChromaDB
```bash
pip install chromadb sentence-transformers
```

### Issue: Port already in use
**Solution**: Use different port
```bash
streamlit run app.py --server.port 8502
```

---

## 📊 Performance Tips

1. **First Run**: Takes ~2 minutes to load models
2. **Subsequent Runs**: Much faster (~10-15s per report)
3. **Image Size**: Keep under 2MB for best performance
4. **GPU**: Use CUDA if available (edit config.py)
5. **Batch Processing**: Generate multiple reports in succession

---

## 🎨 UI Customization

### Theme
Edit the CSS in `app.py` to customize:
- Colors (neon green: #39FF14)
- Fonts
- Layout
- Dark mode settings

### Layout
Modify in `app.py`:
- Column widths
- Tab arrangement
- Sidebar content
- Component order

---

## 💡 Pro Tips

1. **Patient ID**: Use existing IDs (P001-P015) to see RAG in action
2. **Test Images**: Start with NORMAL images for quick testing
3. **Reset Button**: Click to start fresh
4. **Edit Reports**: Make changes directly in text areas
5. **Save Often**: Use Save button to preserve reports

---

## 📸 Example Workflow

```
1. Start app: streamlit run app.py
2. Enter Patient ID: P001
3. Upload: data/raw/NORMAL/IM-0001-0001.jpeg
4. Click: Generate Report
5. Wait: ~15 seconds
6. Review: Check Findings/Impression/Recommendations
7. Edit: Make any corrections
8. Save: Click Save Report
9. Done! Check data/reports/generated/
```

---

## 🔗 URLs

### Local Development
- **App**: http://localhost:8501
- **Docs**: http://localhost:8501/docs (if enabled)

### Network Access
If you want to access from other devices:
```bash
streamlit run app.py --server.address 0.0.0.0
```
Then access via: `http://your-ip:8501`

---

## 📝 Keyboard Shortcuts

While in Streamlit:
- `r` - Rerun the app
- `c` - Clear cache
- `q` - Quit (in terminal)

---

## 🎯 Next Steps

After running the app successfully:

1. **Test thoroughly** with different X-rays
2. **Try different patients** to see RAG working
3. **Edit and save** reports
4. **Check performance** metrics
5. **Share feedback** or report issues

---

## 🆘 Need Help?

- **Documentation**: Check README.md
- **Configuration**: See config.py
- **Week 4 Details**: Read WEEK4_COMPLETE.md
- **Logs**: Check terminal output
- **Issues**: Look for error messages

---

## ✨ Enjoy!

You've built a complete AI-powered radiology report generator!

**Remember**: This is for educational/research use only. Not for clinical diagnosis.

---

**Happy reporting!** 🏥🤖
