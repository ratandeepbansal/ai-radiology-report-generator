# Quick Setup Guide

## ✅ Environment Setup Complete!

Your MedAssist Copilot development environment has been successfully set up.

## 📁 Project Structure

```
medassist-copilot/
├── data/
│   ├── raw/              ✅ Created
│   ├── processed/        ✅ Created
│   └── reports/          ✅ Created
├── models/               ✅ Created
├── src/                  ✅ Created
├── venv/                 ✅ Virtual environment created
├── .env.example          ✅ Environment template
├── .gitignore            ✅ Git ignore rules
├── config.py             ✅ Configuration file
├── requirements.txt      ✅ Dependencies list
└── README.md             ✅ Documentation
```

## 🚀 Next Steps

### 1. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** This may take 5-10 minutes depending on your internet connection.

### 3. Set Up API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your favorite editor
nano .env  # or code .env, vim .env, etc.
```

**Required API Keys:**
- **OpenAI API Key**: Get from https://platform.openai.com/api-keys
- **Hugging Face Token** (optional): Get from https://huggingface.co/settings/tokens

Example `.env` file:
```env
OPENAI_API_KEY=sk-proj-...your-key-here
HUGGINGFACE_API_TOKEN=hf_...your-token-here
USE_GPU=false
DEBUG=false
```

### 4. Verify Configuration

```bash
python config.py
```

This will validate your configuration and show which models will be used.

### 5. Test Installation

```bash
python -c "import torch; import transformers; import streamlit; print('✅ All core dependencies installed!')"
```

## 📋 Pre-Project Setup Checklist

Based on the task breakdown, here's what's done and what's next:

### Environment Setup ✅
- [x] Create project directory structure
- [x] Set up Python virtual environment (Python 3.13.5)
- [x] Create requirements.txt with core dependencies
- [x] Create config.py with configuration
- [x] Create .gitignore and .env.example

### Account Setup 🔄 (Your Action Required)
- [X] Create OpenAI API account and obtain API key
  - Visit: https://platform.openai.com/signup
  - Create API key: https://platform.openai.com/api-keys
  - Add credits to your account

- [X] Set up Hugging Face account for model access
  - Visit: https://huggingface.co/join
  - Get token: https://huggingface.co/settings/tokens

- [ ] Create Streamlit secrets file for API keys (for deployment)
  - Will be done when deploying to Streamlit Cloud

- [X] Set up Git repository for version control
  ```bash
  git init
  git add .
  git commit -m "Initial commit: Project setup"
  ```

## 🎯 Week 1 Tasks (Next)

Once setup is complete, you'll start Week 1:

1. **Dataset Acquisition**
   - Download Chest X-Ray Pneumonia Dataset from Kaggle
   - Explore BLIP-MIMIC-CXR dataset

2. **Data Preprocessing**
   - Create `data_loader.py` module
   - Implement image loading and preprocessing

3. **Report Database Setup**
   - Create sample patient reports JSON file
   - Set up RAG database

## 🐛 Troubleshooting

### Issue: pip install fails
**Solution:** Upgrade pip first
```bash
pip install --upgrade pip setuptools wheel
```

### Issue: PyAudio installation fails
**Solution:** Install system dependencies first

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

### Issue: CUDA/GPU not detected
**Solution:** Install PyTorch with CUDA support
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: ImportError for specific package
**Solution:** Install missing dependencies
```bash
pip install <package-name>
```

## 📚 Resources

- **OpenAI API Docs**: https://platform.openai.com/docs
- **Transformers Docs**: https://huggingface.co/docs/transformers
- **Streamlit Docs**: https://docs.streamlit.io
- **BLIP-2 Model**: https://huggingface.co/Salesforce/blip-image-captioning-base
- **Whisper Model**: https://github.com/openai/whisper

## 💡 Tips

1. **Start Small**: Test each component individually before integration
2. **Monitor Costs**: Keep an eye on OpenAI API usage
3. **Version Control**: Commit often with clear messages
4. **Documentation**: Document as you build, not at the end
5. **Testing**: Create test cases for each module

## ✨ Ready to Start?

Once you've completed steps 1-4 above, you're ready to begin Week 1 development!

Run this command to verify everything is working:
```bash
python config.py
```

If you see "Configuration validated successfully!", you're all set! 🎉

---

**Need Help?** Refer to the main [README.md](README.md) for detailed documentation.
