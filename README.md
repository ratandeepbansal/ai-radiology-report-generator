# MedAssist Copilot 🏥

An AI-powered radiology report generator that processes chest X-rays and generates structured reports using multimodal LLMs with RAG (Retrieval-Augmented Generation) for contextual accuracy.

## 🌟 Features

- **Multimodal AI Analysis**: Uses BLIP-2 vision model to analyze chest X-rays
- **Intelligent Report Generation**: Leverages GPT-4o-mini to create structured radiology reports
- **RAG Integration**: Retrieves relevant prior patient reports for contextual accuracy
- **Voice Dictation**: Whisper-powered voice input for hands-free report editing
- **Professional UI**: Clean, intuitive Streamlit interface for radiologists
- **Structured Output**: Generates reports with Findings, Impression, and Recommendations sections

## 🏗️ Architecture

```
medassist-copilot/
├── data/
│   ├── raw/              # Raw X-ray images
│   ├── processed/        # Preprocessed images
│   └── reports/          # Patient reports database
├── models/               # Downloaded model files
├── src/                  # Source code modules
│   ├── vision.py         # Vision model integration
│   ├── rag.py            # RAG system implementation
│   ├── llm_processor.py  # LLM processing
│   ├── audio_processor.py # Voice input handling
│   └── data_loader.py    # Data loading utilities
├── app.py                # Main Streamlit application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- (Optional) GPU with CUDA for faster processing

### Installation

1. **Clone the repository**
   ```bash
   cd medassist-copilot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv

   # On macOS/Linux:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env and add your API keys
   # OPENAI_API_KEY=your_openai_api_key_here
   # HUGGINGFACE_API_TOKEN=your_huggingface_token_here
   ```

5. **Verify configuration**
   ```bash
   python config.py
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage

1. **Upload X-ray**: Click the file uploader to select a chest X-ray image
2. **Enter Patient Info**: Fill in patient ID, age, gender, and date
3. **Generate Report**: Click "Generate Report" to process the image
4. **Review & Edit**: Review the generated report and make edits if needed
5. **Voice Input**: Use the microphone button to dictate additional findings
6. **Save Report**: Save the final report to the database

## 🔧 Configuration

Edit `config.py` to customize:

- **Model Selection**: Choose different vision models, LLMs, or embedding models
- **Prompt Templates**: Modify report generation prompts
- **System Parameters**: Adjust image sizes, token limits, temperature, etc.
- **Feature Flags**: Enable/disable RAG, voice input, or other features

## 🗄️ Dataset Setup

### Chest X-ray Dataset

1. Download the Chest X-Ray Pneumonia Dataset from Kaggle:
   - [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
   - Place images in `data/raw/`

2. (Optional) Download BLIP-MIMIC-CXR dataset for training:
   - Search on Hugging Face for multimodal chest X-ray datasets
   - Used for fine-tuning vision models

### Sample Reports Database

Create sample patient reports in `data/reports/patient_reports.json`:

```json
[
  {
    "patient_id": "P001",
    "date": "2024-01-15",
    "report": {
      "findings": "Clear lung fields. Normal cardiac silhouette.",
      "impression": "No acute cardiopulmonary abnormality.",
      "recommendations": "Routine follow-up."
    },
    "metadata": {
      "age": 45,
      "gender": "M"
    }
  }
]
```

## 🧪 Development Status

### ✅ Completed
- [x] Project structure setup
- [x] Configuration management
- [x] Requirements specification

### 🔄 In Progress
- [ ] Vision model integration
- [ ] LLM processor implementation
- [ ] RAG system setup
- [ ] Voice input processing
- [ ] Streamlit UI development

### 📋 Planned
- [ ] Report evaluation metrics
- [ ] Performance optimization
- [ ] Cloud deployment
- [ ] Fine-tuning on medical datasets

## 📊 Tech Stack

- **Frontend**: Streamlit
- **Vision**: BLIP-2, CLIP, or LLaVA
- **LLM**: GPT-4o-mini (OpenAI) or Llama-3 8B
- **Embeddings**: Sentence Transformers
- **Vector DB**: ChromaDB or FAISS
- **Voice**: OpenAI Whisper
- **Framework**: PyTorch, Transformers

## 🔐 Security & Privacy

- All API keys stored in `.env` file (never committed to git)
- Patient data anonymization options available in config
- HIPAA compliance considerations documented
- Secure handling of sensitive medical information

## ⚠️ Disclaimer

**This is an educational/research project and NOT intended for clinical use.**

This application is designed for learning purposes and academic projects. It should NOT be used for actual medical diagnosis or patient care. Always consult qualified healthcare professionals for medical decisions.

## 📝 License

This project is for educational purposes. Please ensure compliance with medical data regulations (HIPAA, GDPR) when handling patient information.

## 🤝 Contributing

This is a capstone project, but suggestions and feedback are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📧 Contact

For questions or feedback about this project, please open an issue on GitHub.

## 🙏 Acknowledgments

- BLIP-2 by Salesforce Research
- OpenAI for GPT models and Whisper
- Hugging Face for model hosting and transformers library
- Streamlit for the UI framework
- Medical imaging datasets from Kaggle and MIMIC-CXR

## 📚 References

- [BLIP-2: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2301.12597)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Whisper: Robust Speech Recognition](https://arxiv.org/abs/2212.04356)
- [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/)

---

**Built with ❤️ for advancing AI in healthcare**
