# Quick Setup Guide

## Prerequisites
1. Install Python 3.8 or higher
2. Install Ollama from https://ollama.ai

## Step-by-Step Setup

### 1. Install Ollama and Download Model
```bash
# After installing Ollama
ollama pull qwen:4b
```

### 2. Clone and Setup Project
```bash
# Clone repository
git clone https://github.com/yourusername/dual-mode-rag-chatbot.git
cd dual-mode-rag-chatbot

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Add Your Documents
```bash
# Create documents folder
mkdir documents

# Copy your PDF file to the documents folder
cp /path/to/your/document.pdf documents/
```

### 4. Update Configuration
Open `main.py` and update the PDF path:
```python
PDF_PATH = "documents/your-document.pdf"
```

### 5. Run the Application
```bash
python main.py
```

The Gradio interface will launch automatically. Open the URL shown in the terminal (usually http://localhost:7860)

## Testing the System

### Test Text Input
1. Go to the "Text Input" tab
2. Type: "What is this document about?"
3. Click "Ask"

### Test Voice Input
1. Go to the "Voice Input" tab
2. Allow microphone access in your browser
3. Click the microphone icon and speak your question
4. Click "Ask"

### Understanding Responses
- Responses tagged with "🧭 Answer Type: RAG" come from your documents
- Responses tagged with "🧭 Answer Type: LLM" come from general knowledge
- Page numbers are shown when the answer comes from RAG

## Common Issues

### Issue: Ollama Connection Error
**Solution**: Make sure Ollama is running
```bash
ollama serve
```

### Issue: Audio Not Working
**Solution**: 
- Check browser microphone permissions
- Try using Chrome or Firefox
- Ensure your microphone is properly connected

### Issue: Slow First Query
**Solution**: This is normal - the embedding model loads on first use. Subsequent queries will be faster.

### Issue: "Model not found"
**Solution**: Pull the model again
```bash
ollama pull qwen:4b
```

## Customization Tips

### Use a Different PDF
Just update the `PDF_PATH` variable in `main.py`

### Change the LLM Model
Update these variables in `main.py`:
```python
LLM_MODEL = "llama2:7b"  # or any other Ollama model
```

### Adjust Retrieval Settings
Modify these in `main.py`:
```python
CHUNK_SIZE = 1500  # Larger chunks for more context
TOP_K_RETRIEVAL = 6  # Retrieve more documents
```

## Next Steps
- Add multiple PDFs by loading them all before creating the vectorstore
- Experiment with different embedding models
- Try different Ollama models for various use cases
- Adjust temperature for more creative or focused responses

## Getting Help
- Check the main README.md for detailed documentation
- Open an issue on GitHub
- Review Ollama documentation: https://ollama.ai/docs
