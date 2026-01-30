# 🎤📚 Dual-mode RAG Chatbot with Intelligent Rerouting

A sophisticated Retrieval-Augmented Generation (RAG) chatbot that intelligently routes queries between document-based retrieval and general knowledge, supporting both text and voice inputs through an interactive Gradio interface.

## 🌟 Features

### Core Capabilities
- **Intelligent Query Routing**: Automatically determines whether to use RAG (document retrieval) or direct LLM responses based on query context
- **Multi-modal Input**: Supports both text and voice queries via Gradio UI
- **Conversational Memory**: Maintains chat history for coherent multi-turn conversations
- **Fast Vector Retrieval**: Utilizes FAISS for efficient similarity search
- **Source Transparency**: Displays whether answers came from RAG or LLM, with page references

### Technical Highlights
- **Document Processing**: RecursiveCharacterTextSplitter for optimal chunking (1000 chars, 100 overlap)
- **Semantic Embeddings**: HuggingFace's `intfloat/e5-small-v2` model for high-quality embeddings
- **Local LLM**: Powered by Ollama's `qwen:4b` model with controlled temperature (0.2)
- **Top-k Retrieval**: Fetches 4 most relevant document chunks per query

## 🏗️ Architecture

```
User Query (Text/Voice)
    ↓
Query Router (LLM-based)
    ↓
┌───────────────┬───────────────┐
│   RAG Path    │   LLM Path    │
├───────────────┼───────────────┤
│ FAISS Retrieval│ Direct Answer │
│ Top-k Chunks  │ General       │
│ + Context     │ Knowledge     │
└───────────────┴───────────────┘
    ↓
Formatted Response + Source Tags
```

## 📋 Prerequisites

- Python 3.8+
- Ollama installed with `qwen:4b` model
- PDF document(s) for RAG knowledge base

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/dual-mode-rag-chatbot.git
cd dual-mode-rag-chatbot
```

2. **Install dependencies**
```bash
pip install sentence-transformers
pip install pypdf
pip install faiss-cpu
pip install langchain
pip install langchain-community
pip install langchain-ollama
pip install gradio
```

3. **Set up Ollama**
```bash
# Install Ollama from https://ollama.ai
ollama pull qwen:4b
```

## 💻 Usage

### Basic Setup

1. **Update PDF path** in the code:
```python
loader = PyPDFLoader("path/to/your/document.pdf")
```

2. **Run the application**:
```bash
python main.py
```

3. **Access the interface**: Open the Gradio URL (typically `http://localhost:7860`)

### Example Queries

**RAG-based queries** (answered from documents):
- "What is the main finding of the research paper?"
- "Summarize the methodology section"
- "What datasets were used in the study?"

**LLM-based queries** (general knowledge):
- "What is machine learning?"
- "Explain neural networks in simple terms"
- "Hello, how are you?"

**Meta queries**:
- "Was that answer from RAG or LLM?" - Check the source of the last response

## 🔧 Configuration

### Adjust Chunk Size
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Modify based on document complexity
    chunk_overlap=100   # Increase for better context preservation
)
```

### Change Retrieval Settings
```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}  # Adjust number of retrieved chunks
)
```

### Modify LLM Parameters
```python
llm = ChatOllama(
    model="qwen:4b",
    temperature=0.2  # Lower = more focused, Higher = more creative
)
```

## 📊 Project Structure

```
.
├── main.py                 # Main application file
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── documents/            # Store your PDF files here
    └── example.pdf
```

## 🎯 How It Works

1. **Document Preprocessing**: PDFs are loaded and split into manageable chunks
2. **Embedding Creation**: Text chunks are converted to vector embeddings using HuggingFace models
3. **Vector Storage**: Embeddings stored in FAISS index for fast similarity search
4. **Query Routing**: LLM analyzes each query to determine optimal response method
5. **Retrieval/Generation**: Either retrieves relevant chunks or generates direct answers
6. **Response Formatting**: Adds source tags and maintains conversation history

## 🛠️ Customization

### Using Different Models

**Change embedding model**:
```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

**Change LLM**:
```python
llm = ChatOllama(model="llama2:7b")  # or any Ollama model
```

### Adding Multiple Documents

```python
pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
all_docs = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    all_docs.extend(loader.load())
```

## 🔍 Features in Detail

### Conversational Retrieval Chain
- Maintains context across multiple turns
- Condenses follow-up questions into standalone queries
- Returns source documents with page numbers

### Intelligent Routing
- Analyzes query intent using LLM
- Defaults to RAG for document-specific questions
- Uses direct LLM for general knowledge and casual conversation
- Provides transparency with answer type tags

### Gradio Interface
- **Text Tab**: Type questions directly
- **Voice Tab**: Speak questions using microphone
- Real-time transcription and response
- Clean, intuitive UI

## 📈 Performance Considerations

- **FAISS**: CPU-optimized for single-machine deployment
- **Embedding Cache**: First query slower due to model loading
- **Voice Processing**: Requires microphone permissions in browser
- **Memory**: Chat history grows with conversation length

## 🐛 Troubleshooting

**Ollama connection issues**:
```bash
# Ensure Ollama is running
ollama serve
```

**Audio not working**:
- Check browser microphone permissions
- Ensure audio input device is available

**Slow responses**:
- Reduce `k` value in retriever
- Use smaller embedding model
- Decrease chunk_size

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- LangChain for the RAG framework
- Ollama for local LLM deployment
- Gradio for the intuitive UI
- HuggingFace for embedding models
- FAISS for efficient vector search

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: Update the PDF path in `main.py` before running. This chatbot works best with domain-specific documents like research papers, technical manuals, or documentation.
