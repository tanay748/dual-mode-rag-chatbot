pip install sentence-transformers
pip install pypdf
pip install faiss-cpu
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen:4b", temperature=0.2)
response = llm.invoke("What is electricity?")
print(response.content)
loader = PyPDFLoader("/Users/tanaysinghchauhan/Downloads/RAGReasearchPaper.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
vectorstore = FAISS.from_documents(splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

print("Vectorstore is ready")
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

Condense_Question_Prompt = PromptTemplate.from_template("""
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}
Follow up Input: {question}
Standalone question:
""")

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,   
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),  
    condense_question_prompt=Condense_Question_Prompt,
    return_source_documents=True,
    verbose=False
)
from langchain.prompts import PromptTemplate

CONTEXT_PROMPT = PromptTemplate.from_template("""
You are a research assistant. Use ONLY the following retrieved documents to answer the question. 
Do not answer from general knowledge. If the answer is not found in the documents, say "I don't know".

Context:
{context}

Question:
{question}
Answer:
""")
chat_history = []
query = input("Enter your query:")

result = qa.invoke({"question": query, "chat_history": chat_history})

print("Answer:\n", result["answer"].strip())

chat_history.append((query, result["answer"]))chat_history = []
query = input("Enter your query:")

result = qa.invoke({"question": query, "chat_history": chat_history})

print("Answer:\n", result["answer"].strip())

chat_history.append((query, result["answer"]))
# --- 🔄 Rerouting instead of classification ---
from langchain.prompts import PromptTemplate

# Router prompt to decide between RAG or LLM
router_prompt = PromptTemplate.from_template("""
Decide how to answer the following user query.

Query: {query}

Options:
- "RAG" if it requires information from the provided documents.
- "LLM" if it can be answered from general knowledge or casual conversation.

Answer only with "RAG" or "LLM".
""")

def route_query(query: str) -> str:
    """
    Uses the LLM to decide whether to use RAG or direct LLM.
    Returns either "RAG" or "LLM".
    """
    decision = llm.invoke(router_prompt.format(query=query)).content.strip()
    if decision not in ["RAG", "LLM"]:
        return "RAG"   # default fallback
    return decision


def chat(query, history):
    global chat_history

    # 🛑 Special handling for meta-questions about source
    if "rag" in query.lower() and "llm" in query.lower():
        if chat_history:
            last_answer = chat_history[-1][1]
            if "🧭 Answer Type: RAG" in last_answer:
                return "The last answer came from **RAG** (retrieved documents)."
            elif "🧭 Answer Type: LLM" in last_answer:
                return "The last answer came from the **LLM** (general knowledge)."
            else:
                return "I cannot determine the source of the last answer."
        else:
            return "No previous answer to check."

    # 🔄 Decide which path to use
    route = route_query(query)

    if route == "RAG":
        result = qa.invoke({"question": query, "chat_history": chat_history})
        answer = result["answer"]

        # optionally add sources
        sources = [doc.metadata.get("page_label", "") for doc in result.get("source_documents", [])]
        if sources:
            answer += f"\n\n📖 Sources: Pages {', '.join(sources)}"

        answer += "\n\n🧭 Answer Type: RAG"
    else:
        response = llm.invoke(query)
        answer = response.content
        answer += "\n\n🧭 Answer Type: LLM"

    # update chat history
    chat_history.append((query, answer))
    return answer
import gradio as gr

chat_history = []

def chat(input_data):
    global chat_history

    # Convert audio to text if input_data is not string
    if isinstance(input_data, str):
        query = input_data
    else:
        query = gr.audio_to_text(input_data)  # Gradio converts audio to text
        if not query:
            return "I couldn't understand you. Please try again."

    # Run RAG chain
    result = qa.invoke({"question": query, "chat_history": chat_history})
    answer = result["answer"]

    # Update chat history
    chat_history.append((query, answer))
    return answer

# Gradio UI using Tabs
with gr.Blocks() as demo:
    gr.Markdown("## 🎤📚 Dual-mode RAG Chatbot")

    with gr.Tabs():
        with gr.TabItem("Text Input"):
            text_input = gr.Textbox(label="Type your question")
            text_btn = gr.Button("Ask")
        with gr.TabItem("Voice Input"):
            audio_input = gr.Audio(label="Speak your question", type="filepath")
            audio_btn = gr.Button("Ask")
    
    output = gr.Textbox(label="Answer")

    # Link buttons to chat function
    text_btn.click(chat, inputs=text_input, outputs=output)
    audio_btn.click(chat, inputs=audio_input, outputs=output)
demo.launch()
