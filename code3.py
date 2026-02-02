# app.py
import streamlit as st
from ollama import Client
import chromadb
from chromadb.api.models.Collection import Collection
import fitz
import uuid

# ---------------------------
# 1️⃣ Ollama client
# ---------------------------
OLLAMA_HOST = "https://epigamic-migdalia-sporogonial.ngrok-free.dev"
client = Client(host=OLLAMA_HOST)

# ---------------------------
# 2️⃣ Chroma client & PDF collection
# ---------------------------
chroma_client = chromadb.Client()
COLLECTION_NAME = "grade10_history"
try:
    collection: Collection = chroma_client.get_collection(name=COLLECTION_NAME)
except Exception:
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

# ---------------------------
# 3️⃣ Embedding Function
# ---------------------------
def get_embedding(text: str):
    response = client.embeddings(model="nomic-embed-text:v1.5", prompt=text)
    return response["embedding"]

# ---------------------------
# 4️⃣ PDF Loader & Chunking
# ---------------------------
def load_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text: str, chunk_size: int = 1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def embed_pdf(path: str):
    text = load_pdf(path)
    chunks = chunk_text(text)
    for chunk in chunks:
        embedding = get_embedding(chunk)
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[chunk]
        )

# ---------------------------
# 5️⃣ PDF ChatBot class
# ---------------------------
class PDFChatBot:
    def __init__(self, collection, client, top_k=6):
        self.collection = collection
        self.client = client
        self.top_k = top_k

    def retrieve_context(self, query: str, prev_qas=None):
        if prev_qas is None:
            prev_qas = []
        query_embedding = get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )
        docs = results["documents"][0]
        used_texts = [a for _, a in prev_qas]
        fresh_docs = [d for d in docs if d not in used_texts]
        if not fresh_docs:
            fresh_docs = docs
        return "\n".join(fresh_docs)

    def ask(self, question: str, conversation_history):
        lower_q = question.lower()
        context = self.retrieve_context(question, conversation_history)

        if lower_q.startswith("elaborate on"):
            topic = question[len("elaborate on"):].strip()
            context = self.retrieve_context(topic, conversation_history)
            style_prompt = f"Elaborate on '{topic}' in detailed paragraphs with examples and context."
        elif "summarize" in lower_q:
            style_prompt = "Summarize the following clearly. Use bullets if requested, otherwise concise paragraphs."
        elif "prepare 5 exam questions" in lower_q:
            style_prompt = "Create 5 exam-ready questions with structured answers using bullets and numbering."
        else:
            style_prompt = "Answer in structured paragraphs with clarity, context, and examples."

        system_prompt = (
            "You are an expert academic assistant for Grade 10 Bhutan History. "
            "Answer using the provided context, expand naturally, and continue the conversation logically. "
            "Use paragraphs for elaboration and bullets only when requested. Reference previous Q&As."
        )

        messages = [{"role": "system", "content": system_prompt}]
        for q, a in conversation_history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": f"{style_prompt}\n\nContext:\n{context}\n\nQuestion: {question}"})

        response = self.client.chat(model="llama3.1:8b", messages=messages)
        return response["message"]["content"]

# ---------------------------
# 6️⃣ Streamlit UI
# ---------------------------
st.set_page_config(page_title="Grade 10 History PDF ChatBot", layout="wide")

# ---- Initialize session state ----
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "bot" not in st.session_state:
    st.session_state.bot = PDFChatBot(collection, client, top_k=6)

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# ---- Clean ChatGPT-style UI ----
st.markdown(
    """
    <style>
    /* Chat container */
    .chat-box {
        max-width: 800px;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        height: 70vh;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 8px;
        background-color: #F8F8F8;
    }
    /* Message bubbles */
    .chat-message {
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 8px;
        max-width: 75%;
        word-wrap: break-word;
    }
    .user { background-color: #DCF8C6; align-self: flex-end; }
    .assistant { background-color: #F1F0F0; align-self: flex-start; }
    /* Answer indicator */
    .answer-ready { color: green; font-weight: bold; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📚 Grade 10 History PDF ChatBot")
st.write(
    "Type your question below. Commands:\n"
    "- `elaborate on [topic]`\n"
    "- `summarize`\n"
    "- `prepare 5 exam questions`\n"
    "Example: 'tell me about the constitution'"
)

# ---- Scrollable chat container ----
chat_box = st.container()
with chat_box:
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for msg in st.session_state.conversation:
        role_class = "assistant" if msg["role"] == "assistant" else "user"
        st.markdown(f'<div class="chat-message {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg["role"] == "assistant":
            st.markdown('<div class="answer-ready">✅ Answer ready</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Input bar fixed at bottom, enter sends message ----
def send_message():
    if st.session_state.user_input.strip():
        answer = st.session_state.bot.ask(st.session_state.user_input, st.session_state.conversation)
        st.session_state.conversation.append({"role": "user", "content": st.session_state.user_input})
        st.session_state.conversation.append({"role": "assistant", "content": answer})
        st.session_state.user_input = ""  # Clear input

st.text_input(
    "Type your message here...",
    key="user_input",
    placeholder="Ask anything about Grade 10 History...",
    on_change=send_message
)
