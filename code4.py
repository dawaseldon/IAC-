# app.py
import streamlit as st
from ollama import Client
import chromadb
from chromadb.api.models.Collection import Collection
import fitz
import uuid
import os


# ---------------------------
# 1️⃣ Ollama client (LOCAL)
# ---------------------------
OLLAMA_HOST = "https://epigamic-migdalia-sporogonial.ngrok-free.dev"
client = Client(host=OLLAMA_HOST)


# ---------------------------
# 2️⃣ Chroma DB
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
    response = client.embeddings(
        model="nomic-embed-text:v1.5",
        prompt=text
    )
    return response["embedding"]


# ---------------------------
# 4️⃣ PDF Loader
# ---------------------------
PDF_PATH = "X-Bhutan-History-Civics-Citizenship-Edu.-2024.pdf"


def load_pdf(path: str) -> str:
    if not os.path.exists(path):
        return ""
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text: str, chunk_size: int = 1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def embed_pdf_once():
    if collection.count() > 0:
        return  # prevent duplicate embeddings


    text = load_pdf(PDF_PATH)
    if not text:
        return


    chunks = chunk_text(text)
    for chunk in chunks:
        embedding = get_embedding(chunk)
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[chunk]
        )


embed_pdf_once()


# ---------------------------
# 5️⃣ ChatBot Class
# ---------------------------
class PDFChatBot:
    def __init__(self, collection, client, top_k=6):
        self.collection = collection
        self.client = client
        self.top_k = top_k


    def retrieve_context(self, query: str):
        query_embedding = get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )
        docs = results["documents"][0]
        return "\n".join(docs)


    def ask(self, question: str, conversation_history):
        lower_q = question.lower()


        # ---- Style Commands ----
        if lower_q.startswith("elaborate on"):
            style_prompt = "Elaborate in detailed structured paragraphs with examples."
        elif "summarize" in lower_q:
            style_prompt = "Summarize clearly and concisely."
        elif "prepare 5 exam questions" in lower_q:
            style_prompt = "Generate 5 board-exam style questions with structured answers."
        else:
            style_prompt = "Answer clearly in structured paragraphs suitable for board exams."


        context = self.retrieve_context(question)


        system_prompt = (
            "You are an academic assistant for Grade 10 Bhutan History. "
            "Answer ONLY using the provided context. "
            "Do NOT mention sources or chunks. "
            "If missing, say: 'I don't know.'"
        )


        messages = [{"role": "system", "content": system_prompt}]


        # Add previous conversation
        for msg in conversation_history:
            messages.append(msg)


        messages.append({
            "role": "user",
            "content": f"{style_prompt}\n\nContext:\n{context}\n\nQuestion: {question}"
        })


        response = self.client.chat(
            model="llama3.1:8b",
            messages=messages
        )


        return response["message"]["content"]


# ---------------------------
# 6️⃣ Streamlit UI (Clean Chat Style)
# ---------------------------
st.set_page_config(page_title="Grade 10 History ChatBot", layout="wide")


st.markdown("""
<style>
.chat-container {
    max-width: 850px;
    margin: auto;
}
.chat-bubble {
    padding: 12px 16px;
    border-radius: 15px;
    margin-bottom: 10px;
    max-width: 75%;
}
.user {
    background-color: #DCF8C6;
    margin-left: auto;
}
.assistant {
    background-color: #F1F0F0;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)


st.title("📚 Grade 10 History PDF ChatBot")


# Session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []


if "bot" not in st.session_state:
    st.session_state.bot = PDFChatBot(collection, client)


# Chat Display
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.conversation:
    role_class = "assistant" if msg["role"] == "assistant" else "user"
    st.markdown(
        f'<div class="chat-bubble {role_class}">{msg["content"]}</div>',
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)


# Input box
def send_message():
    user_input = st.session_state.user_input.strip()
    if user_input:
        answer = st.session_state.bot.ask(
            user_input,
            st.session_state.conversation
        )


        st.session_state.conversation.append(
            {"role": "user", "content": user_input}
        )
        st.session_state.conversation.append(
            {"role": "assistant", "content": answer}
        )


        st.session_state.user_input = ""


st.text_input(
    "Ask anything about Grade 10 History...",
    key="user_input",
    on_change=send_message
)
