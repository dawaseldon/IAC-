# app.py
import streamlit as st
from ollama import Client
import chromadb
from chromadb.api.models.Collection import Collection
import fitz
import uuid


# -------------------------------------------------
# 1️⃣ Ollama client
# -------------------------------------------------
OLLAMA_HOST = "https://epigamic-migdalia-sporogonial.ngrok-free.dev"
client = Client(host=OLLAMA_HOST)


# -------------------------------------------------
# 2️⃣ Chroma client & PDF collection
# -------------------------------------------------
chroma_client = chromadb.Client()
COLLECTION_NAME = "grade10_history"
try:
    collection: Collection = chroma_client.get_collection(name=COLLECTION_NAME)
except Exception:
    collection = chroma_client.create_collection(name=COLLECTION_NAME)


# -------------------------------------------------
# 3️⃣ Embedding Function
# -------------------------------------------------
def get_embedding(text: str):
    response = client.embeddings(model="nomic-embed-text:v1.5", prompt=text)
    return response["embedding"]


# -------------------------------------------------
# 4️⃣ PDF Loader & Chunking
# -------------------------------------------------
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


# -------------------------------------------------
# 5️⃣ PDF ChatBot class
# -------------------------------------------------
class PDFChatBot:
    def __init__(self, collection, client, top_k=6):
        self.collection = collection
        self.client = client
        self.top_k = top_k
        self.prev_qas = []  # [(question, answer)]
        self.used_contexts = set()


    def retrieve_context(self, query: str, exclude_texts=None):
        if exclude_texts is None:
            exclude_texts = []


        query_embedding = get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )
        docs = results["documents"][0]
        fresh_docs = [d for d in docs if d not in exclude_texts and d not in self.used_contexts]
        if not fresh_docs:
            fresh_docs = docs
        self.used_contexts.update(fresh_docs)
        return "\n".join(fresh_docs)


    def ask(self, question: str):
        lower_q = question.lower()


        # ----- Topic-based elaboration -----
        if lower_q.startswith("elaborate on"):
            topic = question[len("elaborate on"):].strip()
            context = self.retrieve_context(topic, exclude_texts=[a for _, a in self.prev_qas])
            prompt = f"Elaborate in depth on the topic: '{topic}' using the following context from the PDF:\n\n{context}"
            answer = self._chat(prompt)
            self.prev_qas.append((question, answer))
            return answer


        # ----- Summarize previous answer -----
        if "summarize" in lower_q and self.prev_qas:
            last_answer = self.prev_qas[-1][1]
            prompt = f"Summarize the following answer clearly and concisely:\n\n{last_answer}"
            answer = self._chat(prompt)
            self.prev_qas.append((question, answer))
            return answer


        # ----- Prepare exam questions -----
        if "prepare 5 exam questions" in lower_q:
            context = self.retrieve_context("important key topics from the textbook")
            prompt = f"Using the context below, generate 5 exam-ready questions with detailed answers. Answers must be structured for board exams.\n\n{context}"
            answer = self._chat(prompt)
            self.prev_qas.append((question, answer))
            return answer


        # ----- Normal question -----
        context = self.retrieve_context(question)
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        answer = self._chat(prompt)
        self.prev_qas.append((question, answer))
        return answer


    def _chat(self, user_prompt: str):
        system_prompt = (
            "You are an academic assistant for a Grade 10 Bhutan History, Civics and Citizenship textbook. "
            "Answer strictly using the PDF context. Provide structured, exam-ready answers. "
            "Do NOT mention chunks or sources. If info is missing, say: 'I don't know.'"
        )


        messages = [{"role": "system", "content": system_prompt}]
        for q, a in self.prev_qas:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_prompt})


        response = self.client.chat(model="llama3.1:8b", messages=messages)
        return response["message"]["content"]


# -------------------------------------------------
# 6️⃣ Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Grade 10 History PDF ChatBot", layout="wide")
st.title("📚 Grade 10 History PDF ChatBot")
st.write("Ask questions from your textbook PDF. Type 'elaborate on [topic]', 'summarize', or 'prepare 5 exam questions'.")


if "bot" not in st.session_state:
    st.session_state.bot = PDFChatBot(collection, client, top_k=6)


user_input = st.text_input("You:", "")
if user_input:
    answer = st.session_state.bot.ask(user_input)
    st.markdown(f"**Assistant:**\n\n{answer}")