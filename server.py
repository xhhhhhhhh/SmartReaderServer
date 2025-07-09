from flask import Flask, request, jsonify
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai import Agent
from dotenv import load_dotenv
import threading
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import google.generativeai as genai
from threading import Event

app = Flask(__name__)
load_dotenv()

# Shared state: each txt_id has an Agent, history, and a readiness flag
sessions: dict[str, threading.Event] = {}
histories = {}

# Constants
SYSTEM_PROMPT = "You are helping reader to understand the book."

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap=20,      # Overlap between chunks to preserve context
    separators=["\n\n", "\n", "。", "，"],  # Paragraphs are preferred first
)
chromadb_client = chromadb.PersistentClient("./chroma.db")
model = GeminiModel('gemini-2.5-flash')
agent = Agent(model, system_prompt="You are helping reader to understand the book")

def get_chunks(input_text) -> list[str]:
    return text_splitter.split_text(input_text)

def embed(text: str, store: bool) -> list[float]:
    result = genai.embed_content(content=text,
                        task_type="RETRIEVAL_DOCUMENT",
                        model="models/text-embedding-004")
    return result['embedding']

def collections_exist(txt_id: str) -> bool:
    collections = chromadb_client.list_collections()
    exists = any(c.name == txt_id for c in collections)
    print(f"Collection '{txt_id}' exists? {exists}")
    return  exists

def preload_text_async(txt_id: str, full_text: str):
    try:
        print("start init")
        exists = collections_exist(txt_id)
        if not exists:
            sessions[txt_id] = threading.Event()
            chromadb_collection = chromadb_client.get_or_create_collection(txt_id)
            chunks = get_chunks(full_text)
            for index in range(0, len(chunks)):
                print(f"init in progress {index}/{len(chunks)}")
                embedding = embed(chunks[index], True)
                chromadb_collection.upsert(ids=[str(index)], documents=[chunks[index]], embeddings=[embedding])
            print("complete init")
            sessions[txt_id].set()

        print(f"[✓] Preload complete for txt_id: {txt_id}")
    except Exception as e:
        print(f"[✗] Failed to preload for txt_id: {txt_id} → {e}")

def query_db(txt_id: str, question: str) -> list[str]:
    exists = collections_exist(txt_id)
    if not exists:
        return []
    chromadb_collection = chromadb_client.get_or_create_collection(txt_id)
    question_embedding = embed(question, store=False)
    result = chromadb_collection.query(
        query_embeddings=question_embedding,
        n_results=5
    )
    assert result["documents"]
    return result["documents"][0]

@app.route("/init", methods=["POST"])
def init_text():
    data = request.get_json()
    txt_id = data.get("txt_id")
    full_text = data.get("full_text")
    histories[txt_id] = []

    if not txt_id or not full_text:
        return jsonify({"error": "Missing txt_id or full_text"}), 400

    # Start background thread
    threading.Thread(target=preload_text_async, args=(txt_id, full_text), daemon=True).start()
    return jsonify({"status": "initializing"})

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        txt_id = data.get("txt_id")
        page_text = data.get("text", "")
        question = data.get("question", "")

        if not collections_exist(txt_id):
            return jsonify({"gemini not ready"})
        if txt_id not in sessions:
            return jsonify({"error": f"No session initialized for txt_id: {txt_id}"}), 404

        # Wait for preload if still in progress
        sessions[txt_id].wait()

        prompt = "Please answer user's question according to context\n"
        prompt += f"Question: {question}\n"
        prompt += f"Current page: {page_text}\n"
        prompt += "Context:\n"
        chunks = query_db(txt_id, question)
        for c in chunks:
            prompt += f"{c}\n"
            prompt += "-------------\n"
        print(f"\nprompt is {prompt}\n")
        answer = agent.run_sync(prompt, message_history=histories[txt_id])
        histories[txt_id] = list(answer.all_messages())
        print(f"answer is {answer}")
        return jsonify({"answer": answer.output})
    except Exception as e:
        print(f"[✗] Error in /ask → {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "sessions": list(sessions.keys())
    })

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    app.run(host='0.0.0.0', port=8080, debug=True)