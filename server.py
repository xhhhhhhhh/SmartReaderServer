from flask import Flask, request, jsonify
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai import Agent
from dotenv import load_dotenv
import threading
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import google.generativeai as genai
from flask import request

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

@app.before_request
def log_request_info():
    print(f"\n[⇩] Incoming {request.method} {request.path}")
    print("Headers:", dict(request.headers))
    if request.is_json:
        print("JSON:", request.get_json(silent=True))
    else:
        print("Body:", request.get_data(as_text=True))

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

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        txt_id = data.get("txt_id")
        page_text = data.get("text", "")
        question = data.get("question", "")

        if not collections_exist(txt_id):
            return jsonify({"error": "gemini not ready"}), 404

        prompt = "Please answer user's question according to context\n"
        prompt += f"Question: {question}\n"
        prompt += f"Current page: {page_text}\n"
        prompt += "Context:\n"
        chunks = query_db(txt_id, question)
        for c in chunks:
            prompt += f"{c}\n"
            prompt += "-------------\n"
        print(f"\nprompt is {prompt}\n")
        if txt_id not in histories:
            histories[txt_id] = []
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

def process_page_worker(page_md5: str, page_text: str, txt_id: str):
    try:
        if collections_exist(page_md5):
            print(f"[•] Page {page_md5} already processed")
            return

        chromadb_collection = chromadb_client.get_or_create_collection(txt_id)
        chunks = get_chunks(page_text)
        for index, chunk in enumerate(chunks):
            embedding = embed(chunk, store=True)
            chromadb_collection.upsert(
                ids=[txt_id+str(index)],
                documents=[chunk],
                embeddings=[embedding]
            )
        print(f"[✓] Page {page_md5} processed successfully")
    except Exception as e:
        print(f"[✗] Error in process_page_worker → {e}")

@app.route("/process_page", methods=["POST"])
def process_page():
    try:
        data = request.get_json()
        page_md5 = data.get("page_md5")
        page_text = data.get("text")
        txt_id = data.get("txt_id")

        if not page_md5 or not page_text:
            return jsonify({"error": "Missing page_md5 or text"}), 400

        threading.Thread(target=process_page_worker, args=(page_md5, page_text, txt_id), daemon=True).start()
        return jsonify({"status": "processing in background"})
    except Exception as e:
        print(f"[✗] Error in /process_page → {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    app.run(host='0.0.0.0', port=8080, debug=True)