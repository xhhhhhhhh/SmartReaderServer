from flask import Flask, request, jsonify
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai import Agent
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()
model = GeminiModel('gemini-2.5-flash')
agent = Agent(model,
              system_prompt="You are helping reader to understand the book")

@app.route("/message", methods=["POST"])
def receive_message():
    data = request.get_json()
    print("Received message:", data)
    return jsonify({"status": "ok", "echo": data.get("text", "")})

@app.route('/ask', methods=['POST'])
def handle_question():
    try:
        data = request.get_json()

        full_text = data.get("full_text", "")
        page_text = data.get("text", "")
        question = data.get("question", "")
        page = data.get("page", "Unknown")

        if not page_text or not question:
            return jsonify({"error": "Missing fields"}), 400

        prompt = f"""You are an AI assistant helping a user understand a book.

The user is currently reading **page {page}** of the book. This is the content of the current page:

--- Page Content ---
{page_text}
---------------------

The full book content is also available below (do not repeat it, but use it to reason if needed):

{full_text}

User's question: {question}

Please provide a clear answer and brief explanation using the current page primarily, and the full book if needed. Please Privode the answer and
explanation in Chinese, only use other language when reference to original text is needed.
"""

        response = agent.run_sync(prompt)
        answer = response.output

        return jsonify({"answer": answer})
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
