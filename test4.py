from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
import os

from dotenv import load_dotenv

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("langchainvector")

app = Flask(__name__)
CORS(app)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def retrive_result(query,k=2):
    vectore=embeddings.embed_query(query)
    matchhing_result=index.query(vector=vectore,top_k=k, namespace="example-namespace", include_metadata=True)
    return matchhing_result

def answer_user_question(query):
    client = Groq(api_key=groq_api_key)
    results = retrive_result(query, k=4)

    context = "\n".join([match["metadata"]["text"] for match in results["matches"]])

    suggested_questions = []

    for match in results["matches"]:
        qs = match["metadata"].get("question", [])

        if isinstance(qs, list):
            suggested_questions.extend(qs)
        else:
            suggested_questions.append(qs)

    suggested_questions = list(set(suggested_questions))[:3]

    response = client.chat.completions.create(
    model="llama-3.1-8b-instant",  # fast + cheap
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer only from the given context."
        },
        {
            "role": "user",
            "content": f"""
        Context:
        {context}

        Question:
        {query}
        """
                }
            ],
            temperature=0
        )
    return response.choices[0].message.content,suggested_questions

@app.route('/', methods=['GET'])
def welcome():
    return render_template('chat.html')

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    
    query = data.get("question", "")

    answer, suggested_questions = answer_user_question(query)

    return jsonify({
        "answer": answer,
        "suggested_questions": suggested_questions
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)