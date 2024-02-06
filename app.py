from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from flask import Flask, request, jsonify
from tempfile import NamedTemporaryFile

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = "api_key"

@app.route("/api/query", methods=["POST"])
def query():
    file = request.files["pdf_file"]
    content = file.read()
    file.close()

    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content)
        tmp.seek(0)
        pdf_path = tmp.name

    store_path = os.path.splitext(os.path.basename(pdf_path))[0]

    embeddings = OpenAIEmbeddings()

    if os.path.exists(f"embeddings/{store_path}"):
        new_db = FAISS.load_local(f"embeddings/{store_path}", embeddings)
    else:
        pdfreader = PdfReader(pdf_path)
        raw_text = ''
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text += content

        text_splitter = CharacterTextSplitter(
           separator = "\n",
           chunk_size = 512,
           chunk_overlap  = 20,
           length_function = len,
        )
        texts = text_splitter.split_text(raw_text)

        db = FAISS.from_texts(texts, embeddings)
        db.save_local(f"embeddings/{store_path}")
        new_db = FAISS.load_local(f"embeddings/{store_path}", embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    docs = new_db.similarity_search(request.form["query"])
    result = chain.run(input_documents=docs, question=request.form["query"])

    return jsonify({"answer": result})

if __name__ == "__main__":
    app.run(debug=True)
