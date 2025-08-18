import tempfile
import gc
from flask import Flask, jsonify, request
import torch
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import re
import os
from langchain.document_loaders import TextLoader
from langchain_chroma import Chroma

embedding_function = SentenceTransformerEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B"
)

app = Flask(__name__)

# Home page
@app.route("/")
def home():
    return "<h1>Welcome to RAG App</h1>"

@app.route("/api/file_upload", methods=["POST"])
def file_upload():
    try:
        # Get log_id from form data
        log_id = request.form.get("log_id")
        if not log_id:
            return jsonify({"error": "Missing log_id"}), 400

        # Get file from request
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Save file temporarily
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        # Load document
        loader = TextLoader(file_path)
        documents = loader.load()

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)

        # Clean up temp file
        os.remove(file_path)
        #document_embeddings = embedding_function.embed_documents([split.page_content for split in splits])
        collection_name = "collection"+log_id
        Chroma.from_documents(
            collection_name=collection_name,
            documents=splits,
            embedding=embedding_function,
            persist_directory="./chroma_db"
        )
        del loader, documents, text_splitter, splits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return jsonify({
            "log_id": log_id,
            "collection_created":collection_name,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def query_vector_db():
    try:
        log_id = request.json.get("log_id")
        question = request.json.get("query")

        if not log_id or not question:
            return jsonify({"error": "Missing log_id or query"}), 400

        # Load vectorstore for this log_id
        collection_name = "collection" + log_id
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory="./chroma_db",
            embedding_function=embedding_function
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        # LLM config
        llm = ChatOpenAI(
            base_url="https://yykofxr8k5k5ic-8000.proxy.runpod.net/v1",   # llama.cpp server
            api_key="389e5f28-62d0-46c6-9cbc-0099da90ff30",
            model="gpt_oss_20b",
            temperature=1,
        )

        # Prompt template
        template = """
        When the user asks about the "patient", always interpret it as the "subscriber" and answer using subscriber details from the context.
If the information is not in the context, state that it is unavailable and never invent data dont be too long be shorter that human can understand.
        Answer the question based only on the following context:
        {context}
        Question: {question}
        Answer: """
        prompt = ChatPromptTemplate.from_template(template)

        def docs2str(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Build RAG chain
        rag_chain = (
            {"context": retriever | docs2str, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Run query
        answer = rag_chain.invoke(question)

        final_message = re.search(r"<\|start\|>assistant<\|channel\|>final<\|message\|>(.*)", answer, re.DOTALL).group(1).strip()
        del rag_chain, retriever, vectorstore, llm, prompt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return jsonify({
            "log_id": log_id,
            "query": question,
            "answer": final_message
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run in debug mode for development
    app.run(host="0.0.0.0", port=8888)
