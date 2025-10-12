import os
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class DocumentProcessor:

    def __init__(self):
        self.retriever = None
        self.vector_db = None
    
    def process_document(self, caminho_arquivo: str):
        if not os.path.exists(caminho_arquivo):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    
        loader = PyPDFLoader(caminho_arquivo)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        separando = text_splitter.split_documents(docs)

        embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

        self.vector_db = FAISS.from_documents(separando, embedding)
        self.vector_db.save_local("banco_faiss")


        self.retriever = self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 3, 'fetch_k': 4}
        )
        
        return self.retriever
    
    def get_retriever(self):
        if not self.retriever:
            raise ValueError("Retriever não foi configurado. Processe um documento primeiro.")
        return self.retriever