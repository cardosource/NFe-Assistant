import numpy as np
from typing import Any, Dict, List, Optional
import json
import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class FAISSCacheManager:
    """Gerenciador de cache usando FAISS para armazenar e recuperar respostas"""
    
    def __init__(self, index_path: str = "cache_faiss"):
        self.index_path = index_path
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Força CPU para evitar warnings
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.initialize_cache()
    
    def initialize_cache(self):
        try:
            self.vectorstore = FAISS.load_local(
                self.index_path, 
                self.embedding, 
                allow_dangerous_deserialization=True
            )
            print("✅ Cache FAISS carregado com sucesso")
        except Exception as e:
            self.vectorstore = FAISS.from_texts(
                ["cache inicial"], 
                self.embedding, 
                metadatas=[{"pergunta": "init", "resposta": "init"}]
            )
            self.vectorstore.save_local(self.index_path)
            print("🆕 Novo cache FAISS criado")
    
    def lookup(self, input_text: str, namespace: str = "") -> Optional[str]:
        if not input_text.strip():
            return None
            
        try:
            docs = self.vectorstore.similarity_search(
                input_text, 
                k=1,
                filter={"namespace": namespace} if namespace else {}
            )
            
            if docs and self._is_similar_enough(input_text, docs[0].page_content):
                return docs[0].metadata.get("resposta", None)
            
            return None
            
        except Exception as e:
            print(f"⚠️ Erro ao buscar no cache: {e}")
            return None
    
    def _is_similar_enough(self, input_text: str, cached_text: str, threshold: float = 0.7) -> bool:
        input_words = set(input_text.lower().split())
        cached_words = set(cached_text.lower().split())
        
        stop_words = {
            "o", "a", "os", "as", "de", "da", "do", "em", "para", "com", 
            "é", "são", "um", "uma", "uns", "umas", "que", "como", "qual"
        }
        input_words = input_words - stop_words
        cached_words = cached_words - stop_words
        
        if not input_words or not cached_words:
            return False
            
        intersection = input_words.intersection(cached_words)
        similarity = len(intersection) / max(len(input_words), len(cached_words))
        
        return similarity >= threshold
    
    def update(self, input_text: str, response: Any, namespace: str = ""):
        if not input_text.strip():
            return
            
        try:
            response_str = str(response) if not isinstance(response, str) else response
            
            metadata = {
                "pergunta": input_text,
                "resposta": response_str,
                "namespace": namespace,
                "timestamp": str(np.datetime64('now'))
            }
            
            self.vectorstore.add_texts(
                texts=[input_text],
                metadatas=[metadata]
            )
            
            if np.random.random() < 0.1:  
                self.vectorstore.save_local(self.index_path)
                
        except Exception as e:
            print(f"⚠️ Erro ao atualizar cache: {e}")
    
    def clear_cache(self):
        try:
            self.vectorstore = FAISS.from_texts(
                ["cache inicial"], 
                self.embedding, 
                metadatas=[{"pergunta": "init", "resposta": "init"}]
            )
            self.vectorstore.save_local(self.index_path)
            print("✅ Cache limpo com sucesso")
        except Exception as e:
            print(f"⚠️ Erro ao limpar cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        try:
            docs = self.vectorstore.similarity_search("", k=1000)
            return {
                "total_documentos": len(docs),
                "tamanho_cache": f"{len(docs)} entradas"
            }
        except:
            return {"total_documentos": 0, "tamanho_cache": "0 entradas"}
    
    def save_cache(self):
        try:
            self.vectorstore.save_local(self.index_path)
            print("✅ Cache salvo com sucesso")
        except Exception as e:
            print(f"⚠️ Erro ao salvar cache: {e}")
