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
        # Usando um embedding mais leve e estável
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Força CPU para evitar warnings
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.initialize_cache()
    
    def initialize_cache(self):
        """Inicializa ou carrega o cache FAISS"""
        try:
            self.vectorstore = FAISS.load_local(
                self.index_path, 
                self.embedding, 
                allow_dangerous_deserialization=True
            )
            print("✅ Cache FAISS carregado com sucesso")
        except Exception as e:
            # Cria um novo cache se não existir
            self.vectorstore = FAISS.from_texts(
                ["cache inicial"], 
                self.embedding, 
                metadatas=[{"pergunta": "init", "resposta": "init"}]
            )
            self.vectorstore.save_local(self.index_path)
            print("🆕 Novo cache FAISS criado")
    
    def lookup(self, input_text: str, namespace: str = "") -> Optional[str]:
        """Busca uma resposta no cache usando similaridade semântica"""
        if not input_text.strip():
            return None
            
        try:
            # Busca perguntas similares no cache
            docs = self.vectorstore.similarity_search(
                input_text, 
                k=1,
                filter={"namespace": namespace} if namespace else {}
            )
            
            if docs and self._is_similar_enough(input_text, docs[0].page_content):
                # Retorna a resposta do cache se encontrou pergunta similar
                return docs[0].metadata.get("resposta", None)
            
            return None
            
        except Exception as e:
            print(f"⚠️ Erro ao buscar no cache: {e}")
            return None
    
    def _is_similar_enough(self, input_text: str, cached_text: str, threshold: float = 0.7) -> bool:
        """Verifica se as perguntas são suficientemente similares"""
        # Implementação simples de similaridade
        input_words = set(input_text.lower().split())
        cached_words = set(cached_text.lower().split())
        
        # Remove palavras muito comuns
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
        """Adiciona nova pergunta-resposta ao cache"""
        if not input_text.strip():
            return
            
        try:
            # Converte a resposta para string se não for
            response_str = str(response) if not isinstance(response, str) else response
            
            # Cria documento com metadados
            metadata = {
                "pergunta": input_text,
                "resposta": response_str,
                "namespace": namespace,
                "timestamp": str(np.datetime64('now'))
            }
            
            # Adiciona ao vectorstore
            self.vectorstore.add_texts(
                texts=[input_text],
                metadatas=[metadata]
            )
            
            # Salva o cache periodicamente (não a cada chamada para performance)
            if np.random.random() < 0.1:  # Salva aleatoriamente ~10% das vezes
                self.vectorstore.save_local(self.index_path)
                
        except Exception as e:
            print(f"⚠️ Erro ao atualizar cache: {e}")
    
    def clear_cache(self):
        """Limpa todo o cache"""
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
        """Retorna estatísticas do cache"""
        try:
            # Método aproximado para contar documentos
            docs = self.vectorstore.similarity_search("", k=1000)
            return {
                "total_documentos": len(docs),
                "tamanho_cache": f"{len(docs)} entradas"
            }
        except:
            return {"total_documentos": 0, "tamanho_cache": "0 entradas"}
    
    def save_cache(self):
        """Salva o cache explicitamente"""
        try:
            self.vectorstore.save_local(self.index_path)
            print("✅ Cache salvo com sucesso")
        except Exception as e:
            print(f"⚠️ Erro ao salvar cache: {e}")