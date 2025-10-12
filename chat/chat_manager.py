from typing import List
from langchain_core.messages import AIMessage, HumanMessage

class ChatManager:
    """Responsável por gerenciar a conversação"""
    
    def __init__(self, rag_configurator, cache_manager):
        self.rag_configurator = rag_configurator
        self.cache_manager = cache_manager
        self.chat_history: List = []
    
    def process_message(self, input_text: str, modelo_llm: str = 'huggingface') -> str:
        """Processa uma mensagem e retorna a resposta"""
        # Verifica cache primeiro - agora usando FAISS
        cached_response = self.cache_manager.lookup(input_text)
        if cached_response:
            print("Usando resposta do cache FAISS:")
            return cached_response
        
        # Obtém a cadeia RAG
        rag_chain = self.rag_configurator.configure_rag_chain(modelo_llm)
        
        # Executa a consulta
        result = rag_chain.invoke({
            "input": input_text,
            "chat_history": self.chat_history
        })
        
        # Armazena no cache FAISS
        self.cache_manager.update(input_text, result['answer'])
        
        # Atualiza histórico
        self.update_chat_history(input_text, result['answer'])
        
        return result['answer']
    
    def update_chat_history(self, pergunta: str, resposta: str):
        """Atualiza o histórico da conversa"""
        self.chat_history.extend([
            HumanMessage(content=pergunta),
            AIMessage(content=resposta)
        ])
    
    def clear_chat_history(self):
        """Limpa o histórico da conversa"""
        self.chat_history = []