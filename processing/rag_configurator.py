from typing import Any
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class RAGConfigurator:
    """Cadeia RAG"""
    
    def __init__(self, model_loader: Any, document_processor: Any):
        self.model_loader = model_loader
        self.document_processor = document_processor
    
    def configure_rag_chain(self, modelo_llm: str) -> Any:
        llm = self.model_loader.get_model(modelo_llm)
        if not llm:
            raise ValueError(f"Modelo {modelo_llm} não encontrado")
        
        retriever = self.document_processor.get_retriever()
        
        if modelo_llm == 'ollama':
            token_s, token_e = "<|system|>", "<|end|><|assistant|>"
        else:
            token_s = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
            token_e = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        contexto_prompt = ChatPromptTemplate.from_messages([
            ("system", token_s + "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is." + token_e),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        """
        Tradução:
        Com base no histórico da conversa e na próxima pergunta que pode se referir a algo dito antes, crie uma pergunta completa que faça sentido sozinha, sem precisar do histórico. NÃO responda a pergunta, só reformule se for necessário. Se não precisar reformular, deixe como está.
        """
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=contexto_prompt
        )
        
        pr_prompt_template = token_s + """
        Você é um assistente virtual prestativo e está respondendo perguntas sobre notas fiscais eletrônicas.
        Use os seguintes pedaços de contexto recuperado para responder à pergunta.
        Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa.
        Responda em português.

        Pergunta: {input}
        Contexto: {context}
        """ + token_e
        
        pr_prompt = PromptTemplate.from_template(pr_prompt_template)
        pr_chain = create_stuff_documents_chain(llm=llm, prompt=pr_prompt)
        
        return create_retrieval_chain(history_aware_retriever, pr_chain)