from config.config_loader import ConfigLoader
from cache.cache_manager import FAISSCacheManager
from models.model_loader import ModelLoader
from processing.document_processor import DocumentProcessor
from processing.rag_configurator import RAGConfigurator
from chat.chat_manager import ChatManager

class NotaFiscalEletronica:
    """Classe principal que orquestra todas as funcionalidades"""
    
    def __init__(self):
        # Configuração
        config_loader = ConfigLoader()
        config_loader.setup_environment()
        
        # Componentes
        self.cache_manager = FAISSCacheManager()
        self.model_loader = ModelLoader()
        self.document_processor = DocumentProcessor()
        
        # Inicializa modelos
        self.model_loader.initialize_models(
            {'modelo': "mistralai/Mistral-7B-Instruct-v0.2", 'temperature': 0.1},
            {'modelo': "phi3", 'temperature': 0.1}
        )
        
        # Configura RAG e Chat
        self.rag_configurator = RAGConfigurator(self.model_loader, self.document_processor)
        self.chat_manager = ChatManager(self.rag_configurator, self.cache_manager)
    
    def processar_documento(self, caminho_arquivo: str):
        """Processa um documento PDF"""
        return self.document_processor.process_document(caminho_arquivo)
    
    def perguntar(self, pergunta: str, modelo_llm: str = 'huggingface'):
        """Faz uma pergunta ao sistema"""
        return self.chat_manager.process_message(pergunta, modelo_llm)
    
    def limpar_cache(self):
        """Limpa o cache de respostas"""
        self.cache_manager.clear_cache()
    
    def estatisticas_cache(self):
        """Retorna estatísticas do cache"""
        return self.cache_manager.get_cache_stats()

def main():
    nfe = NotaFiscalEletronica()

    caminho_pdf = "base/ANEXOI-LeiauteeRegradeValidacao-NF-eeNFC-e.pdf"
    
    try:
        nfe.processar_documento(caminho_pdf)
        print("\nSistema de Perguntas sobre Nota Fiscal Eletrônica")
        print("Digite 'sair' para encerrar.")
        print("Digite 'limpar_cache' para limpar o cache.")
        print("Digite 'estatisticas' para ver estatísticas do cache.\n")

        while True:
            try:
                pergunta = input("Você: ")
                if pergunta.lower() == 'sair':
                    break
                elif pergunta.lower() == 'limpar_cache':
                    nfe.limpar_cache()
                    print("Cache limpo!")
                    continue
                elif pergunta.lower() == 'estatisticas':
                    stats = nfe.estatisticas_cache()
                    print(f"Estatísticas do cache: {stats}")
                    continue
                
                resposta = nfe.perguntar(pergunta, modelo_llm='huggingface')
                print(f"Assistente: {resposta}\n")
            except Exception as e:
                print(f"Ocorreu um erro: {e}")
    
    except FileNotFoundError:
        print("Arquivo PDF não encontrado!")
    except Exception as e:
        print(f"Erro ao processar documento: {e}")

if __name__ == "__main__":
    main()