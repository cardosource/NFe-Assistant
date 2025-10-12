from typing import Dict, Any
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama

class ModelLoader:
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
    
    def load_huggingface_model(self, modelo: str, temperature: float) -> ChatHuggingFace:
        print(f"Carregando modelo HuggingFace: {modelo}")
        llm = HuggingFaceEndpoint(
            repo_id=modelo,
            task="text-generation",
            temperature=temperature,
            max_new_tokens=300
        )
        return ChatHuggingFace(llm=llm)
    
    def load_ollama_model(self, modelo: str, temperature: float) -> ChatOllama:
        print(f"Carregando modelo Ollama: {modelo}")
        return ChatOllama(model=modelo, temperature=temperature)
    
    def get_model(self, model_type: str) -> Any:
        return self.models.get(model_type)
    
    def initialize_models(self, huggingface_config: Dict[str, Any], ollama_config: Dict[str, Any]):
        self.models['huggingface'] = self.load_huggingface_model(
            huggingface_config['modelo'], 
            huggingface_config['temperature']
        )
        self.models['ollama'] = self.load_ollama_model(
            ollama_config['modelo'], 
            ollama_config['temperature']
        )