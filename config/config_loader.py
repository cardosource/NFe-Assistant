import os
import yaml
from typing import Dict, Any

class ConfigLoader:
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
    
    def load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
    
    def setup_environment(self):
        config = self.load_config()
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = config['HUGGINGFACEHUB_API_TOKEN']