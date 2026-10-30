from abc import ABC, abstractmethod
from typing import Any, Dict
from PIL import Image

class BasePipeline(ABC):
    def __init__(self, model_info: Dict[str, Any]):
        self.model_info = model_info
        self.is_loaded = False

    @abstractmethod
    def load(self) -> bool:
        pass

    @abstractmethod
    def unload(self) -> bool:
        pass
