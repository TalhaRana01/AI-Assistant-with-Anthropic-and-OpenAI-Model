from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class LLMProvider(ABC):
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> None:
      
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass
       
    
    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        pass
     
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass
       
    
    @abstractmethod
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        pass
       
    
    @abstractmethod
    async def close(self) -> None:
        pass
        
  
    