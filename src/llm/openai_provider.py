"""OpenAI LLM provider implementation.

This module implements the OpenAI provider with retry logic,
error handling, and cost tracking.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

import httpx
import openai
import tiktoken
from openai import AsyncOpenAI

from .base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
   
    PRICING = {
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    }
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 30,
        max_retries: int = 3
    ) -> None:
        
        super().__init__(model, temperature, max_tokens)
        self.max_retries = max_retries
        
        # Initialize async client with timeout
        timeout_config = httpx.Timeout(timeout, connect=10.0)
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout_config,
            max_retries=0  # We handle retries manually
        )
        
        # Initialize tiktoken encoding
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Model {model} not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "openai"
    
    async def generate(
        self,
        messages: list[dict[str, str]],
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:

        if stream:
            return self._generate_stream(messages)
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"OpenAI API call attempt {attempt + 1}/{self.max_retries}"
                )
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                content = response.choices[0].message.content or ""
                
                # Log token usage
                if response.usage:
                    logger.info(
                        f"OpenAI tokens - Input: {response.usage.prompt_tokens}, "
                        f"Output: {response.usage.completion_tokens}"
                    )
                
                return content
                
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit exceeded: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
                    
            except openai.APIConnectionError as e:
                logger.warning(f"Connection error: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
                    
            except openai.APIStatusError as e:
                logger.error(f"API status error: {e.status_code} - {e.message}")
                # Don't retry on client errors (4xx)
                if 400 <= e.status_code < 500:
                    raise
                # Retry on server errors (5xx)
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    async def _generate_stream(
        self,
        messages: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
      
        for attempt in range(self.max_retries):
            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
                
            except (openai.APIConnectionError, openai.RateLimitError) as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Stream error: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    def count_tokens(self, text: str) -> int:
        
        return len(self.encoding.encode(text))
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
       
        pricing = self.PRICING.get(
            self.model,
            {"input": 0.150, "output": 0.600}  # Default to gpt-4o-mini
        )
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    async def close(self) -> None:
       
        await self.client.close()