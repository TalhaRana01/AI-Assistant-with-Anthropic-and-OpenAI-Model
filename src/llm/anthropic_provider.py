from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

import anthropic
import httpx
from anthropic import AsyncAnthropic

from .base import LLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider implementation.
    
    Implements async communication with Anthropic API including retry logic
    and cost estimation.
    
    Attributes:
        client: Async Anthropic client
        
    Example:
        >>> provider = AnthropicProvider(
        ...     api_key="sk-ant-...",
        ...     model="claude-3-5-haiku-20241022"
        ... )
        >>> response = await provider.generate([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
    """
    
    # Pricing per 1M tokens (as of December 2024)
    PRICING = {
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    }
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 30,
        max_retries: int = 3
    ) -> None:
        """Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        super().__init__(model, temperature, max_tokens)
        self.max_retries = max_retries
        
        # Initialize async client with timeout
        timeout_config = httpx.Timeout(timeout, connect=10.0)
        self.client = AsyncAnthropic(
            api_key=api_key,
            timeout=timeout_config,
            max_retries=0  # We handle retries manually
        )
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "anthropic"
    
    def _convert_messages(
        self,
        messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Convert messages to Anthropic format.
        
        Anthropic requires system messages to be separate from the
        messages list.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Tuple of (system_message, converted_messages)
        """
        system_message = None
        converted_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                converted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return system_message, converted_messages
    
    async def generate(
        self,
        messages: list[dict[str, str]],
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        """Generate response from Anthropic.
        
        Implements exponential backoff retry logic for transient failures.
        
        Args:
            messages: List of message dictionaries
            stream: Whether to stream the response
            
        Returns:
            Complete response string or async generator for streaming
            
        Raises:
            anthropic.APIConnectionError: Connection failed after retries
            anthropic.RateLimitError: Rate limit exceeded
            anthropic.APIStatusError: API returned error status
        """
        system_message, converted_messages = self._convert_messages(messages)
        
        if stream:
            return self._generate_stream(system_message, converted_messages)
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Anthropic API call attempt {attempt + 1}/{self.max_retries}"
                )
                
                kwargs = {
                    "model": self.model,
                    "messages": converted_messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
                
                if system_message:
                    kwargs["system"] = system_message
                
                response = await self.client.messages.create(**kwargs)
                
                content = response.content[0].text if response.content else ""
                
                # Log token usage
                logger.info(
                    f"Anthropic tokens - Input: {response.usage.input_tokens}, "
                    f"Output: {response.usage.output_tokens}"
                )
                
                return content
                
            except anthropic.RateLimitError as e:
                logger.warning(f"Rate limit exceeded: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
                    
            except anthropic.APIConnectionError as e:
                logger.warning(f"Connection error: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
                    
            except anthropic.APIStatusError as e:
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
        system_message: str | None,
        messages: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from Anthropic.
        
        Args:
            system_message: Optional system message
            messages: List of message dictionaries
            
        Yields:
            Text chunks as they arrive
        """
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": True
                }
                
                if system_message:
                    kwargs["system"] = system_message
                
                async with self.client.messages.stream(**kwargs) as stream:
                    async for text in stream.text_stream:
                        yield text
                return
                
            except (anthropic.APIConnectionError, anthropic.RateLimitError) as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Stream error: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count for Anthropic.
        
        Anthropic doesn't provide a public tokenizer, so we use
        a rough approximation of ~4 characters per token.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated number of tokens
        """
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Estimate cost based on token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        pricing = self.PRICING.get(
            self.model,
            {"input": 0.80, "output": 4.00}  # Default to Haiku
        )
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    async def close(self) -> None:
        """Close the async client."""
        await self.client.close()