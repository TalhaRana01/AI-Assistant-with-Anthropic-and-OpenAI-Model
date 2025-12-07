"""Tests for LLM provider implementations.

This module tests the OpenAI and Anthropic providers using mocks
to avoid actual API calls during testing.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.anthropic_provider import AnthropicProvider
from src.llm.openai_provider import OpenAIProvider


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Test response from OpenAI"
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    return response


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response."""
    response = MagicMock()
    response.content = [MagicMock()]
    response.content[0].text = "Test response from Anthropic"
    response.usage = MagicMock()
    response.usage.input_tokens = 10
    response.usage.output_tokens = 5
    return response


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=1000
        )
        
        assert provider.provider_name == "openai"
        assert provider.model == "gpt-4o-mini"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 1000
    
    @pytest.mark.asyncio
    async def test_generate(self, mock_openai_response):
        """Test message generation."""
        provider = OpenAIProvider(api_key="test-key")
        
        with patch.object(
            provider.client.chat.completions,
            'create',
            new_callable=AsyncMock,
            return_value=mock_openai_response
        ):
            messages = [{"role": "user", "content": "Hello"}]
            response = await provider.generate(messages)
            
            assert response == "Test response from OpenAI"
    
    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Test retry logic on rate limit errors."""
        provider = OpenAIProvider(api_key="test-key", max_retries=2)
        
        import openai
        
        # First call fails, second succeeds
        mock_create = AsyncMock()
        mock_create.side_effect = [
            openai.RateLimitError("Rate limit", response=MagicMock(), body={}),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="Success"))],
                usage=MagicMock(prompt_tokens=10, completion_tokens=5)
            )
        ]
        
        with patch.object(
            provider.client.chat.completions,
            'create',
            mock_create
        ):
            messages = [{"role": "user", "content": "Test"}]
            response = await provider.generate(messages)
            
            assert response == "Success"
            assert mock_create.call_count == 2
    
    def test_count_tokens(self):
        """Test token counting."""
        provider = OpenAIProvider(api_key="test-key")
        
        text = "Hello, world!"
        token_count = provider.count_tokens(text)
        
        # Should return a positive integer
        assert isinstance(token_count, int)
        assert token_count > 0
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")
        
        # Calculate cost for 1M input and 1M output tokens
        cost = provider.estimate_cost(1_000_000, 1_000_000)
        
        # Should match pricing: $0.150 + $0.600 = $0.750
        assert cost == pytest.approx(0.750)
    
    def test_estimate_cost_small_amounts(self):
        """Test cost estimation for typical usage."""
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")
        
        # 100 input tokens, 50 output tokens
        cost = provider.estimate_cost(100, 50)
        
        # (100/1M * 0.150) + (50/1M * 0.600) = 0.000045
        assert cost == pytest.approx(0.000045)


class TestAnthropicProvider:
    """Test Anthropic provider implementation."""
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = AnthropicProvider(
            api_key="test-key",
            model="claude-3-5-haiku-20241022",
            temperature=0.5,
            max_tokens=1000
        )
        
        assert provider.provider_name == "anthropic"
        assert provider.model == "claude-3-5-haiku-20241022"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 1000
    
    @pytest.mark.asyncio
    async def test_generate(self, mock_anthropic_response):
        """Test message generation."""
        provider = AnthropicProvider(api_key="test-key")
        
        with patch.object(
            provider.client.messages,
            'create',
            new_callable=AsyncMock,
            return_value=mock_anthropic_response
        ):
            messages = [{"role": "user", "content": "Hello"}]
            response = await provider.generate(messages)
            
            assert response == "Test response from Anthropic"
    
    def test_convert_messages(self):
        """Test message format conversion."""
        provider = AnthropicProvider(api_key="test-key")
        
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        
        system, converted = provider._convert_messages(messages)
        
        assert system == "You are helpful"
        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"
    
    def test_count_tokens(self):
        """Test token estimation."""
        provider = AnthropicProvider(api_key="test-key")
        
        # Anthropic uses ~4 chars per token approximation
        text = "Hello, world!"  # 13 characters
        token_count = provider.count_tokens(text)
        
        # Should be approximately 3-4 tokens
        assert token_count == 3  # 13 // 4 = 3
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        provider = AnthropicProvider(
            api_key="test-key",
            model="claude-3-5-haiku-20241022"
        )
        
        # Calculate cost for 1M input and 1M output tokens
        cost = provider.estimate_cost(1_000_000, 1_000_000)
        
        # Should match pricing: $0.80 + $4.00 = $4.80
        assert cost == pytest.approx(4.80)
    
    def test_estimate_cost_small_amounts(self):
        """Test cost estimation for typical usage."""
        provider = AnthropicProvider(
            api_key="test-key",
            model="claude-3-5-haiku-20241022"
        )
        
        # 100 input tokens, 50 output tokens
        cost = provider.estimate_cost(100, 50)
        
        # (100/1M * 0.80) + (50/1M * 4.00) = 0.00028
        assert cost == pytest.approx(0.00028)


class TestProviderInterface:
    """Test that both providers implement the same interface."""
    
    def test_openai_has_required_methods(self):
        """Test OpenAI provider has all required methods."""
        provider = OpenAIProvider(api_key="test-key")
        
        assert hasattr(provider, 'generate')
        assert hasattr(provider, 'count_tokens')
        assert hasattr(provider, 'estimate_cost')
        assert hasattr(provider, 'close')
        assert hasattr(provider, 'provider_name')
    
    def test_anthropic_has_required_methods(self):
        """Test Anthropic provider has all required methods."""
        provider = AnthropicProvider(api_key="test-key")
        
        assert hasattr(provider, 'generate')
        assert hasattr(provider, 'count_tokens')
        assert hasattr(provider, 'estimate_cost')
        assert hasattr(provider, 'close')
        assert hasattr(provider, 'provider_name')
    
    def test_provider_name_property(self):
        """Test provider_name is correctly implemented."""
        openai_provider = OpenAIProvider(api_key="test")
        anthropic_provider = AnthropicProvider(api_key="test")
        
        assert openai_provider.provider_name == "openai"
        assert anthropic_provider.provider_name == "anthropic"