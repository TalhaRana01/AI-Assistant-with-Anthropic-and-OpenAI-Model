"""Tests for configuration management.

This module tests the Pydantic settings configuration including
validation, API key management, and environment variable loading.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from src.config import Settings


def test_settings_defaults():
    """Test that settings load with default values."""
    settings = Settings(
        openai_api_key=SecretStr("test-key"),
        anthropic_api_key=SecretStr("test-key")
    )
    
    assert settings.default_provider == "openai"
    assert settings.temperature == 0.7
    assert settings.max_tokens == 2000
    assert settings.timeout == 30
    assert settings.max_retries == 3
    assert settings.log_level == "INFO"


def test_temperature_validation():
    """Test temperature validation range."""
    # Valid temperatures
    Settings(
        openai_api_key=SecretStr("test"),
        temperature=0.0
    )
    Settings(
        openai_api_key=SecretStr("test"),
        temperature=2.0
    )
    
    # Invalid temperatures
    with pytest.raises(ValidationError):
        Settings(
            openai_api_key=SecretStr("test"),
            temperature=-0.1
        )
    
    with pytest.raises(ValidationError):
        Settings(
            openai_api_key=SecretStr("test"),
            temperature=2.1
        )


def test_provider_validation():
    """Test provider name validation."""
    # Valid providers
    settings = Settings(
        openai_api_key=SecretStr("test"),
        default_provider="openai"
    )
    assert settings.default_provider == "openai"
    
    settings = Settings(
        openai_api_key=SecretStr("test"),
        default_provider="ANTHROPIC"
    )
    assert settings.default_provider == "anthropic"
    
    # Invalid provider
    with pytest.raises(ValidationError):
        Settings(
            openai_api_key=SecretStr("test"),
            default_provider="invalid"
        )


def test_log_level_validation():
    """Test log level validation."""
    # Valid log levels
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        settings = Settings(
            openai_api_key=SecretStr("test"),
            log_level=level.lower()
        )
        assert settings.log_level == level
    
    # Invalid log level
    with pytest.raises(ValidationError):
        Settings(
            openai_api_key=SecretStr("test"),
            log_level="INVALID"
        )


def test_api_key_validation():
    """Test API key validation."""
    # OpenAI provider requires OpenAI key
    settings = Settings(
        openai_api_key=SecretStr("sk-test"),
        default_provider="openai"
    )
    settings.validate_api_keys()  # Should not raise
    
    # Anthropic provider requires Anthropic key
    settings = Settings(
        anthropic_api_key=SecretStr("sk-ant-test"),
        default_provider="anthropic"
    )
    settings.validate_api_keys()  # Should not raise
    
    # Missing required key
    settings = Settings(default_provider="openai")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.validate_api_keys()


def test_get_api_key():
    """Test API key retrieval."""
    settings = Settings(
        openai_api_key=SecretStr("sk-openai-test"),
        anthropic_api_key=SecretStr("sk-ant-test")
    )
    
    assert settings.get_api_key("openai") == "sk-openai-test"
    assert settings.get_api_key("anthropic") == "sk-ant-test"
    
    # Unknown provider
    with pytest.raises(ValueError, match="Unknown provider"):
        settings.get_api_key("unknown")
    
    # Missing key
    settings = Settings()
    with pytest.raises(ValueError, match="not configured"):
        settings.get_api_key("openai")


def test_secret_str_not_exposed():
    """Test that SecretStr prevents accidental key exposure."""
    settings = Settings(openai_api_key=SecretStr("sk-secret-key"))
    
    # String representation should not show the key
    settings_str = str(settings)
    assert "sk-secret-key" not in settings_str
    assert "**********" in settings_str or "SecretStr" in settings_str


def test_max_tokens_positive():
    """Test that max_tokens must be positive."""
    # Valid
    Settings(
        openai_api_key=SecretStr("test"),
        max_tokens=1
    )
    
    # Invalid
    with pytest.raises(ValidationError):
        Settings(
            openai_api_key=SecretStr("test"),
            max_tokens=0
        )
    
    with pytest.raises(ValidationError):
        Settings(
            openai_api_key=SecretStr("test"),
            max_tokens=-1
        )


def test_timeout_positive():
    """Test that timeout must be positive."""
    # Valid
    Settings(
        openai_api_key=SecretStr("test"),
        timeout=1
    )
    
    # Invalid
    with pytest.raises(ValidationError):
        Settings(
            openai_api_key=SecretStr("test"),
            timeout=0
        )


def test_cost_thresholds():
    """Test cost threshold configuration."""
    settings = Settings(
        openai_api_key=SecretStr("test"),
        cost_warning_threshold=5.0,
        cost_hard_limit=10.0
    )
    
    assert settings.cost_warning_threshold == 5.0
    assert settings.cost_hard_limit == 10.0
    
    # No hard limit
    settings = Settings(
        openai_api_key=SecretStr("test"),
        cost_hard_limit=None
    )
    assert settings.cost_hard_limit is None