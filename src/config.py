from __future__ import annotations
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key"
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key"
    )
    
    # Provider Configuration
    default_provider: str = Field(
        default="openai",
        description="Default LLM provider"
    )
    
    # Model Configuration
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name"
    )
    anthropic_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Anthropic model name"
    )
    
    # Generation Parameters
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=2000,
        gt=0,
        description="Maximum tokens in response"
    )
    
    # API Configuration
    timeout: int = Field(
        default=30,
        gt=0,
        description="API request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts"
    )
    
    # Cost Management
    cost_warning_threshold: float = Field(
        default=1.0,
        ge=0.0,
        description="Cost warning threshold in USD"
    )
    cost_hard_limit: float | None = Field(
        default=None,
        description="Hard cost limit in USD (None for no limit)"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    @field_validator("default_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
       
        v = v.lower()
        if v not in ["openai", "anthropic"]:
            raise ValueError("Provider must be 'openai' or 'anthropic'")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
       
        v = v.upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v
    
    def validate_api_keys(self) -> None:
        
        if self.default_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required when using OpenAI provider"
            )
        if self.default_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required when using Anthropic provider"
            )
    
    def get_api_key(self, provider: str) -> str:
      
        provider = provider.lower()
        if provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            return self.openai_api_key.get_secret_value()
        elif provider == "anthropic":
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")
            return self.anthropic_api_key.get_secret_value()
        else:
            raise ValueError(f"Unknown provider: {provider}")


# Global settings instance
settings = Settings()