


# AI Assistant - Production-Ready Multi-Provider CLI

A command-line AI Assistant that demonstrates production-grade patterns for integrating multiple LLM providers (OpenAI and Anthropic) with secure API key management, comprehensive error handling, cost tracking, and conversation management.

## 🚀 Features

- **Multi-Provider Support**: Seamlessly switch between OpenAI (GPT-4o-mini) and Anthropic (Claude 3.5 Haiku)
- **Async/Await**: Non-blocking I/O operations for efficient API calls
- **Retry Logic**: Automatic exponential backoff for transient failures
- **Cost Tracking**: Real-time cost estimation with configurable thresholds
- **Conversation Management**: Maintains context across multiple turns
- **Secure Configuration**: Pydantic-based settings with environment variable validation
- **Comprehensive Logging**: Structured logging with configurable levels
- **Export Capabilities**: Export conversations to JSON or Markdown
- **Type Safety**: Full type hints for better IDE support and error detection

## 📋 Prerequisites

- Python 3.12 or higher
- `uv` package manager (recommended) or `pip`
- OpenAI API key and/or Anthropic API key

## 🛠️ Installation

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone or extract the project
cd ai-assistant

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync

# Install optional enhanced features (Rich CLI)
uv pip install -e ".[enhanced]"
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## ⚙️ Configuration

### 1. Create Environment File

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

### 2. Configure API Keys

Edit `.env` and add your API keys:

```env
# Required: At least one provider key
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Provider Configuration
DEFAULT_PROVIDER=openai  # or 'anthropic'

# Model Selection
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_MODEL=claude-3-5-haiku-20241022

# Generation Parameters
TEMPERATURE=0.7
MAX_TOKENS=2000

# Cost Management
COST_WARNING_THRESHOLD=1.0  # USD
COST_HARD_LIMIT=  # Optional: Set a hard spending limit

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### 3. Verify Configuration

The application will validate your configuration on startup and provide clear error messages if anything is missing.

## 🎯 Usage

### Basic Usage

```bash
# Activate virtual environment if not already active
source .venv/bin/activate

# Run the assistant
python src/main.py
```

### Available Commands

Once running, you can use these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear conversation history |
| `/model` | Switch between OpenAI and Anthropic |
| `/cost` | Display current session costs |
| `/export` | Export conversation to JSON or Markdown |
| `/quit` or `/exit` | Exit the application |

### Example Session

```
🤖 AI Assistant - Multi-Provider Command-Line Interface
======================================================================
Current Provider: openai
Model: gpt-4o-mini
Available Providers: openai, anthropic
======================================================================

👤 You: What is the capital of France?

🤖 Assistant: The capital of France is Paris. It's not only the largest 
city in France but also serves as the country's political, economic, 
and cultural center.

👤 You: /cost

==================================================
Cost Summary
==================================================
Total Cost: $0.000024
Total Input Tokens: 15
Total Output Tokens: 28
Number of API Calls: 1

Cost by Provider:
  openai: $0.000024

Cost by Model:
  gpt-4o-mini: $0.000024

Hard Limit: Not set
==================================================

👤 You: /model
Available providers:
  1. [✓] openai
  2. [ ] anthropic

Select provider (number): 2
✓ Switched to anthropic

👤 You: /quit
👋 Goodbye!
```

## 🏗️ Architecture

### Project Structure

```
ai-assistant/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Application entry point
│   ├── config.py                  # Pydantic settings
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract base class
│   │   ├── openai_provider.py   # OpenAI implementation
│   │   └── anthropic_provider.py # Anthropic implementation
│   └── utils/
│       ├── __init__.py
│       ├── cost_tracker.py       # Cost tracking
│       ├── conversation.py       # Conversation management
│       └── logger.py             # Logging configuration
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_llm_providers.py
│   └── test_cost_tracker.py
├── pyproject.toml                # Project configuration
├── .env.example                  # Environment template
├── .gitignore
└── README.md
```

### Key Design Patterns

**Abstract Factory Pattern**: The `LLMProvider` base class defines a common interface that both OpenAI and Anthropic providers implement, making it easy to add new providers.

**Settings Management**: Pydantic Settings automatically validates environment variables and provides type-safe access to configuration.

**Retry Logic**: Exponential backoff with configurable retries handles transient API failures gracefully.

**Resource Management**: Async context managers ensure proper cleanup of HTTP connections.

## 💰 Cost Estimates

### Pricing (December 2024)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-4o-mini | $0.150 | $0.600 |
| Claude 3.5 Haiku | $0.80 | $4.00 |

### Typical Usage Costs

| Usage Pattern | Estimated Cost |
|---------------|----------------|
| 10 short exchanges (200 tokens each) | $0.01 - $0.05 |
| 1 hour conversation (~50 exchanges) | $0.05 - $0.25 |
| Full day of development testing | $1.00 - $5.00 |

**Note**: Actual costs depend on conversation length and complexity. Use the `/cost` command to track spending in real-time.

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_config.py

# Run in verbose mode
pytest -v
```

### View Coverage Report

```bash
# Open HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## 🔍 Troubleshooting

### Common Issues

**Issue**: `Module not found` errors
```bash
# Solution: Install dependencies
uv sync
# or
pip install -e .
```

**Issue**: `Invalid API key` errors
```bash
# Solution: Verify .env file format
cat .env  # Check for extra spaces or quotes
# Keys should be: OPENAI_API_KEY=sk-...
```

**Issue**: `Rate limit exceeded`
```bash
# Solution: Wait briefly; retry logic should handle this automatically
# Or check your API usage at:
# - OpenAI: https://platform.openai.com/usage
# - Anthropic: https://console.anthropic.com/settings/usage
```

**Issue**: Import errors
```bash
# Solution: Ensure running from project root
cd ai-assistant
python src/main.py
```

### Debug Mode

Enable debug logging for detailed information:

```bash
# In .env file
LOG_LEVEL=DEBUG

# Or set temporarily
export LOG_LEVEL=DEBUG
python src/main.py
```

## 🔐 Security Best Practices

1. **Never commit `.env` files** - They contain sensitive API keys
2. **Use `.env.example`** - Commit this as a template without real keys
3. **Rotate API keys regularly** - Especially if accidentally exposed
4. **Set cost limits** - Use `COST_HARD_LIMIT` to prevent unexpected charges
5. **Monitor usage** - Check provider dashboards regularly

## 🚀 Future Enhancements

Potential improvements for future versions:

- [ ] **Streaming responses** - Display text as it's generated
- [ ] **Model comparison mode** - Compare responses side-by-side
- [ ] **Context window management** - Auto-trim old messages
- [ ] **Conversation summarization** - Reduce token usage
- [ ] **Rich terminal UI** - Enhanced formatting with colors
- [ ] **Plugin system** - Extensible architecture for custom providers
- [ ] **Conversation search** - Find past conversations
- [ ] **Multi-turn planning** - Agent-like capabilities

## 📚 Additional Resources

- [OpenAI Python Library](https://github.com/openai/openai-python)
- [Anthropic SDK Documentation](https://docs.anthropic.com/en/api/client-sdks)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
- [Python Asyncio Guide](https://docs.python.org/3/library/asyncio.html)

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is an educational project for the NexusBerry Full-Stack AI Engineer Course. 

## 📧 Support

For questions or issues:
- Review the troubleshooting section above
- Check the course discussion forum
- Attend virtual office hours

---

**Built with ❤️ for the NexusBerry AI Engineer Course**
