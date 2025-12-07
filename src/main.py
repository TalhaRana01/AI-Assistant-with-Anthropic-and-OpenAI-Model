

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from config import settings
from llm.anthropic_provider import AnthropicProvider
from llm.base import LLMProvider
from llm.openai_provider import OpenAIProvider
from utils.conversation import ConversationManager
from utils.cost_tracker import CostTracker
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


class AIAssistant:
    def __init__(self) -> None:
        """Initialize AI Assistant."""
        self.conversation = ConversationManager()
        self.cost_tracker = CostTracker(
            warning_threshold=settings.cost_warning_threshold,
            hard_limit=settings.cost_hard_limit
        )
        self.providers: dict[str, LLMProvider] = {}
        self.current_provider: LLMProvider | None = None
    
    async def initialize(self, provider_name: str | None = None) -> None:
        provider_name = provider_name or settings.default_provider
    
        # Initialize OpenAI if key available
        if settings.openai_api_key:
            try:
                self.providers["openai"] = OpenAIProvider(
                    api_key=settings.get_api_key("openai"),
                    model=settings.openai_model,
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                    timeout=settings.timeout,
                    max_retries=settings.max_retries
                )
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
        
        # Initialize Anthropic if key available
        if settings.anthropic_api_key:
            try:
                self.providers["anthropic"] = AnthropicProvider(
                    api_key=settings.get_api_key("anthropic"),
                    model=settings.anthropic_model,
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                    timeout=settings.timeout,
                    max_retries=settings.max_retries
                )
                logger.info("Anthropic provider initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic: {e}")
        
        if not self.providers:
            raise ValueError(
                "No providers available. Please configure at least one API key."
            )
        
        # Set current provider
        if provider_name in self.providers:
            self.current_provider = self.providers[provider_name]
            logger.info(f"Using {provider_name} provider")
        else:
            self.current_provider = list(self.providers.values())[0]
            logger.warning(
                f"Provider {provider_name} not available, "
                f"using {self.current_provider.provider_name}"
            )
    
    def print_welcome(self) -> None:
        """Print welcome message."""
        print("\n" + "=" * 70)
        print("🤖 AI Assistant - Multi-Provider Command-Line Interface")
        print("=" * 70)
        print(f"Current Provider: {self.current_provider.provider_name}")
        print(f"Model: {self.current_provider.model}")
        print(f"Available Providers: {', '.join(self.providers.keys())}")
        print("\nCommands:")
        print("  /help    - Show this help message")
        print("  /clear   - Clear conversation history")
        print("  /model   - Switch provider")
        print("  /cost    - Show cost summary")
        print("  /export  - Export conversation")
        print("  /quit    - Exit the application")
        print("=" * 70 + "\n")
        
        
    def print_help(self) -> None:
        """Print help message."""
        print("\n📚 Available Commands:")
        print("  /help    - Show this help message")
        print("  /clear   - Clear conversation history")
        print("  /model   - Switch between providers (OpenAI/Anthropic)")
        print("  /cost    - Display current session costs")
        print("  /export  - Export conversation to file")
        print("  /quit    - Exit the application (also /exit)")
        print()
    
    
    
    async def switch_provider(self) -> None:
        """Switch between available providers."""
        if len(self.providers) < 2:
            print("⚠️  Only one provider is available.\n")
            return
        
        print("\nAvailable providers:")
        for i, name in enumerate(self.providers.keys(), 1):
            current = "✓" if name == self.current_provider.provider_name else " "
            print(f"  {i}. [{current}] {name}")
        
        try:
            choice = input("\nSelect provider (number): ").strip()
            provider_names = list(self.providers.keys())
            index = int(choice) - 1
            
            if 0 <= index < len(provider_names):
                provider_name = provider_names[index]
                self.current_provider = self.providers[provider_name]
                print(f"✓ Switched to {provider_name}\n")
                logger.info(f"Switched to {provider_name}")
            else:
                print("❌ Invalid choice\n")
        except (ValueError, IndexError):
            print("❌ Invalid input\n")
    
    def show_cost(self) -> None:
        """Display cost summary."""
        print("\n" + self.cost_tracker.format_summary() + "\n")
    
    def export_conversation(self) -> None:
        """Export conversation to file."""
        if self.conversation.count_messages() == 0:
            print("⚠️  No messages to export.\n")
            return
        
        print("\nExport format:")
        print("  1. JSON")
        print("  2. Markdown")
        
        try:
            choice = input("Select format (1 or 2): ").strip()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if choice == "1":
                filepath = f"conversation_{timestamp}.json"
                self.conversation.export_to_json(filepath)
                print(f"✓ Exported to {filepath}\n")
            elif choice == "2":
                filepath = f"conversation_{timestamp}.md"
                self.conversation.export_to_markdown(filepath)
                print(f"✓ Exported to {filepath}\n")
            else:
                print("❌ Invalid choice\n")
        except Exception as e:
            print(f"❌ Export failed: {e}\n")
            logger.error(f"Export failed: {e}")
    
    async def process_message(self, user_input: str) -> None:
       
        # Check hard limit
        if not self.cost_tracker.check_hard_limit():
            print(
                f"❌ Hard cost limit of ${self.cost_tracker.hard_limit:.2f} "
                "exceeded. No more API calls allowed.\n"
            )
            return
        
        # Add user message
        self.conversation.add_user_message(user_input)
        
        # Count input tokens
        messages = self.conversation.get_messages()
        input_tokens = sum(
            self.current_provider.count_tokens(msg["content"])
            for msg in messages
        )
        
        try:
            # Generate response
            print("\n🤖 Assistant: ", end="", flush=True)
            
            response = await self.current_provider.generate(messages)
            print(response + "\n")
            
            # Add assistant message
            self.conversation.add_assistant_message(response)
            
            # Count output tokens and track cost
            output_tokens = self.current_provider.count_tokens(response)
            cost = self.current_provider.estimate_cost(
                input_tokens,
                output_tokens
            )
            
            self.cost_tracker.add_cost(
                provider=self.current_provider.provider_name,
                model=self.current_provider.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost
            )
            
            logger.info(
                f"API call completed - Cost: ${cost:.6f}, "
                f"Tokens: {input_tokens}/{output_tokens}"
            )
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            logger.error(f"API call failed: {e}", exc_info=True)
    
    async def run(self) -> None:
        """Run the main conversation loop."""
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ["/quit", "/exit"]:
                    print("\n👋 Goodbye!\n")
                    break
                
                elif user_input.lower() == "/help":
                    self.print_help()
                    continue
                
                elif user_input.lower() == "/clear":
                    self.conversation.clear()
                    print("✓ Conversation history cleared\n")
                    continue
                
                elif user_input.lower() == "/model":
                    await self.switch_provider()
                    continue
                
                elif user_input.lower() == "/cost":
                    self.show_cost()
                    continue
                
                elif user_input.lower() == "/export":
                    self.export_conversation()
                    continue
                
                # Process regular message
                await self.process_message(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except EOFError:
                print("\n\n👋 Goodbye!\n")
                break
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        for provider in self.providers.values():
            await provider.close()
        logger.info("Application cleanup completed")


async def main() -> None:
    """Main entry point."""
    # Setup logging
    setup_logging(level=settings.log_level)
    
    # Validate configuration
    try:
        settings.validate_api_keys()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}\n")
        print("Please ensure your .env file contains the required API keys.")
        print("\nExample .env file:")
        print("OPENAI_API_KEY=sk-...")
        print("ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    
    # Initialize and run assistant
    assistant = AIAssistant()
    
    try:
        await assistant.initialize()
        await assistant.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal Error: {e}\n")
        sys.exit(1)
    finally:
        await assistant.cleanup()


if __name__ == "__main__":
    asyncio.run(main())