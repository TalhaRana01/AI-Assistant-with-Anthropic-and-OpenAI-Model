"""
Streamlit Frontend for AI Assistant
Modular design for easy model addition
"""

import asyncio
import streamlit as st
from datetime import datetime
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Import your existing modules
from src.config import settings
from src.llm.anthropic_provider import AnthropicProvider
from src.llm.openai_provider import OpenAIProvider
from src.utils.conversation import ConversationManager
from src.utils.cost_tracker import CostTracker
from src.utils.logger import setup_logging

# Page configuration
st.set_page_config(
    page_title="AI Assistant Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-align: center;
    }
    .provider-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .provider-card:hover {
        transform: scale(1.02);
    }
    .model-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .cost-warning {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #f39c12;
        color: #2d3436;
        font-weight: bold;
    }
    .cost-danger {
        background: linear-gradient(135deg, #ff7675 0%, #d63031 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #c0392b;
        color: white;
        font-weight: bold;
    }
    .cost-safe {
        background: linear-gradient(135deg, #55efc4 0%, #00b894 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00b894;
        color: white;
        font-weight: bold;
    }
    .message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .message-assistant {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #2d3436;
    }
    .config-section {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .info-badge {
        display: inline-block;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        color: #2d3436;
        margin: 0.2rem;
    }
    .status-bar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL CONFIGURATIONS - Easy to add new providers/models
# ============================================================================

MODEL_CONFIGS = {
    'openai': {
        'display_name': 'OpenAI',
        'icon': '🟢',
        'color': '#10a37f',
        'models': {
            'gpt-4o': {
                'name': 'GPT-4o',
                'description': 'Most capable, best for complex tasks',
                'input_cost': 2.50,
                'output_cost': 10.00,
                'icon': '🧠',
                'speed': 'Medium',
                'quality': 'Excellent',
                'context_window': '128K'
            },
            'gpt-4o-mini': {
                'name': 'GPT-4o Mini',
                'description': 'Fast and affordable, great for most tasks',
                'input_cost': 0.150,
                'output_cost': 0.600,
                'icon': '⚡',
                'speed': 'Fast',
                'quality': 'Good',
                'context_window': '128K'
            },
            'gpt-4-turbo': {
                'name': 'GPT-4 Turbo',
                'description': 'Powerful and versatile',
                'input_cost': 10.00,
                'output_cost': 30.00,
                'icon': '🚀',
                'speed': 'Medium',
                'quality': 'Excellent',
                'context_window': '128K'
            }
        }
    },
    'anthropic': {
        'display_name': 'Anthropic',
        'icon': '🟣',
        'color': '#7c3aed',
        'models': {
            'claude-sonnet-4-20250514': {
                'name': 'Claude Sonnet 4',
                'description': 'Latest and most intelligent',
                'input_cost': 3.00,
                'output_cost': 15.00,
                'icon': '🌟',
                'speed': 'Fast',
                'quality': 'Excellent',
                'context_window': '200K'
            },
            'claude-3-5-sonnet-20241022': {
                'name': 'Claude 3.5 Sonnet',
                'description': 'Balanced performance and speed',
                'input_cost': 3.00,
                'output_cost': 15.00,
                'icon': '💎',
                'speed': 'Fast',
                'quality': 'Excellent',
                'context_window': '200K'
            },
            'claude-3-5-haiku-20241022': {
                'name': 'Claude 3.5 Haiku',
                'description': 'Fast and efficient',
                'input_cost': 0.80,
                'output_cost': 4.00,
                'icon': '⚡',
                'speed': 'Very Fast',
                'quality': 'Good',
                'context_window': '200K'
            },
            'claude-3-opus-20240229': {
                'name': 'Claude 3 Opus',
                'description': 'Most powerful reasoning',
                'input_cost': 15.00,
                'output_cost': 75.00,
                'icon': '🏆',
                'speed': 'Slow',
                'quality': 'Best',
                'context_window': '200K'
            }
        }
    },
    # ============================================================================
    # PLACEHOLDER FOR FUTURE PROVIDERS
    # ============================================================================
    # 'google': {
    #     'display_name': 'Google',
    #     'icon': '🔵',
    #     'color': '#4285f4',
    #     'models': {
    #         'gemini-pro': {
    #             'name': 'Gemini Pro',
    #             'description': 'Google\'s advanced AI model',
    #             'input_cost': 0.50,
    #             'output_cost': 1.50,
    #             'icon': '💫',
    #             'speed': 'Fast',
    #             'quality': 'Excellent',
    #             'context_window': '32K'
    #         }
    #     }
    # },
    # 'cohere': {
    #     'display_name': 'Cohere',
    #     'icon': '🟡',
    #     'color': '#ff6b6b',
    #     'models': {
    #         'command-r-plus': {
    #             'name': 'Command R+',
    #             'description': 'Enterprise-grade AI',
    #             'input_cost': 3.00,
    #             'output_cost': 15.00,
    #             'icon': '🔥',
    #             'speed': 'Medium',
    #             'quality': 'Excellent',
    #             'context_window': '128K'
    #         }
    #     }
    # }
}

# ============================================================================
# PROVIDER INITIALIZATION FACTORY
# ============================================================================

async def initialize_provider(provider_name, model, temperature, max_tokens):
    """
    Factory function to initialize providers
    Easy to extend for new providers
    """
    try:
        if provider_name == 'openai':
            return OpenAIProvider(
                api_key=settings.get_api_key('openai'),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=settings.timeout,
                max_retries=settings.max_retries
            )
        elif provider_name == 'anthropic':
            return AnthropicProvider(
                api_key=settings.get_api_key('anthropic'),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=settings.timeout,
                max_retries=settings.max_retries
            )
        # ============================================================================
        # ADD NEW PROVIDERS HERE
        # ============================================================================
        # elif provider_name == 'google':
        #     return GoogleProvider(
        #         api_key=settings.get_api_key('google'),
        #         model=model,
        #         temperature=temperature,
        #         max_tokens=max_tokens
        #     )
        # elif provider_name == 'cohere':
        #     return CohereProvider(
        #         api_key=settings.get_api_key('cohere'),
        #         model=model,
        #         temperature=temperature,
        #         max_tokens=max_tokens
        #     )
        else:
            st.error(f"Unknown provider: {provider_name}")
            return None
    except Exception as e:
        st.error(f"{provider_name.title()} initialization failed: {e}")
        return None

def get_available_providers():
    """Check which providers have API keys configured"""
    available = {}
    
    if settings.openai_api_key and 'openai' in MODEL_CONFIGS:
        available['openai'] = MODEL_CONFIGS['openai']
    
    if settings.anthropic_api_key and 'anthropic' in MODEL_CONFIGS:
        available['anthropic'] = MODEL_CONFIGS['anthropic']
    
    # ============================================================================
    # CHECK FOR NEW PROVIDERS
    # ============================================================================
    # if hasattr(settings, 'google_api_key') and settings.google_api_key:
    #     available['google'] = MODEL_CONFIGS['google']
    # 
    # if hasattr(settings, 'cohere_api_key') and settings.cohere_api_key:
    #     available['cohere'] = MODEL_CONFIGS['cohere']
    
    return available

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'conversation' not in st.session_state:
    st.session_state.conversation = ConversationManager()

if 'cost_tracker' not in st.session_state:
    st.session_state.cost_tracker = CostTracker(
        warning_threshold=settings.cost_warning_threshold,
        hard_limit=settings.cost_hard_limit
    )

if 'current_provider' not in st.session_state:
    st.session_state.current_provider = None

if 'current_model' not in st.session_state:
    st.session_state.current_model = None

if 'temperature' not in st.session_state:
    st.session_state.temperature = settings.temperature

if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = settings.max_tokens

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Initialize on first run
if not st.session_state.initialized:
    available_providers = get_available_providers()
    
    if not available_providers:
        st.error("❌ No API keys configured! Please add keys to .env file")
        st.info("""
        **Required Environment Variables:**
        - OPENAI_API_KEY=your_key_here
        - ANTHROPIC_API_KEY=your_key_here
        """)
        st.stop()
    
    # Set default provider
    if 'openai' in available_providers:
        st.session_state.current_provider = 'openai'
        st.session_state.current_model = 'gpt-4o-mini'
    elif 'anthropic' in available_providers:
        st.session_state.current_provider = 'anthropic'
        st.session_state.current_model = 'claude-3-5-haiku-20241022'
    else:
        st.session_state.current_provider = list(available_providers.keys())[0]
        st.session_state.current_model = list(available_providers[st.session_state.current_provider]['models'].keys())[0]
    
    st.session_state.initialized = True

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown('<p class="main-header">⚙️ Configuration</p>', unsafe_allow_html=True)
    
    # Provider Selection
    st.markdown("### 🔄 Select Provider")
    
    available_providers = get_available_providers()
    
    # Create columns for provider buttons
    cols = st.columns(len(available_providers))
    
    for idx, (prov_key, prov_config) in enumerate(available_providers.items()):
        with cols[idx]:
            if st.button(
                f"{prov_config['icon']} {prov_config['display_name']}", 
                use_container_width=True,
                type="primary" if st.session_state.current_provider == prov_key else "secondary"
            ):
                st.session_state.current_provider = prov_key
                # Set default model for this provider
                st.session_state.current_model = list(prov_config['models'].keys())[0]
                st.rerun()
    
    st.divider()
    
    # Model Selection
    st.markdown("### 🤖 Select Model")
    
    if st.session_state.current_provider:
        provider_config = MODEL_CONFIGS[st.session_state.current_provider]
        available_models = provider_config['models']
        
        for model_id, config in available_models.items():
            with st.expander(
                f"{config['icon']} {config['name']}", 
                expanded=(model_id == st.session_state.current_model)
            ):
                st.markdown(f"**{config['description']}**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"<span class='info-badge'>⚡ {config['speed']}</span>", 
                              unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<span class='info-badge'>✨ {config['quality']}</span>", 
                              unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<span class='info-badge'>📊 {config['context_window']}</span>", 
                              unsafe_allow_html=True)
                
                st.caption(f"💰 Cost: ${config['input_cost']:.2f}/1M in • ${config['output_cost']:.2f}/1M out")
                
                if st.button(
                    f"Use {config['name']}", 
                    key=f"select_{model_id}", 
                    use_container_width=True,
                    type="primary" if model_id == st.session_state.current_model else "secondary"
                ):
                    st.session_state.current_model = model_id
                    st.success(f"✅ Switched to {config['name']}")
                    st.rerun()
    
    st.divider()
    
    # Advanced Settings
    st.markdown("### 🎛️ Advanced Settings")
    
    with st.container():
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        
        # Temperature
        st.markdown("**🌡️ Temperature**")
        st.caption("Controls randomness (0 = focused, 2 = creative)")
        temperature = st.slider(
            "temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.temperature,
            step=0.1,
            label_visibility="collapsed"
        )
        
        if temperature != st.session_state.temperature:
            st.session_state.temperature = temperature
        
        # Temperature indicator
        if temperature < 0.3:
            st.info("❄️ Very Focused - Deterministic responses")
        elif temperature < 0.7:
            st.info("🎯 Balanced - Good for most tasks")
        elif temperature < 1.3:
            st.info("🎨 Creative - More varied responses")
        else:
            st.info("🌈 Very Creative - Highly varied outputs")
        
        st.divider()
        
        # Max Tokens
        st.markdown("**📊 Max Output Tokens**")
        st.caption("Maximum length of response")
        max_tokens = st.slider(
            "max_tokens",
            min_value=100,
            max_value=4000,
            value=st.session_state.max_tokens,
            step=100,
            label_visibility="collapsed"
        )
        
        if max_tokens != st.session_state.max_tokens:
            st.session_state.max_tokens = max_tokens
        
        # Token indicator
        if max_tokens < 500:
            st.caption("📝 Short responses")
        elif max_tokens < 1500:
            st.caption("📄 Medium responses")
        elif max_tokens < 3000:
            st.caption("📚 Long responses")
        else:
            st.caption("📖 Very long responses")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Current Configuration Summary
    st.markdown("### 📋 Current Config")
    
    if st.session_state.current_provider and st.session_state.current_model:
        provider_config = MODEL_CONFIGS[st.session_state.current_provider]
        model_config = provider_config['models'][st.session_state.current_model]
        
        st.markdown(f"""
        <div class="model-card">
            <h4>{model_config['icon']} {model_config['name']}</h4>
            <p><strong>Provider:</strong> {provider_config['display_name']}</p>
            <p><strong>Model ID:</strong> {st.session_state.current_model}</p>
            <p><strong>Temperature:</strong> {st.session_state.temperature}</p>
            <p><strong>Max Tokens:</strong> {st.session_state.max_tokens}</p>
            <p><strong>Context:</strong> {model_config['context_window']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Cost Tracking
    st.markdown("### 💰 Cost Tracking")
    tracker = st.session_state.cost_tracker
    
    # Cost status indicator
    if tracker.hard_limit:
        percentage = (tracker.total_cost / tracker.hard_limit) * 100
        if percentage >= 100:
            st.markdown('<div class="cost-danger">⛔ LIMIT REACHED!</div>', unsafe_allow_html=True)
        elif percentage >= 80:
            st.markdown(f'<div class="cost-warning">⚠️ {percentage:.1f}% Used</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="cost-safe">✅ {percentage:.1f}% Used</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Cost", f"${tracker.total_cost:.6f}", 
                 delta=f"{len(tracker.entries)} calls")
    with col2:
        if tracker.hard_limit:
            remaining = tracker.hard_limit - tracker.total_cost
            st.metric("Remaining", f"${remaining:.6f}")
        else:
            st.metric("Limit", "None")
    
    st.metric("Input Tokens", f"{tracker.total_input_tokens:,}")
    st.metric("Output Tokens", f"{tracker.total_output_tokens:,}")
    
    # Cost breakdown
    with st.expander("📊 Cost Breakdown"):
        st.markdown("**By Provider:**")
        for prov, cost in tracker.get_cost_by_provider().items():
            st.text(f"  {prov}: ${cost:.6f}")
        
        st.markdown("**By Model:**")
        for model, cost in tracker.get_cost_by_model().items():
            st.text(f"  {model}: ${cost:.6f}")
    
    st.divider()
    
    # Actions
    st.markdown("### 🔧 Actions")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.conversation.clear()
        st.success("✅ Conversation cleared!")
        st.rerun()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 JSON", use_container_width=True):
            if st.session_state.conversation.count_messages() > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"chat_{timestamp}.json"
                st.session_state.conversation.export_to_json(filepath)
                st.success(f"✅ Saved!")
            else:
                st.warning("No messages")
    
    with col2:
        if st.button("📄 MD", use_container_width=True):
            if st.session_state.conversation.count_messages() > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"chat_{timestamp}.md"
                st.session_state.conversation.export_to_markdown(filepath)
                st.success(f"✅ Saved!")
            else:
                st.warning("No messages")
    
    if st.button("🔄 Reset Costs", use_container_width=True):
        if tracker.total_cost > 0:
            tracker.reset()
            st.success("✅ Costs reset!")
            st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<p class="main-header">🤖 AI Assistant Pro</p>', unsafe_allow_html=True)

# Status bar
if st.session_state.current_provider and st.session_state.current_model:
    provider_config = MODEL_CONFIGS[st.session_state.current_provider]
    model_config = provider_config['models'][st.session_state.current_model]
    
    st.markdown(f"""
    <div class="status-bar">
        <strong>{provider_config['icon']} Provider:</strong> {provider_config['display_name']} &nbsp;|&nbsp;
        <strong>{model_config['icon']} Model:</strong> {model_config['name']} &nbsp;|&nbsp;
        <strong>💬 Messages:</strong> {st.session_state.conversation.count_messages()} &nbsp;|&nbsp;
        <strong>💰 Cost:</strong> ${st.session_state.cost_tracker.total_cost:.6f}
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Display conversation
messages_container = st.container()

with messages_container:
    if st.session_state.conversation.count_messages() == 0:
        st.markdown("""
        <div class="model-card" style="text-align: center; padding: 2rem;">
            <h2>👋 Welcome to AI Assistant Pro!</h2>
            <p>Configure your model in the sidebar and start chatting below.</p>
            <p>💡 <strong>Tip:</strong> Try different models and temperatures for different tasks!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.conversation.messages:
            if msg.role == "user":
                st.markdown(f"""
                <div class="message-user">
                    <strong>👤 You</strong> <small>({msg.timestamp.strftime('%H:%M:%S')})</small><br><br>
                    {msg.content}
                </div>
                """, unsafe_allow_html=True)
            elif msg.role == "assistant":
                st.markdown(f"""
                <div class="message-assistant">
                    <strong>🤖 Assistant</strong> <small>({msg.timestamp.strftime('%H:%M:%S')})</small><br><br>
                    {msg.content}
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# MESSAGE PROCESSING
# ============================================================================

async def process_message(user_input):
    """Process user message and get response"""
    # Check hard limit
    if not st.session_state.cost_tracker.check_hard_limit():
        st.error(f"⛔ Hard cost limit of ${st.session_state.cost_tracker.hard_limit:.2f} exceeded!")
        return
    
    # Initialize provider with current settings
    provider = await initialize_provider(
        st.session_state.current_provider,
        st.session_state.current_model,
        st.session_state.temperature,
        st.session_state.max_tokens
    )
    
    if not provider:
        st.error("❌ Failed to initialize provider")
        return
    
    # Add user message
    st.session_state.conversation.add_user_message(user_input)
    
    # Count input tokens
    messages = st.session_state.conversation.get_messages()
    input_tokens = sum(provider.count_tokens(msg['content']) for msg in messages)
    
    try:
        # Generate response
        with st.spinner('🤔 Thinking...'):
            response = await provider.generate(messages)
        
        # Add assistant message
        st.session_state.conversation.add_assistant_message(response)
        
        # Track cost
        output_tokens = provider.count_tokens(response)
        cost = provider.estimate_cost(input_tokens, output_tokens)
        
        st.session_state.cost_tracker.add_cost(
            provider=provider.provider_name,
            model=provider.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost
        )
        
        # Close provider
        await provider.close()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        if provider:
            await provider.close()

# ============================================================================
# CHAT INPUT
# ============================================================================

st.divider()

with st.form(key='chat_form', clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Message",
            placeholder="Type your message here... (Press Enter or click Send)",
            label_visibility="collapsed"
        )
    
    with col2:
        submit = st.form_submit_button("Send 🚀", use_container_width=True, type="primary")

if submit and user_input:
    if not st.session_state.current_provider or not st.session_state.current_model:
        st.error("❌ Please select a provider and model first!")
    else:
        asyncio.run(process_message(user_input))
        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
    💡 <strong>Pro Tip:</strong> Lower temperature (0-0.5) for factual tasks, higher (1-2) for creative work |
    ⚙️ Adjust settings in sidebar for optimal results
</div>
<<<<<<< HEAD
""", unsafe_allow_html=True)
=======
""", unsafe_allow_html=True)





# """
# Streamlit Frontend for AI Assistant
# Complete web interface with advanced model configuration
# """

# import asyncio
# import streamlit as st
# from datetime import datetime
# import json
# import os
# import sys


# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(ROOT_DIR)

# # Import your existing modules
# from src.config import settings

# from src.llm.anthropic_provider import AnthropicProvider
# from src.llm.openai_provider import OpenAIProvider
# from src.utils.conversation import ConversationManager
# from src.utils.cost_tracker import CostTracker
# from src.utils.logger import setup_logging

# # Page configuration
# st.set_page_config(
#     page_title="AI Assistant Pro",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.8rem;
#         font-weight: bold;
#         background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 1rem;
#         text-align: center;
#     }
#     .metric-card {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 1.5rem;
#         border-radius: 15px;
#         color: white;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         margin: 0.5rem 0;
#     }
#     .provider-card {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#         padding: 1rem;
#         border-radius: 10px;
#         color: white;
#         margin: 0.5rem 0;
#         cursor: pointer;
#         transition: transform 0.2s;
#     }
#     .provider-card:hover {
#         transform: scale(1.02);
#     }
#     .model-card {
#         background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
#         padding: 1rem;
#         border-radius: 10px;
#         color: white;
#         margin: 0.5rem 0;
#     }
#     .cost-warning {
#         background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 5px solid #f39c12;
#         color: #2d3436;
#         font-weight: bold;
#     }
#     .cost-danger {
#         background: linear-gradient(135deg, #ff7675 0%, #d63031 100%);
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 5px solid #c0392b;
#         color: white;
#         font-weight: bold;
#     }
#     .cost-safe {
#         background: linear-gradient(135deg, #55efc4 0%, #00b894 100%);
#         padding: 1rem;
#         border-radius: 10px;
#         border-left: 5px solid #00b894;
#         color: white;
#         font-weight: bold;
#     }
#     .message-user {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 1.2rem;
#         border-radius: 15px;
#         margin: 0.8rem 0;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
#     .message-assistant {
#         background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
#         padding: 1.2rem;
#         border-radius: 15px;
#         margin: 0.8rem 0;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         color: #2d3436;
#     }
#     .config-section {
#         background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
#         padding: 1.5rem;
#         border-radius: 15px;
#         margin: 1rem 0;
#         box-shadow: 0 4px 8px rgba(0,0,0,0.1);
#     }
#     .stSlider > div > div > div {
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#     }
#     .info-badge {
#         display: inline-block;
#         background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
#         padding: 0.3rem 0.8rem;
#         border-radius: 20px;
#         font-size: 0.85rem;
#         font-weight: bold;
#         color: #2d3436;
#         margin: 0.2rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Model configurations
# MODEL_CONFIGS = {
#     'openai': {
#         'gpt-4o': {
#             'name': 'GPT-4o',
#             'description': 'Most capable, best for complex tasks',
#             'input_cost': 2.50,
#             'output_cost': 10.00,
#             'icon': '🧠',
#             'speed': 'Medium',
#             'quality': 'Excellent'
#         },
#         'gpt-4o-mini': {
#             'name': 'GPT-4o Mini',
#             'description': 'Fast and affordable, great for most tasks',
#             'input_cost': 0.150,
#             'output_cost': 0.600,
#             'icon': '⚡',
#             'speed': 'Fast',
#             'quality': 'Good'
#         },
#         'gpt-4-turbo': {
#             'name': 'GPT-4 Turbo',
#             'description': 'Powerful and versatile',
#             'input_cost': 10.00,
#             'output_cost': 30.00,
#             'icon': '🚀',
#             'speed': 'Medium',
#             'quality': 'Excellent'
#         }
#     },
#     'anthropic': {
#         'claude-sonnet-4-20250514': {
#             'name': 'Claude Sonnet 4',
#             'description': 'Latest and most intelligent',
#             'input_cost': 3.00,
#             'output_cost': 15.00,
#             'icon': '🌟',
#             'speed': 'Fast',
#             'quality': 'Excellent'
#         },
#         'claude-3-5-sonnet-20241022': {
#             'name': 'Claude 3.5 Sonnet',
#             'description': 'Balanced performance',
#             'input_cost': 3.00,
#             'output_cost': 15.00,
#             'icon': '💎',
#             'speed': 'Fast',
#             'quality': 'Excellent'
#         },
#         'claude-3-5-haiku-20241022': {
#             'name': 'Claude 3.5 Haiku',
#             'description': 'Fast and efficient',
#             'input_cost': 0.80,
#             'output_cost': 4.00,
#             'icon': '⚡',
#             'speed': 'Very Fast',
#             'quality': 'Good'
#         },
#         'claude-3-opus-20240229': {
#             'name': 'Claude 3 Opus',
#             'description': 'Most powerful reasoning',
#             'input_cost': 15.00,
#             'output_cost': 75.00,
#             'icon': '🏆',
#             'speed': 'Slow',
#             'quality': 'Best'
#         }
#     }
# }

# # Initialize session state
# if 'conversation' not in st.session_state:
#     st.session_state.conversation = ConversationManager()

# if 'cost_tracker' not in st.session_state:
#     st.session_state.cost_tracker = CostTracker(
#         warning_threshold=settings.cost_warning_threshold,
#         hard_limit=settings.cost_hard_limit
#     )

# if 'providers' not in st.session_state:
#     st.session_state.providers = {}
    
# if 'current_provider' not in st.session_state:
#     st.session_state.current_provider = None

# if 'current_model' not in st.session_state:
#     st.session_state.current_model = None

# if 'temperature' not in st.session_state:
#     st.session_state.temperature = settings.temperature

# if 'max_tokens' not in st.session_state:
#     st.session_state.max_tokens = settings.max_tokens

# if 'initialized' not in st.session_state:
#     st.session_state.initialized = False

# # Initialize providers
# async def initialize_provider(provider_name, model, temperature, max_tokens):
#     """Initialize a specific provider with given settings"""
#     try:
#         if provider_name == 'openai':
#             return OpenAIProvider(
#                 api_key=settings.get_api_key('openai'),
#                 model=model,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 timeout=settings.timeout,
#                 max_retries=settings.max_retries
#             )
#         elif provider_name == 'anthropic':
#             return AnthropicProvider(
#                 api_key=settings.get_api_key('anthropic'),
#                 model=model,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 timeout=settings.timeout,
#                 max_retries=settings.max_retries
#             )
#     except Exception as e:
#         st.error(f"{provider_name.title()} initialization failed: {e}")
#         return None

# # Initialize on first run
# if not st.session_state.initialized:
#     if settings.openai_api_key:
#         st.session_state.current_provider = 'openai'
#         st.session_state.current_model = settings.openai_model
#     elif settings.anthropic_api_key:
#         st.session_state.current_provider = 'anthropic'
#         st.session_state.current_model = settings.anthropic_model
#     else:
#         st.error("❌ No API keys configured! Please add keys to .env file")
#         st.stop()
    
#     st.session_state.initialized = True

# # Sidebar - Advanced Configuration
# with st.sidebar:
#     st.markdown('<p class="main-header">⚙️ Configuration</p>', unsafe_allow_html=True)
    
#     # Provider Selection
#     st.markdown("### 🔄 Select Provider")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         if settings.openai_api_key:
#             if st.button("🟢 OpenAI", use_container_width=True, 
#                         type="primary" if st.session_state.current_provider == 'openai' else "secondary"):
#                 st.session_state.current_provider = 'openai'
#                 st.session_state.current_model = 'gpt-4o-mini'
#                 st.rerun()
    
#     with col2:
#         if settings.anthropic_api_key:
#             if st.button("🟣 Anthropic", use_container_width=True,
#                         type="primary" if st.session_state.current_provider == 'anthropic' else "secondary"):
#                 st.session_state.current_provider = 'anthropic'
#                 st.session_state.current_model = 'claude-3-5-haiku-20241022'
#                 st.rerun()
    
#     st.divider()
    
#     # Model Selection
#     st.markdown("### 🤖 Select Model")
    
#     if st.session_state.current_provider:
#         available_models = MODEL_CONFIGS.get(st.session_state.current_provider, {})
        
#         for model_id, config in available_models.items():
#             with st.expander(f"{config['icon']} {config['name']}", 
#                            expanded=(model_id == st.session_state.current_model)):
#                 st.markdown(f"**{config['description']}**")
                
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.markdown(f"<span class='info-badge'>Speed: {config['speed']}</span>", 
#                               unsafe_allow_html=True)
#                 with col2:
#                     st.markdown(f"<span class='info-badge'>Quality: {config['quality']}</span>", 
#                               unsafe_allow_html=True)
                
#                 st.caption(f"💰 Cost: ${config['input_cost']}/1M input, ${config['output_cost']}/1M output")
                
#                 if st.button(f"Use {config['name']}", key=f"select_{model_id}", use_container_width=True):
#                     st.session_state.current_model = model_id
#                     st.success(f"✅ Switched to {config['name']}")
#                     st.rerun()
    
#     st.divider()
    
#     # Advanced Settings
#     st.markdown("### 🎛️ Advanced Settings")
    
#     with st.container():
#         st.markdown('<div class="config-section">', unsafe_allow_html=True)
        
#         # Temperature
#         st.markdown("**🌡️ Temperature**")
#         st.caption("Controls randomness (0 = focused, 2 = creative)")
#         temperature = st.slider(
#             "temperature",
#             min_value=0.0,
#             max_value=2.0,
#             value=st.session_state.temperature,
#             step=0.1,
#             label_visibility="collapsed"
#         )
        
#         if temperature != st.session_state.temperature:
#             st.session_state.temperature = temperature
        
#         # Show temperature indicator
#         if temperature < 0.3:
#             st.info("❄️ Very Focused - Deterministic responses")
#         elif temperature < 0.7:
#             st.info("🎯 Balanced - Good for most tasks")
#         elif temperature < 1.3:
#             st.info("🎨 Creative - More varied responses")
#         else:
#             st.info("🌈 Very Creative - Highly varied outputs")
        
#         st.divider()
        
#         # Max Tokens
#         st.markdown("**📊 Max Output Tokens**")
#         st.caption("Maximum length of response")
#         max_tokens = st.slider(
#             "max_tokens",
#             min_value=100,
#             max_value=4000,
#             value=st.session_state.max_tokens,
#             step=100,
#             label_visibility="collapsed"
#         )
        
#         if max_tokens != st.session_state.max_tokens:
#             st.session_state.max_tokens = max_tokens
        
#         # Show token indicator
#         if max_tokens < 500:
#             st.caption("📝 Short responses")
#         elif max_tokens < 1500:
#             st.caption("📄 Medium responses")
#         elif max_tokens < 3000:
#             st.caption("📚 Long responses")
#         else:
#             st.caption("📖 Very long responses")
        
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     st.divider()
    
#     # Current Configuration Summary
#     st.markdown("### 📋 Current Config")
    
#     if st.session_state.current_provider and st.session_state.current_model:
#         config = MODEL_CONFIGS[st.session_state.current_provider][st.session_state.current_model]
        
#         st.markdown(f"""
#         <div class="model-card">
#             <h4>{config['icon']} {config['name']}</h4>
#             <p><strong>Provider:</strong> {st.session_state.current_provider.title()}</p>
#             <p><strong>Temperature:</strong> {st.session_state.temperature}</p>
#             <p><strong>Max Tokens:</strong> {st.session_state.max_tokens}</p>
#             <p><strong>Speed:</strong> {config['speed']}</p>
#             <p><strong>Quality:</strong> {config['quality']}</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     st.divider()
    
#     # Cost Tracking
#     st.markdown("### 💰 Cost Tracking")
#     tracker = st.session_state.cost_tracker
    
#     # Cost status indicator
#     if tracker.hard_limit:
#         percentage = (tracker.total_cost / tracker.hard_limit) * 100
#         if percentage >= 100:
#             st.markdown('<div class="cost-danger">⛔ LIMIT REACHED!</div>', unsafe_allow_html=True)
#         elif percentage >= 80:
#             st.markdown(f'<div class="cost-warning">⚠️ {percentage:.1f}% Used</div>', unsafe_allow_html=True)
#         else:
#             st.markdown(f'<div class="cost-safe">✅ {percentage:.1f}% Used</div>', unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Total Cost", f"${tracker.total_cost:.6f}", 
#                  delta=f"{len(tracker.entries)} calls")
#     with col2:
#         if tracker.hard_limit:
#             remaining = tracker.hard_limit - tracker.total_cost
#             st.metric("Remaining", f"${remaining:.6f}")
#         else:
#             st.metric("Limit", "None")
    
#     st.metric("Input Tokens", f"{tracker.total_input_tokens:,}")
#     st.metric("Output Tokens", f"{tracker.total_output_tokens:,}")
    
#     # Cost breakdown
#     with st.expander("📊 Cost Breakdown"):
#         st.markdown("**By Provider:**")
#         for prov, cost in tracker.get_cost_by_provider().items():
#             st.text(f"  {prov}: ${cost:.6f}")
        
#         st.markdown("**By Model:**")
#         for model, cost in tracker.get_cost_by_model().items():
#             st.text(f"  {model}: ${cost:.6f}")
    
#     st.divider()
    
#     # Actions
#     st.markdown("### 🔧 Actions")
    
#     if st.button("🗑️ Clear Chat", use_container_width=True):
#         st.session_state.conversation.clear()
#         st.success("✅ Conversation cleared!")
#         st.rerun()
    
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("💾 JSON", use_container_width=True):
#             if st.session_state.conversation.count_messages() > 0:
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 filepath = f"chat_{timestamp}.json"
#                 st.session_state.conversation.export_to_json(filepath)
#                 st.success(f"✅ Saved!")
#             else:
#                 st.warning("No messages")
    
#     with col2:
#         if st.button("📄 MD", use_container_width=True):
#             if st.session_state.conversation.count_messages() > 0:
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 filepath = f"chat_{timestamp}.md"
#                 st.session_state.conversation.export_to_markdown(filepath)
#                 st.success(f"✅ Saved!")
#             else:
#                 st.warning("No messages")
    
#     if st.button("🔄 Reset Costs", use_container_width=True):
#         if tracker.total_cost > 0:
#             tracker.reset()
#             st.success("✅ Costs reset!")
#             st.rerun()

# # Main content
# st.markdown('<p class="main-header">🤖 AI Assistant Pro</p>', unsafe_allow_html=True)

# # Status bar
# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     if st.session_state.current_provider:
#         st.markdown(f"**Provider:** {st.session_state.current_provider.title()} 🟢")
# with col2:
#     if st.session_state.current_model:
#         config = MODEL_CONFIGS[st.session_state.current_provider][st.session_state.current_model]
#         st.markdown(f"**Model:** {config['icon']} {config['name']}")
# with col3:
#     st.markdown(f"**Messages:** {st.session_state.conversation.count_messages()}")
# with col4:
#     st.markdown(f"**Cost:** ${st.session_state.cost_tracker.total_cost:.4f}")

# st.divider()

# # Display conversation
# messages_container = st.container()

# with messages_container:
#     if st.session_state.conversation.count_messages() == 0:
#         st.markdown("""
#         <div class="model-card" style="text-align: center; padding: 2rem;">
#             <h2>👋 Welcome to AI Assistant Pro!</h2>
#             <p>Configure your model in the sidebar and start chatting below.</p>
#             <p>💡 <strong>Tip:</strong> Try different models and temperatures for different tasks!</p>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         for msg in st.session_state.conversation.messages:
#             if msg.role == "user":
#                 st.markdown(f"""
#                 <div class="message-user">
#                     <strong>👤 You</strong> <small>({msg.timestamp.strftime('%H:%M:%S')})</small><br><br>
#                     {msg.content}
#                 </div>
#                 """, unsafe_allow_html=True)
#             elif msg.role == "assistant":
#                 st.markdown(f"""
#                 <div class="message-assistant">
#                     <strong>🤖 Assistant</strong> <small>({msg.timestamp.strftime('%H:%M:%S')})</small><br><br>
#                     {msg.content}
#                 </div>
#                 """, unsafe_allow_html=True)

# # Input area
# st.divider()

# async def process_message(user_input):
#     """Process user message and get response"""
#     # Check hard limit
#     if not st.session_state.cost_tracker.check_hard_limit():
#         st.error(f"⛔ Hard cost limit of ${st.session_state.cost_tracker.hard_limit:.2f} exceeded!")
#         return
    
#     # Initialize provider with current settings
#     provider = await initialize_provider(
#         st.session_state.current_provider,
#         st.session_state.current_model,
#         st.session_state.temperature,
#         st.session_state.max_tokens
#     )
    
#     if not provider:
#         st.error("❌ Failed to initialize provider")
#         return
    
#     # Add user message
#     st.session_state.conversation.add_user_message(user_input)
    
#     # Count input tokens
#     messages = st.session_state.conversation.get_messages()
#     input_tokens = sum(provider.count_tokens(msg['content']) for msg in messages)
    
#     try:
#         # Generate response
#         with st.spinner('🤔 Thinking...'):
#             response = await provider.generate(messages)
        
#         # Add assistant message
#         st.session_state.conversation.add_assistant_message(response)
        
#         # Track cost
#         output_tokens = provider.count_tokens(response)
#         cost = provider.estimate_cost(input_tokens, output_tokens)
        
#         st.session_state.cost_tracker.add_cost(
#             provider=provider.provider_name,
#             model=provider.model,
#             input_tokens=input_tokens,
#             output_tokens=output_tokens,
#             cost=cost
#         )
        
#         # Close provider
#         await provider.close()
        
#     except Exception as e:
#         st.error(f"❌ Error: {str(e)}")
#         await provider.close()

# # Chat input
# with st.form(key='chat_form', clear_on_submit=True):
#     col1, col2 = st.columns([6, 1])
    
#     with col1:
#         user_input = st.text_input(
#             "Message",
#             placeholder="Type your message here... (Press Enter or click Send)",
#             label_visibility="collapsed"
#         )
    
#     with col2:
#         submit = st.form_submit_button("Send 🚀", use_container_width=True, type="primary")

# if submit and user_input:
#     if not st.session_state.current_provider or not st.session_state.current_model:
#         st.error("❌ Please select a provider and model first!")
#     else:
#         asyncio.run(process_message(user_input))
#         st.rerun()

# # Footer
# st.divider()
# st.markdown("""
# <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
#     💡 <strong>Pro Tip:</strong> Lower temperature (0-0.5) for factual tasks, higher (1-2) for creative work |
#     ⚙️ Adjust settings in sidebar for optimal results
# </div>
# """, unsafe_allow_html=True)



>>>>>>> a1dbe58 ( ui code update version)
