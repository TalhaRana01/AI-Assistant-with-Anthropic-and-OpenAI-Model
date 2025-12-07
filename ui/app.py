import streamlit as st
from src.llm.openai_provider import OpenAIProvider
from src.config import Settings
from src.utils.conversation import ConversationManager
from dotenv import load_dotenv

load_dotenv()
settings = Settings()
llm = OpenAIProvider(api_key=settings.openai_api_key)
conv = ConversationManager()

st.title("AI Assistant 🤖")

user_input = st.text_input("Ask something...")

if st.button("Send"):
    conv.add_user_message(user_input)
    response = llm.generate(user_input)
    conv.add_bot_message(response)
    st.write("### Response:")
    st.write(response)

st.write("----")
st.write("### Conversation History")
for msg in conv.messages:
    role = msg["role"].capitalize()
    st.write(f"**{role}:** {msg['content']}")
