import streamlit as st
import asyncio
import nest_asyncio
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

nest_asyncio.apply()

# Load your API key from environment variable
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Gemini-compatible client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

# Streamlit UI
st.set_page_config(page_title="Gemini Agent", layout="centered")
st.title("🤖 Gemini Agent Assistant")
st.markdown("Chat with an intelligent agent powered by Gemini 2.5 that respond in haikus")

user_input = st.text_input("Enter your message")

if user_input:
    with st.spinner("Thinking..."):
        async def get_agent_response(prompt):
            agent = Agent(name="HaikuAssistant", instructions="You only respond in haikus.", model=model)
            result = await Runner.run(agent, prompt)
            return result.final_output

        response = asyncio.run(get_agent_response(user_input))
        st.success("Response:")
        st.write(response)
