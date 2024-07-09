import streamlit as st
import os
from typing import Generator
from groq import Groq
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import random

load_dotenv()

st.set_page_config(page_icon="🏎️", layout="wide", page_title="EnGuru")

def icon(emoji: str):
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🏎️")

st.subheader("EnGuru", divider="rainbow", anchor=False)

groq_api_key = st.secrets["api_credentials"]["groq_api"]
client = Groq(api_key=groq_api_key)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=20, memory_key="chat_history", return_messages=True)

if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = 0

# Define model details
models = {
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}

# Sidebar content
proficiency_levels = ["Beginner", "Intermediate", "Advanced", "Fluent"]
selected_level = st.sidebar.selectbox("Select your English proficiency level:", proficiency_levels)

st.sidebar.info(f"You've selected the {selected_level} level. The English Teacher will tailor responses to this level.")

st.sidebar.markdown("---")

motivational_quotes = [
    "The limits of my language mean the limits of my world. - Ludwig Wittgenstein",
    "To have another language is to possess a second soul. - Charlemagne",
    "Language is the road map of a culture. It tells you where its people come from and where they are going. - Rita Mae Brown",
    "Learning another language is not only learning different words for the same things, but learning another way to think about things. - Flora Lewis",
]
st.sidebar.markdown("### Quote of the Day")
st.sidebar.markdown(f"*{random.choice(motivational_quotes)}*")

st.sidebar.markdown("---")
st.sidebar.markdown("Built by Sujay.")

# Model selection
model_option = st.selectbox(
    "Choose a model:",
    options=list(models.keys()),
    format_func=lambda x: models[x]["name"],
    index=2  
)

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option
    st.session_state.memory.clear()

# Topic input
topic = st.text_input("Enter the topic you want to learn English about:")

# User input for conversation
prompt = st.chat_input("Your response:")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

system_prompt = f"""You are an English Teacher. The user wants to learn about the topic: {topic}. Their proficiency level is {selected_level}. 
Your role is to engage the user in a conversation about the topic, asking focused, concise questions to encourage their English practice.

Strict Guidelines:
1. Start with a brief greeting and a short, engaging question about the topic.
2. Keep your responses and questions concise, Only one short question.
3. Ask only one question at a time to maintain focus.
4. Tailor your language complexity to the user's proficiency level.
5. Aim to elicit more detailed responses from the user to practice their English skills.

your goal is to stimulate conversation and English practice, not to provide lengthy explanations or no context."""

if topic and not st.session_state.messages:  # Only generate initial message if there are no messages
    try:
        groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model_option)

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template("Let's start discussing the topic: {topic}"),
        ])

        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt_template,
            verbose=False,
        )

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                response = conversation.predict(topic=topic)
                st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.memory.chat_memory.add_ai_message(response)

    except Exception as e:
        st.error(f"An error occurred: {e}", icon="🚨")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.user_inputs += 1
    
    # Display user message
    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                chat_completion = client.chat.completions.create(
                    model=model_option,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                    ],
                    max_tokens=models[model_option]["tokens"],
                    stream=True
                )
                chat_responses_generator = generate_chat_responses(chat_completion)
                full_response = st.write_stream(chat_responses_generator)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.memory.chat_memory.add_user_message(prompt)
        st.session_state.memory.chat_memory.add_ai_message(full_response)

    except Exception as e:
        st.error(f"An error occurred: {e}", icon="🚨")

# Feedback button logic
if st.session_state.user_inputs >= 5:
    if st.button("Get Feedback on Your English"):
        feedback_prompt = f"""Analyze the user's English proficiency based on their responses. Provide detailed feedback on grammar, vocabulary, and sentence structure. Give specific examples of errors and suggest improvements.

Guidelines:
1. Evaluate grammar usage (e.g., verb tenses, subject-verb agreement, article usage).
2. Assess vocabulary range and appropriateness for the {selected_level} level.
3. Comment on sentence structure and complexity.
4. Provide at least 3 specific examples of errors or areas for improvement.
5. Suggest alternative phrasings or vocabulary to enhance their English.
6. Score grammar and vocabulary on a scale of 1-10, considering the user's proficiency level.
7. Offer tailored advice for improvement based on the {selected_level} level.

User's responses:
{[m["content"] for m in st.session_state.messages if m["role"] == "user"]}

Format your response as follows:
Grammar Score: [1-10]
Vocabulary Score: [1-10]
Overall Analysis: [Your analysis here]
Specific Examples and Suggestions:
1. [Example 1]: [Suggestion 1]
2. [Example 2]: [Suggestion 2]
3. [Example 3]: [Suggestion 3]
Advice for Improvement: [Tailored advice here]"""

        try:
            with st.spinner("Analyzing your English..."):
                feedback_response = client.chat.completions.create(
                    model=model_option,
                    messages=[
                        {"role": "system", "content": feedback_prompt}
                    ],
                    max_tokens=models[model_option]["tokens"]
                )
                feedback_content = feedback_response.choices[0].message.content
                
                # Parse the feedback content
                feedback_lines = feedback_content.split('\n')
                grammar_score = "N/A"
                vocab_score = "N/A"
                overall_analysis = ""
                examples_and_suggestions = []
                advice = ""
                
                current_section = ""
                for line in feedback_lines:
                    if line.startswith("Grammar Score:"):
                        grammar_score = line.split(":")[1].strip()
                    elif line.startswith("Vocabulary Score:"):
                        vocab_score = line.split(":")[1].strip()
                    elif line.startswith("Overall Analysis:"):
                        current_section = "analysis"
                    elif line.startswith("Specific Examples and Suggestions:"):
                        current_section = "examples"
                    elif line.startswith("Advice for Improvement:"):
                        current_section = "advice"
                    elif current_section == "analysis":
                        overall_analysis += line + " "
                    elif current_section == "examples" and line.strip():
                        examples_and_suggestions.append(line.strip())
                    elif current_section == "advice":
                        advice += line + " "

                st.success(f"Grammar Score: {grammar_score}/10\nVocabulary Score: {vocab_score}/10")
                st.markdown("### Overall Analysis")
                st.write(overall_analysis.strip())
                st.markdown("### Specific Examples and Suggestions")
                for example in examples_and_suggestions:
                    st.markdown(f"- {example}")
                st.markdown("### Advice for Improvement")
                st.write(advice.strip())

        except Exception as e:
            st.error(f"An error occurred while generating feedback: {e}", icon="🚨")
