import streamlit as st
import os
# to specify the type of Generator Function
from typing import Generator, List, Tuple
from groq import Groq
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from time import sleep

load_dotenv()

st.set_page_config(page_icon="🏎️", layout="wide", page_title="Enguru")

def icon(emoji: str):
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🏎️")

st.subheader("EnGuru", divider="rainbow", anchor=False)

groq_api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=20, memory_key="chat_history", return_messages=True)
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = 0
if "topic_selected" not in st.session_state:
    st.session_state.topic_selected = False
if "story_mode" not in st.session_state:
    st.session_state.story_mode = False
if "story_parts" not in st.session_state:
    st.session_state.story_parts = []
if "current_part" not in st.session_state:
    st.session_state.current_part = 0
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "embeddings" not in st.session_state:
    st.session_state.embeddings = {}
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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

# Topic selection
topics = ["Basic_Greetings", "Hobbies", "Family", "Weather", "Restaurant_Ordering"]
selected_topic = st.selectbox("Choose a topic to learn about:", topics)

# Add buttons to start conversation or story mode
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Conversation"):
        st.session_state.topic_selected = True
        st.session_state.story_mode = False
        st.session_state.messages = []
        st.session_state.memory.clear()
with col2:
    if st.button("Start Story"):
        st.session_state.topic_selected = True
        st.session_state.story_mode = True
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.session_state.current_part = 0
        st.session_state.user_answers = {}
        st.session_state.story_parts = []
        st.rerun() 

# Load and embed knowledge base
def load_and_embed_knowledge_base(topic: str) -> Tuple[str, np.ndarray]:
    file_path = os.path.join("knowledge_base", f"{topic}.txt")
    with open(file_path, "r") as file:
        content = file.read()
    
    # Split content into chunks
    chunks = content.split('\n\n')
    
    # Embed chunks
    embeddings = st.session_state.embedding_model.encode(chunks)
    
    return content, embeddings

# Retrieve relevant information
def retrieve_relevant_info(query: str, embeddings: np.ndarray, content: str, top_k: int = 3) -> str:
    query_embedding = st.session_state.embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    chunks = content.split('\n\n')
    relevant_chunks = [chunks[i] for i in top_indices]
    
    return '\n\n'.join(relevant_chunks)

# Parse story and blanks
def parse_story(content: str) -> List[Tuple[List[str], str, int]]:
    lines = content.split('\n')
    story_start = lines.index("Story with Blanks:") + 1
    story_lines = lines[story_start:]
    story = ' '.join(story_lines)
    parts = story.split('[')
    story_parts = [parts[0]]
    for part in parts[1:]:
        options, text = part.split(']', 1)
        options = options.split('/')
        correct_index = next((i for i, opt in enumerate(options) if opt.endswith('*')), 0)
        options = [opt.rstrip('*') for opt in options]
        story_parts.append((options, text.replace(" ", " [BLANK] ", 1), correct_index))
    return story_parts

# Display story with blanks
def display_story():
    if not st.session_state.story_parts:
        knowledge_base_content, _ = load_and_embed_knowledge_base(selected_topic)
        st.session_state.story_parts = parse_story(knowledge_base_content)
    
    # Display accumulated story
    accumulated_story = ""
    for i, part in enumerate(st.session_state.story_parts):
        if i < st.session_state.current_part:
            if isinstance(part, str):
                accumulated_story += part
            else:
                options, text, correct_index = part
                correct_answer = st.session_state.user_answers.get(i, options[correct_index])
                accumulated_story += text.replace("[BLANK]", correct_answer)
    
    if accumulated_story:
        st.markdown("### Story so far:")
        st.write(accumulated_story)
    
    if st.session_state.current_part < len(st.session_state.story_parts):
        part = st.session_state.story_parts[st.session_state.current_part]
        if isinstance(part, str):
            st.write(part)
            st.session_state.current_part += 1
            st.rerun()
        else:
            options, text, correct_index = part
            st.markdown("### Fill in the blank:")
            blank_text = text.replace("[BLANK]", "___________")
            st.write(blank_text)
            st.write("Choose the correct option:")
            cols = st.columns(len(options))
            for i, option in enumerate(options):
                if cols[i].button(option, key=f"option_{i}_{st.session_state.current_part}"):
                    if i == correct_index:
                        st.success(f"Correct! The answer is: {option}")
                        sleep(0.5)
                        st.session_state.user_answers[st.session_state.current_part] = option
                        st.session_state.current_part += 1
                        st.rerun()
                    else:
                        st.error(f"Incorrect. Try again!")
    else:
        st.success("You've completed the story!")
        if st.button("Restart Story"):
            st.session_state.current_part = 0
            st.session_state.user_answers = {}
            st.rerun()

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Main app logic
if st.session_state.topic_selected:
    if st.session_state.story_mode:
        display_story()
    else:
        # Conversation mode
        prompt = st.chat_input("Your response:")

        for message in st.session_state.messages:
            avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if selected_topic not in st.session_state.embeddings:
            knowledge_base_content, embeddings = load_and_embed_knowledge_base(selected_topic)
            st.session_state.embeddings[selected_topic] = (knowledge_base_content, embeddings)
        else:
            knowledge_base_content, embeddings = st.session_state.embeddings[selected_topic]

        if not st.session_state.messages:
            try:
                groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model_option)

                relevant_info = retrieve_relevant_info("introduction to " + selected_topic, embeddings, knowledge_base_content)

                system_prompt = f"""You are an English Teacher. The user wants to learn about the topic: {selected_topic}. Their proficiency level is {selected_level}. 
                Use the following relevant information to inform your responses and generate engaging lessons:

                {relevant_info}

                Your role is to engage the user in a conversation about the topic, asking focused, concise questions to encourage their English practice.

                Strict Guidelines:
                1. Start with a brief greeting and a short, engaging question about the topic.
                2. Keep your responses and questions concise, Only one short question.
                3. Ask only one question at a time to maintain focus.
                4. Tailor your language complexity to the user's proficiency level.
                5. Aim to elicit more detailed responses from the user to practice their English skills.
                6. Use the provided information to provide accurate information and examples.

                Your goal is to stimulate conversation and English practice, not to provide lengthy explanations or no context."""

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
                        response = conversation.predict(topic=selected_topic)
                        st.write(response)

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.memory.chat_memory.add_ai_message(response)

            except Exception as e:
                st.error(f"An error occurred: {e}", icon="🚨")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.user_inputs += 1
            
            with st.chat_message("user", avatar='👨‍💻'):
                st.markdown(prompt)

            try:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Thinking..."):
                        relevant_info = retrieve_relevant_info(prompt, embeddings, knowledge_base_content)
                        
                        system_prompt = f"""You are an English Teacher. The user wants to learn about the topic: {selected_topic}. Their proficiency level is {selected_level}. 
                        Use the following relevant information to inform your responses:

                        {relevant_info}

                        Your role is to engage the user in a conversation about the topic, asking focused, concise questions to encourage their English practice.

                        Strict Guidelines:
                        1. Keep your responses and questions concise, Only one short question.
                        2. Ask only one question at a time to maintain focus.
                        3. Tailor your language complexity to the user's proficiency level.
                        4. Aim to elicit more detailed responses from the user to practice their English skills.
                        5. Use the provided information to provide accurate information and examples.

                        Your goal is to stimulate conversation and English practice, not to provide lengthy explanations or no context."""

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
                Advice for Improvement: [Tailored advice here]

                For each example, use the format:
                Mistake: [Original text]
                Correction: [Corrected text]
                Explanation: [Brief explanation]"""

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
                            if example.startswith("Mistake:"):
                                st.markdown(f"<span style='color: red;'>{example}</span>", unsafe_allow_html=True)
                            elif example.startswith("Correction:"):
                                st.markdown(f"<span style='color: green;'>{example}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"- {example}")
                        st.markdown("### Advice for Improvement")
                        st.write(advice.strip())

                except Exception as e:
                    st.error(f"An error occurred while generating feedback: {e}", icon="🚨")

if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2024 EnGuru. All rights reserved.")
