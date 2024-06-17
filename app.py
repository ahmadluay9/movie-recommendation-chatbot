# Import Library
import os
import streamlit as st
from dataclasses import dataclass
from utils import fetch_data, loader, model_embedding, setup_chain, run_chain
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Access the OpenAI API key
OPENAI_API_KEY = os.getenv("MY_OPENAI_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Set the page config
st.set_page_config(layout="wide")

st.sidebar.header("💬 Movie / TV Show Recommendation ChatBot")
st.sidebar.markdown('''
        ## About
        Welcome to the **Movie / TV Show Recommendation ChatBot**! This application helps you discover new movies and TV shows tailored to your preferences.
        
        By leveraging **Generative AI** and data from **TMDB (The Movie Database)**, the chatbot provides personalized recommendations based on your input queries.
        
        Documentation:
        - [Streamlit](https://streamlit.io/)
        - [TMDB](https://developer.themoviedb.org/docs/getting-started)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models)
                 
        ''')

user_option = st.sidebar.radio('Movie or TV Show ?',['Movie','TV Show'])

st.image('tmdb.jpeg')

if user_option=='TV Show':
    user_option = 'tv'
else:
    user_option = 'movie'

df = fetch_data(user_option)
loader(df)
model_embedding()
chain = setup_chain(OPENAI_API_KEY,df)

# Constants for chat actors
USER = "user"
ASSISTANT = "assistant"

@dataclass
class Message:
    actor: str
    payload: str

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]

# Clear chat history when the button is pressed
if st.sidebar.button("Clear Chat"):
    st.session_state["messages"] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]

# initialize_session_state()

# Display chat history
msg: Message
for msg in st.session_state["messages"]:
    st.chat_message(msg.actor).write(msg.payload)

# Prompt
query: str = st.chat_input("Enter a prompt here")
if query:
    st.session_state["messages"].append(Message(actor=USER, payload=str(query)))
    st.chat_message(USER).write(query)

    with st.spinner("Please wait..."):
        response = run_chain(chain,query)
        st.session_state["messages"].append(Message(actor=ASSISTANT, payload=response))
        st.chat_message(ASSISTANT).write(response) 

st.sidebar.markdown(''' 
## Created by: 
Ahmad Luay Adnani - [GitHub](https://github.com/ahmadluay9) 
''')
        # response = requests.get(f"{BASE_URL}/fetch/{user_option}")
#         # response.raise_for_status()  # Raise an HTTPError on bad status
#         # data = response.json()
#         st.write("Fetched Data")
#         st.dataframe(df)
#     except requests.RequestException as e:
#         st.error(f"Error fetching data: {e}")
# else:
#     try:
#         df = fetch_data(user_option)
#         # response.raise_for_status()  # Raise an HTTPError on bad status
#         # data = response.json()
#         st.write("Fetched Data")
#         st.dataframe(df)
#     except requests.RequestException as e:
#         st.error(f"Error fetching data: {e}")


# if user:  # Check if user choice is not empty
#     try:
#         df = fetch_data(user)
#         if df is not None:
#             # Vector DB
#             vectorstore  = FAISS.from_documents(loader(df), model_embedding())
#             conversation_chain()


#         st.dataframe(df)

#     except ValueError as e:
#         st.write(e)

# Constants for chat actors
# USER = "user"
# ASSISTANT = "assistant"

# setup_chain()

# @dataclass
# class Message:
#     actor: str
#     payload: str

# # Initialize session state
# def initialize_session_state():
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]

# initialize_session_state()

# # Display chat history
# msg: Message
# for msg in st.session_state["messages"]:
#     st.chat_message(msg.actor).write(msg.payload)

# # Prompt
# query: str = st.chat_input("Enter a prompt here")
# if query:
#     st.session_state["messages"].append(Message(actor=USER, payload=str(query)))
#     st.chat_message(USER).write(query)

#     with st.spinner("Please wait..."):
#         try:
#             payload = {"user": f"{user_option}", "query": query}  
#             response = requests.post(f"{BASE_URL}/recommend/", json=payload)
#             response.raise_for_status()  # Raise an HTTPError on bad status
#             result = response.json()
#             response_text = result['result']
#         except requests.RequestException as e:
#             response_text = f"Error getting recommendation: {e}"

#         st.session_state["messages"].append(Message(actor=ASSISTANT, payload=response_text))
#         st.chat_message(ASSISTANT).write(response_text) 
    



