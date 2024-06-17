# Import Library
import streamlit as st
from dataclasses import dataclass
from utils import fetch_data, loader, model_embedding, setup_chain, run_chain
import re

# Define the base URL for the images
base_url = "https://image.tmdb.org/t/p/original/"

# Use regular expression to find all jpg paths in the ai_response
pattern = r"Poster Path: (/[a-zA-Z0-9_/.-]+\.jpg)"

# Set the page config
st.set_page_config(
    page_title="Movie / TV Show Recommendation ChatBot",
    page_icon="💬 ",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "#Welcome to the **Movie / TV Show Recommendation ChatBot**! This application, leveraging generative AI and data from **TMDB (the Movie Database)**, helps you discover new movies and TV shows tailored to your preferences."
    }
)

st.sidebar.header("💬 Movie / TV Show Recommendation ChatBot")
st.sidebar.markdown('''
        ## About
        Welcome to the **Movie / TV Show Recommendation ChatBot**! This application, leveraging generative AI and data from **TMDB (the Movie Database)**, helps you discover new movies and TV shows tailored to your preferences.
       
        Documentation:
        - [TMDB](https://developer.themoviedb.org/docs/getting-started)
        - [OpenAI](https://platform.openai.com/docs/models)
                 
        ''')

# Access the OpenAI API key
OPENAI_API_KEY = st.sidebar.text_input("Enter OpenAI API key:",type="password")
TMDB_API_KEY = st.sidebar.text_input("Enter TMDB API key:",type="password")

user_option = st.sidebar.radio('Seeking a new movie or tv show to watch?',['Movie','TV Show'])

if user_option=='TV Show':
    user_option = 'tv'
    st.header("TV Show Recommendation ChatBot")
    
else:
    user_option = 'movie'
    st.header("Movie Recommendation ChatBot")

st.write('---')

df = fetch_data(user_option)
loader(df)
model_embedding()
chain = setup_chain(df)

# set initial message
if "messages" not in st.session_state.keys():
    st.session_state["messages"] = [
        {"role":"assistant","content":"Hello, how can i help you today?"}]

# Display messages
if "messages" in st.session_state.keys(): 
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

user_prompt = st.chat_input("Enter a prompt here")

if user_prompt is not None:
    st.session_state.messages.append({"role":"user",
                                      "content":user_prompt})
    
    with st.chat_message("user"):
        st.write(user_prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Please wait..."):
            ai_response = run_chain(chain,user_prompt)
            # Display each found image
            matches = re.findall(pattern, ai_response)
            for match in matches:
                st.image(base_url + match,width=450)
            matches = re.findall(pattern, ai_response)
            st.write(ai_response)
    
    new_ai_message = {"role":"assistant","content":ai_response}
    st.session_state.messages.append(new_ai_message)

if st.sidebar.button("Clear Chat"):
    st.session_state["messages"] = [
        {"role":"assistant","content":"Hello, how can i help you today?"}]

st.sidebar.markdown(''' 
## Created by: 
Ahmad Luay Adnani - [GitHub](https://github.com/ahmadluay9) 
''')

