# Import Library
import os
import requests
import pandas as pd
import streamlit as st
import json
from dotenv import load_dotenv
from datetime import datetime

# import tiktoken
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# Load environment variables from .env
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv("MY_OPENAI_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def fetch_data(user):
    if user == 'Movie':
        url = "https://api.themoviedb.org/3/movie/now_playing?language=en-US&page=1"
        info_columns = ['title', 'release_date', 'genre', 'poster_path','popularity']
    elif user == 'TV Show':
        url = "https://api.themoviedb.org/3/tv/airing_today?language=en-US&page=1"
        info_columns = ['name', 'first_air_date', 'genre', 'poster_path','popularity']
    else:
        raise ValueError("User must be 'Movie' or 'TV Show'")

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_KEY}"
    }

    response = requests.get(url, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Extract the list of movies or TV shows from the response
        results = data.get('results', [])
        
        # Create a DataFrame
        df = pd.DataFrame(results)

        # Add a new column 'genre' based on genre_ids
        genre_mapping = {
            28: 'Action',
            12: 'Adventure',
            16: 'Animation',
            35: 'Comedy',
            80: 'Crime',
            99: 'Documentary',
            18: 'Drama',
            10751: 'Family',
            14: 'Fantasy',
            36: 'History',
            27: 'Horror',
            10402: 'Music',
            9648: 'Mystery',
            10749: 'Romance',
            878: 'Science Fiction',
            10770: 'TV Movie',
            53: 'Thriller',
            10752: 'War',
            37: 'Western',
            10759: 'Action & Adventure',
            10762: 'Kids',
            10763: 'News',
            10764: 'Reality',
            10765: 'Sci-Fi & Fantasy',
            10766: 'Soap',
            10767: 'Talk',
            10768: 'War & Politics'  
        }
        
        df['genre'] = df['genre_ids'].apply(lambda x: [genre_mapping.get(genre_id, 'Unknown') for genre_id in x])
        
        # Create combined_info column based on user type
        df['combined_info'] = df.apply(lambda row: f"{info_columns[0]}: {row[info_columns[0]]}, {info_columns[1]}: {row[info_columns[1]]},{info_columns[2]}: {row[info_columns[2]]}, {info_columns[3]}: {row[info_columns[3]]}, {info_columns[4]}: {row[info_columns[4]]}", axis=1)
        
        return df
    
    else:
        print(f"Error fetching data. Status code: {response.status_code}")
        return None

st.header("Movie / TV Show Recommendation Chatbot")
user = st.radio('',['Movie','TV Show'])

if user:
    df = fetch_data(user)
    if df is not None:
        st.dataframe(df)


