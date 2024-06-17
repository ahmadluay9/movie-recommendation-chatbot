import os
import requests
import pandas as pd
import tiktoken

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

def fetch_data(user):
    if user == 'movie':
        url = "https://api.themoviedb.org/3/movie/now_playing?language=en-US&page=1"
        info_columns = ['title', 'release_date', 'genre', 'poster_path','popularity','overview']
    elif user == 'tv':
        url = "https://api.themoviedb.org/3/tv/airing_today?language=en-US&page=1"
        info_columns = ['name', 'first_air_date', 'genre', 'poster_path','popularity','overview']
    else:
        raise ValueError("User must be 'movie' or 'tv'")

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
        df['combined_info'] = df.apply(lambda row: f"{info_columns[0]}: {row[info_columns[0]]}, {info_columns[1]}: {row[info_columns[1]]}, {info_columns[2]}: {row[info_columns[2]]}, {info_columns[3]}: {row[info_columns[3]]}, {info_columns[4]}: {row[info_columns[4]]}, {info_columns[5]}: {row[info_columns[5]]}", axis=1)
        
        return df
    
    else:
        print(f"Error fetching data. Status code: {response.status_code}")
        return None

# Load Processed Dataset and textsplitting
def loader(df):
    loader = DataFrameLoader(df, page_content_column="combined_info")
    docs  = loader.load()
    # Document splitting
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    return texts

# embeddings model
def model_embedding():
    # Define the path to the pre-trained model you want to use
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
)
    return embeddings

def setup_chain(df):
    # Vector DB
    openai_api_key = OPENAI_API_KEY
    vectorstore  = FAISS.from_documents(loader(df), model_embedding())
    llm = ChatOpenAI(openai_api_key=openai_api_key, 
                    model_name="gpt-3.5-turbo",
                    temperature=0)
    MOVIE_TVSHOW_QUESTION_CREATOR_TEMPLATE = """
    You are a movie recommender system. 
    From the following context and chat history about movies or TV shows, help users to find movie or tv shows that match their preferences.
    For example, if the user requests a Movie/ TV show recommendation, recommend only 1 newly released or aired movie / tv shows with the following format: 
    Here are our recommendation:
    1.  - Title:
        - Release Date / First Aired:
        - Genre:
        - Popularity:
        - Overview:
        - Poster Path:

    You shouldn't change the language of the question, just reformulate it. If it is not needed to reformulate the question or it is not a question, just output the same text.
       
    {context}

    {chat_history}

    Last Message: {question}
    Your Response: """

    prompt = PromptTemplate.from_template(MOVIE_TVSHOW_QUESTION_CREATOR_TEMPLATE)
    # Set up the memory
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
    qa_chain =  RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": memory},
                verbose=True,
    )
    return qa_chain

    # return response.get("answer")
    # # return chain

def run_chain(chain, query):
    result = chain.run(query=query)
    return result

# qa_chain =  ConversationalRetrievalChain.from_llm(
#             llm=llm,
#             memory = memory,
#             condense_question_llm=llm,
#             condense_question_prompt = prompt_template,
#             retriever=vectorstore.as_retriever(
#                 search_kwargs={"fetch_k":4,"k":3},search_type="mmr"
#             ),
#             chain_type="refine",
#             verbose=True
    # )