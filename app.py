# Import Library
import requests
import streamlit as st
import json
from dotenv import load_dotenv
from datetime import datetime

@st.cache_resource

st.set_page_config(layout="wide")
st.header("💬 Movie / TV Show Recommendation ChatBot")
user = st.radio('',['Movie','TV Show'])

if user:  # Check if user choice is not empty
    try:
        df = fetch_data(user)
        if df is not None:
            # Vector DB
            vectorstore  = FAISS.from_documents(loader(df), model_embedding())
            conversation_chain()


        st.dataframe(df)

    except ValueError as e:
        st.write(e)


