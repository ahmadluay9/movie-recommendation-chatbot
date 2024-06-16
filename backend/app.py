import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
# from pydantic import BaseModel
from dotenv import load_dotenv
from utils import fetch_data, setup_chain, run_chain

app = FastAPI()

# Load environment variables from .env
load_dotenv()

# Access the OpenAI API key
OPENAI_API_KEY = os.getenv("MY_OPENAI_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

@app.get('/')
def index():
    return {'message': 'Movie Recommendation ChatBot'}

@app.get("/fetch/{user}")
async def fetch(user: str):
    try:
        df = fetch_data(user)
        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend/")
async def recommend(user: str, query: str):
    try:
        df = fetch_data(user)
        chain = setup_chain(OPENAI_API_KEY, df)
        result = run_chain(chain, query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)