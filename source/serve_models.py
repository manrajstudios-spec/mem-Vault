import fastapi
from fastapi import requests,responses
from fastapi.middleware.cors import CORSMiddleware
from call_model import make_embeddings,make_keywords

app = fastapi.FastAPI()

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_headers=["*"],
                   allow_methods=["*"])

@app.get("/embeddings/{chunks}")
def embeddings(chunks:list):
    return make_embeddings(chunks)

@app.get("/keywords/{chunks}")
def keywords(chunks:list):
    return make_keywords(chunks)
