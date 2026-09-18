import torch
from keybert import KeyBERT
from ollama import ChatResponse,chat
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel
from sentence_transformers import SentenceTransformer

key_bert = KeyBERT()

chat_model = "granite4.1:3b"
embeddor = SentenceTransformer("multi-qa-distilbert-cos-v1")
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
model = AutoModel.from_pretrained("microsoft/unixcoder-base")

chat(model=chat_model,messages=[],keep_alive=-1)             

def ask_model(hist,schema=None):
    response:ChatResponse = chat(model=chat_model,messages=hist,format=schema)
    
    return response.message.content

def make_embeddings(messages,normalize=True):
    return embeddor.encode(messages,convert_to_numpy=True,normalize_embeddings=normalize)

def make_keywords(data):
    return key_bert.extract_keywords(data,diversity=0.4,stop_words='english')

def embed_code(codes):
    code_tokens = tokenizer(codes,add_special_tokens=False)

    with torch.no_grad():
        outputs = model(**code_tokens)
    
    embeddings = outputs.last_hidden_state[:,0,:]
    embeddings = F.normalize(embeddings,p=2,dim=1)
    
    return embeddings
    