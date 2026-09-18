import json
from web_search import search
from call_model import ask_model
from doc_handler import get_data_doc,loaded_docs
from mem_handler import retrieve_info

with open("Data/prompts_schema/router_prompt.txt",'r') as file:
    sys_prompt = file.read()

with open("Data/prompts_schema/router_output_schema.json",'r') as file:
    json_schema = json.load(file)

with open("Data/user_info_data.json","r") as file:
    user_info_data = json.load(file)

if loaded_docs:
    for doc in loaded_docs:
        user_info_data["docs"] += doc["doc_name"]

prompt = [{"role":"system","content":sys_prompt},{"role":"system","content":f"{user_info_data}"}]

def route_msg(hist):
    json_out = ask_model(prompt+hist,schema=json_schema)
    parsed = json.loads(json_out)
    
    web_queries = parsed["websearch"]
    doc_queries = parsed["doc_retrieval"]
    rag_queries = parsed["rag_retrieval"]
    user_preferences = parsed["user_preferences"]
    user_events = parsed["user_events"]
    user_decisions = parsed["user_decisions"]
    user_tasks = parsed["user_tasks"]
    
    
    if web_queries:
        web_info = search(web_queries)
    
    if doc_queries:
        doc_info = get_data_doc(doc_queries)
        
    if rag_queries:
        rag_info = retrieve_info(rag_queries)
    
    with open("Data/user_info_data.json","w") as file:
        json.dump({"user_events":user_events,"user_prefrences":user_preferences,"user_tasks":user_tasks,"user_decisions":user_decisions,"docs":[]},file)
    
    return web_info if web_queries else [],rag_info if rag_queries else [] ,doc_info if doc_queries else [],user_preferences,user_events,user_decisions,user_tasks
    

if __name__ =="__main__":
    test_prompts = [
    "What's the weather in Ludhiana right now?",
    "My cousin just got into IIT Bombay, super happy for him.",
    "I've decided to switch my final year project from the RL agent to the RAG chatbot."
]
    for msg in test_prompts:
        route_msg([{"role":"user","content":msg}])