import re
import json
import time
import spacy
import pickle
import pymupdf
import camelot
import subprocess
import numpy as np
from utils import make_groups,make_chunks
from call_model import make_embeddings,make_keywords

path = "Data/doc_data/attention.pdf"
loaded_docs = []

def add_doc(path):
    start_time = time.monotonic()
    doc = pymupdf.open(path)
    
    doc_name = path.split("/")[-1].split(".")[0]
    
    text_per_page = {}
    tabel_pages = []
        
    for i,page in enumerate(doc):
        text_per_page[i] = page.get_text("text")
        
        tabel_finder = page.find_tables()
    
        if tabel_finder.tables:
            tabel_pages.append(i)
    
    df_tabels = camelot.read_pdf(path,pages=",".join([str(p+1) for p in tabel_pages]),flavor="stream",parallel=True,cpu_count=4)
    
    tabels = []
    for tabel in df_tabels:
        tabel_dict = tabel.df.to_dict(orient="records")
        table_str = json.dumps(tabel_dict)
        tabels.append(table_str)
    
    if not tabels:
        tabels = None
    
    text = "\n".join(text_per_page.values())
    
    print(f"tabel text time: {time.monotonic() - start_time}")
    start_time = time.monotonic()

    chunks = make_chunks(text=text)
    
    groups,keywords,embeddings,tabel_embeds = make_groups(chunks=chunks,threshold=0.6,tabels=tabels)    
    
    grouped_embeddings = [[embeddings[g] for g in group] for group in groups]
    grouped_chunks = [[chunks[g] for g in group] for group in groups]
    grouped_keywords_unpacked = [[keywords[g] for g in group] for group in groups]
    
    grouped_keywords = []
    
    for group_k in grouped_keywords_unpacked:
        to_add = {}
        
        for dictt in group_k:
            for keyword,value in dictt.items():
                to_add[keyword] = to_add.get(keyword,0) + value
        
        grouped_keywords.append(to_add)    
    
    group_mean = [np.stack(g_e).mean(axis=0) for g_e in grouped_embeddings]
    group_mean = np.stack(group_mean)
    
    print(f"chunks: {len(chunks)}, groups: {len(groups)},shape: {group_mean.shape}")
    
    return {"doc_name":doc_name,"chunks":chunks,"group_means":group_mean,"grouped_keywords":grouped_keywords,"tabels":tabels,"tabel_embeds":tabel_embeds,"groups":groups,"embeddings":embeddings,"keywords":keywords}

def load_docs():
    while True:
        path = subprocess.run(
                ["zenity", "--file-selection"],
                capture_output=True,
                text=True)
        
        path = path.stdout.strip()
        loaded_docs.append(add_doc(path=path))

        while True:
            add_another = input("Would Yu Like To Add Another Doc:  (yes/no): ")
            
            if add_another:
                break
            
        if add_another == "yes":
            continue
        else:
            break
                    
    return loaded_docs

def get_data_doc(queries,table_needed=False):
    start_time = time.monotonic()
    
    query_embeddings = make_embeddings(queries)
    query_embeddings = np.stack(query_embeddings)
    
    query_tuple_keywords = make_keywords(queries)
    
    if isinstance(query_tuple_keywords[0],tuple):
        query_tuple_keywords = [query_tuple_keywords]
    
    query_keywords = [{m:c for m,c in keyword} for keyword in query_tuple_keywords]
    
    print(f"Embed and key time: {time.monotonic() - start_time}")
    start_time = time.monotonic()
    
    retrieved_info = []
        
    for doc in loaded_docs:
        embedding_sim = query_embeddings @ doc["group_means"].T
        
        print(f"Embedding sim time: {time.monotonic()  -start_time}")
        start_time = time.monotonic()
        
        key_score = []

        for query_dict in query_keywords:
            scores = []
        
            for group_dict in doc["grouped_keywords"]:
                score = sum(value + group_dict[keyword]for keyword, value in query_dict.items() if keyword in group_dict)
                scores.append(score)
    
            key_score.append(scores)
        
        key_score = np.array(key_score)
        
        print(f"keyword Score Time: {time.monotonic() - start_time}")
        
        start_time = time.monotonic()
        
        sims = embedding_sim * 0.6 + 0.4 * np.log1p(key_score)
        
        selected_groups = []
        
        n = 5
        for sim in sims:
            ids = np.argsort(sim)[-min(n,len(sim)):]
            selected_groups.extend(ids)
            
        selected_groups = set(selected_groups)  
        selected_chunks = []
        selected_embeddings = []
        selected_keywords = []
        
        chunks = doc["chunks"]
        embs = doc["embeddings"]
        gs = doc['groups']
                
        for selected in selected_groups:
            cur_g = gs[selected]
            
            selected_embeddings.extend([embs[g] for g in cur_g])
            selected_chunks.extend([chunks[g] for g in cur_g])
            selected_keywords.extend([doc["keywords"][g] for g in cur_g])
                  
        if selected_chunks:
            retrieved_info.append({"doc_name":doc["doc_name"],"content":selected_chunks,"embeddings":np.stack(selected_embeddings),"keywords":selected_keywords})
            
        print(f"selecting_time {time.monotonic() - start_time}")
        start_time = time.monotonic()
    
    table_k=4
    
    if table_needed:
        for doc in loaded_docs:
            if doc["tabels"] is not None:
                tabels_sim = query_embeddings @ doc["tabel_embeds"].T
                selected_tabels = []
                
                for sim in tabels_sim:
                    ids = np.argsort(sim)[-min(len(sim),table_k):]
                    selected_tabels.extend(ids)
                
                selected_tabels = set(selected_tabels)
                
                selected_tabels = [t for i,t in enumerate(doc["tabels"]) if i in selected_tabels]
    
                retrieved_info.append({"tabels":selected_tabels})
    else:
        retrieved_info.append({"tabels":None})
    
    final_info = []
        
    for doc in retrieved_info:
        tabel = doc.get("tabels",0)
        
        if tabel:
            final_info.append({"tabels":tabel})
            continue
        
        embeddings = doc["embeddings"]
        selected = rerank(embeddings=embeddings,query_embeddings=query_embeddings,query_keywords=query_keywords,keywords=doc["keywords"])
        final_info.append({"doc_name":doc["doc_name"],"content":[doc["content"][g] for g in selected]})   

    return final_info

def rerank(embeddings,query_embeddings,query_keywords,keywords,top_k=10):
    start_time= time.monotonic()
    e_sims = query_embeddings @ embeddings.T
    
    key_score=[]
    
    for query_dict in query_keywords:
        scores = []
    
        for e_dict in keywords:
            score = sum(value + e_dict[keyword]for keyword, value in query_dict.items() if keyword in e_dict)
            scores.append(score)

        key_score.append(scores)
    
    key_score = np.array(key_score)
        
    sims = 0.6 * e_sims + 0.4 * np.log1p(key_score)
    
    selected = []
    
    for sim in sims:
        selects = sim.argsort()[-min(top_k,len(sim)):]
        selected.extend(selects)
    
    selected = set(selected)
    
    print(f"rerank time: {time.monotonic() - start_time}")
    return selected
            
if __name__ == "__main__":
    load_docs()
    queries = [
    "How were neurons reconstructed and identified in the FlyWire connectome?",
    "What methods did FlyWire use to trace individual neurons and determine their identities and classifications?"
]
    q_embeddings = make_embeddings(queries)
    
    tabel_needed = True
    final_info = get_data_doc(queries=queries,table_needed=tabel_needed)
    
    for d in final_info:
        tabel = d.get("tabels",0)
        if tabel:
            print(tabel)
            continue
        
        for c in d["content"]:
            print(c)
    