import time
import spacy
import numpy as np
from call_model import make_embeddings,make_keywords

make_sentences = spacy.load("en_core_web_sm")

def make_chunks(text,limit=800):
    start_time = time.monotonic()

    sents = make_sentences(text)
    sents = [sent.text for sent in sents]
    
    print(f"sent time: {time.monotonic() - start_time}") 
    start_time = time.monotonic()
    
    chunks = []
    cur_chunk = ""
        
    for sent in sents:
        added = cur_chunk + " " + sent
        
        if len(added) > limit:
            if cur_chunk:
                chunks.append(cur_chunk)
                
            if len(sent) > limit:
                chunks.append(sent[:limit])
                cur_chunk = sent[limit:]
                continue
            
            cur_chunk = sent
        elif len(added) == limit:
            chunks.append(added)
            cur_chunk = ""
        else:
            cur_chunk = added
        
    if cur_chunk:
        chunks.append(cur_chunk[:min(len(cur_chunk),limit)])
    
    print(f"Chunk Time: {time.monotonic() - start_time}")
    print(f"chunks: {len(chunks)}")
    
    return chunks

def make_groups(chunks,threshold=0.6,tabels=None): 
    start_time = time.monotonic()
       
    tuple_keywords = make_keywords(chunks)
    
    if isinstance(tuple_keywords[0],tuple):
        tuple_keywords = [tuple_keywords]
    
    if tabels is not None:
        chunks = chunks + tabels
    
    embeddings = make_embeddings(chunks)
    
    if tabels is not None:
        tabel_embeds = embeddings[-len(tabels):]
        tabel_embeds = np.stack(tabel_embeds)
        embeddings = embeddings[:-len(tabels)]

    embeddings = np.stack(embeddings)
    keywords = [{keyword:value for keyword,value in tuplee} for tuplee in tuple_keywords]
    
    print(f"embed and keyword time: {time.monotonic() - start_time}")
    start_time = time.monotonic()
    
    # Grouping Logic 
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims,float("-inf"))
    
    last_groups = []
    
    for i,sim in enumerate(sims):
        sim = np.argwhere(sim>=threshold).flatten()
        sim = sim[sim > i].tolist()
        sim.append(i)
        
        if not last_groups:
            last_groups.append(set(sim))
            continue    
        
        if last_groups:
            sim = set(sim)
            founded = sim.copy()
            groups = []
            
            for last in last_groups:
                if last & founded:
                    founded |= last
                else:
                    groups.append(last)
            
            if not founded:
                groups.append(sim)
            else:
                groups.append(founded)  
                    
            last_groups = groups

    groups = last_groups
    
    print(f"finding group: {time.monotonic() - start_time}")

    print(f"group: {len(groups)}")

    return groups,keywords,embeddings,tabel_embeds if tabels else None

