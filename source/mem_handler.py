import json
import time
import pickle
import numpy as np
import pandas as pd
from classes import Group
from call_model import make_embeddings
from sentence_transformers import util
from call_model import make_embeddings,ask_model,make_keywords

test_exchanges = []

with open("Data/rag_data/test_exchanges.json",'r') as file:
    test_exchanges = json.load(file)["exchanges"]
    
def get_old_data():
    try:
        stored_embeddings = np.load("Data/rag_data/all_embeddings.npy")
    except:
        stored_embeddings = None
        
    try:
        stored_mean = np.load("Data/rag_data/all_means.npy")
    except:
        stored_mean = None
    
    try:
        with open("Data/rag_data/exchanges_keywords.pickle",'rb') as f:
            stored_data = pickle.load(f)
            return [Group(group_id=g_id,members=members) for g_id,members in stored_data["groups"]],stored_embeddings,stored_data["keywords"],stored_data["exchanges"],stored_mean
    except:
        return [],stored_embeddings,[],[],stored_mean
        
def save_data(groups,embeddings,exchanges,mean,keywords):
    with open("Data/rag_data/exchanges_keywords.pickle","wb") as f:
        pickle.dump({
            "groups":[(group.group_id,group.members) for group in groups],
            "keywords":keywords,
            "exchanges":exchanges
        },file=f,protocol=pickle.HIGHEST_PROTOCOL)
        
    with open("Data/rag_data/json_keys.json",'w') as f:
        json.dump({"ids":[(group.group_id,group.members) for group in groups],"exchanges":exchanges,"keywords":keywords},f,indent=4)
                            
    np.save("Data/rag_data/all_embeddings",embeddings)
    np.save("Data/rag_data/all_means",mean)

def make_groups(exchanges):
    embeddings = make_embeddings(exchanges,normalize=True)
    
    embedding_sim = embeddings @ embeddings.T    
    tuple_keywords = make_keywords(exchanges)
    
    keywords = [dict(exchange_keywords) for exchange_keywords in tuple_keywords]    

    n = len(exchanges)
    keywords_score = np.zeros((n,n),dtype=float)

    for i,outer in enumerate(keywords):
        for j in range(i+1,n):
            inner = keywords[j]
            score = sum((value+inner[keyword]) for keyword,value in outer.items() if keyword in inner)
            keywords_score[i,j] = score
            keywords_score[j,i] = score
    
    print(f"embedding matrix: {embedding_sim}")
    
    sims = 0.65 * embedding_sim + 0.35 * np.log1p(keywords_score)
    
    print(f"final matice: {sims}")
        
    np.fill_diagonal(sims,float("-inf"))
    
    threshold = 0.4
    groups = []
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
    
    # make 1 dict per group 
    grouped_keywords_unpacked = [[keywords[g] for g in group] for group in groups]
    grouped_keywords = []
    
    for group_k in grouped_keywords_unpacked:
        to_add = {}
        
        for dictt in group_k:
            for keyword,value in dictt.items():
                to_add[keyword] = to_add.get(keyword,0) + value
        
        grouped_keywords.append(to_add)
    
    grouped_exchanges = [[exchanges[g] for g in group] for group in groups]
    grouped_embeddings = [np.vstack([embeddings[i] for i in group]) for group in groups]
    grouped_mean = [group_e.mean(axis=0) for group_e in grouped_embeddings]    
    
    return groups,grouped_exchanges,grouped_embeddings,grouped_mean,grouped_keywords

def save_to_mem(exchanges):
    groups,grouped_exchanges,grouped_embeddings,grouped_mean,grouped_keywords = make_groups(exchanges)
    print(groups)
    
    stored_groups,stored_embeddings,stored_keywords,stored_chats,stored_mean = get_old_data()
    
    threshold = 0.56
    
    groups_to_add = []
    
    for group,group_mean,group_embedidngs,group_keywords,group_exchanges in zip(groups,grouped_mean,grouped_embeddings,grouped_keywords,grouped_exchanges):
        selected_groups = []
        
        for old_group in stored_groups:
            old_id = old_group.group_id
            old_mean = stored_mean[old_id]
            old_keywords = stored_keywords[old_id]
            mean_sim = group_mean @ old_mean
            
            keyword_sim = sum(value + old_keywords[keyword]  for keyword,value in group_keywords.items() if keyword in old_keywords)
            sim = mean_sim * 0.6 + np.log1p(keyword_sim) * 0.4
            
            if sim >= threshold:
                selected_groups.append(old_group)
        
        old_len = len(stored_chats)
        
        if stored_embeddings is None:
            stored_embeddings = group_embedidngs.reshape(1,-1)
        else:
            stored_embeddings = np.concatenate((stored_embeddings,group_embedidngs))
      
        stored_chats.extend(group_exchanges)
        
        new_members = list(range(old_len, old_len + len(group_exchanges)))
        
        if not selected_groups:
            new_group_id = len(stored_groups) + len(groups_to_add)
            stored_keywords.append(group_keywords)
            
            if stored_mean is None:
                stored_mean = group_mean.reshape(1,-1)
            else:
                stored_mean = np.concatenate((stored_mean,group_mean.reshape(1,-1)))
    
            new_group = Group(group_id=new_group_id,members=new_members)
            groups_to_add.append(new_group)
            continue
        
        for group in selected_groups:
            group.members.extend(new_members)
            
            idd = group.group_id
            mean = stored_mean[idd]
            stored_mean[idd] = (mean * mean.size + group_mean * group_mean.size)/ (mean.size + group_mean.size) 
            
            for keyword,value in group_keywords.items():
                stored_keywords[old_id][keyword] = stored_keywords[old_id].get(keyword,0) + value
            
    stored_groups.extend(groups_to_add)
    
    save_data(groups=stored_groups,embeddings=stored_embeddings,exchanges=stored_chats,mean=stored_mean,keywords=stored_keywords)

def retrieve_major_groups(queries,stored_groups,stored_keywords,stored_mean):
    embeddings = make_embeddings(queries,True)
    embeddings = np.stack(embeddings)
    
    tuple_keywords = make_keywords(queries)
    if isinstance(tuple_keywords[0],tuple):
        tuple_keywords = [tuple_keywords]
    keywords = [dict(exchange_keywords) for exchange_keywords in tuple_keywords]     

    selected_groups = set()
    
    threshold = 0.35

    embeddings_sims = embeddings @ stored_mean.T
    
    keywords_scores = []
    
    for new_keyword in keywords:
        score = [sum(value + old_keyword_dict[keyword] for keyword,value in new_keyword.items() if keyword in old_keyword_dict) for old_keyword_dict in stored_keywords]    
        keywords_scores.append(score)
    
    keywords_scores = np.array(keywords_scores)
    
    final_sims = embeddings_sims * 0.5 + np.log1p(keywords_scores) * 0.5
    
    for sim in final_sims:
        ids = np.argwhere(sim>=threshold).flatten()
        
        for i in ids:
            g = stored_groups[i]
            selected_groups.add(g)

    print([g.group_id for g in selected_groups])
    return selected_groups

def rerank(groups,queries):
    retrived_info = []
    stored_groups,stored_embeddings,stored_keywords,stored_chats,stored_mean = get_old_data()
    
    query = "\n".join(queries)
    
    embedding = make_embeddings(query)
    
    threshold = 0.35
    
    for group in groups:
        embedidngs = stored_embeddings[group.members]
        exchanges = [stored_chats[i] for i in group.members]
        
        sim = (embedding.reshape(1,-1) @ embedidngs.T).flatten()
        print(sim)
        selected_ids = np.argwhere(sim>=threshold).flatten()

        selected_exchanges = [exchanges[i] for i in selected_ids]
        
        retrived_info.append(selected_exchanges)
    
    return retrived_info

def retrieve_info(queries):
    stored_groups,stored_embeddings,stored_keywords,stored_chats,stored_mean = get_old_data()
    
    getting_groups_time = time.monotonic()
    selected_groups = retrieve_major_groups(queries=queries,stored_groups=stored_groups,stored_keywords=stored_keywords,stored_mean=stored_mean)
    print(time.monotonic() - getting_groups_time)
    
    shorlist_time = time.monotonic()
    info = rerank(selected_groups,queries)
    print(time.monotonic() - shorlist_time)
    
    return info