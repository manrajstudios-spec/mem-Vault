import time
import asyncio
import requests
import numpy as np
import trafilatura
from ddgs import DDGS
from utils import make_chunks,make_groups,make_sentences
from call_model import make_embeddings,make_keywords
from concurrent.futures import ThreadPoolExecutor

def get_relevant_data_from_page(data):
    start_time = time.monotonic()
    
    query_keyword,query_embedding = data["query_data"]["keywords"],data["query_data"]["embedding"]

    web_embeddings,web_keywords = data["embeddings"],data["keywords"]
    
    groups, _, _, _ = make_groups(embeddings=web_embeddings,keywords=web_keywords,auto=False)
    grouped_embeddings = [[web_embeddings[g] for g in group] for group in groups]
    grouped_keywords_unpacked = [[web_keywords[g] for g in group] for group in groups]
                    
    web_grouped_keywords = []
                        
    for group_k in grouped_keywords_unpacked:
        to_add = {}
        
        for dictt in group_k:
            for keyword,value in dictt.items():
                to_add[keyword] = to_add.get(keyword,0) + value
        
        web_grouped_keywords.append(to_add)    
    
    print(f"----------------WEB CHUNKS Grouping Time: {time.monotonic() - start_time}-----------------------")
    start_time = time.monotonic()

    web_group_mean = [np.stack(g_e).mean(axis=0) for g_e in grouped_embeddings]
    web_group_mean = np.stack(web_group_mean)
    embedding_sim = (web_group_mean @ query_embedding.T).flatten()
    
    print(f"----------------WEB CHUNKS Embedding sim time: {time.monotonic() - start_time}-----------------")
    start_time = time.monotonic()
    
    key_scores = []

    for group_dict in web_grouped_keywords:
        score = sum(value + group_dict[q_keyword] for q_keyword, value in query_keyword.items() if q_keyword in group_dict)
        key_scores.append(score)
    
    key_scores = np.array(key_scores)
    
    print(f"WEB CHUNKS keyword Score Time: {time.monotonic() - start_time}")
    start_time = time.monotonic()
    
    sims = embedding_sim * 0.6 + 0.4 * np.log1p(key_scores)
    sims = sims.flatten()
    selected_chunks = set()
    k = 4
    s_ids = np.argsort(sims)[-min(k,sims.shape[0]):]
    
    selected_chunks = []

    for g in s_ids:
        cur = groups[g]
        selected_chunks.extend([data["chunks"][i] for i in cur])    
    
    print(f"-------------------FINAL COMPARISION TIME: {time.monotonic() - start_time}-----------------")
    print("----------One Query Comparision Done-------------------------\n")

    return selected_chunks

def get_page_text(data):
    start_time = time.monotonic()
    session = data[0]
    url = data[1]["url"]

    try:
        response = session.get(url,timeout=10,headers={"User-Agent": "Mozilla/5.0"})
        extracted_text = trafilatura.extract(response.text)
        data[1]["extracted_text"] = extracted_text

        print(f"-------------------One Page Extract Time: {time.monotonic() - start_time} ---------------------------")
        return data[1] 
        
    except requests.exceptions.Timeout:
        print(f"Timeout: {data[1]["query"]}")
        return {"query_data":data[1],"url":url,"extracted_text":""}

def get_urls(query_data):
    start_time = time.monotonic()
        
    ddgs_data = DDGS().text(query_data["query"],max_results=2,backend="lite")
    
    print(f"------------FINDING URL SINGLE TIME: {time.monotonic() - start_time}------------------")
    
    return [{"query_data":query_data,"url":d["href"]} for d in ddgs_data]

def chunking_handler(sents):
    return make_chunks(sents=sents,auto=False)


def web_search(queries=[]):
    number_of_queries = len(queries)
    search_started_time = time.monotonic()
    start_time = time.monotonic()
    
    query_data = [{"query":q,"keywords":[],"embedding":[]} for q in queries]
         
    session = requests.Session()
    pool = ThreadPoolExecutor(max_workers=8)
    
    ddgs_results = list(pool.map(get_urls,query_data))

    print(f"\n----------------GETIING URL TOTAL TIME: {time.monotonic() - start_time}------------------\n")
    start_time = time.monotonic()

    trifurata_input_temp = []
    
    for dr in ddgs_results:
        if isinstance(dr,list):
            trifurata_input_temp.extend(dr)
        else:
            trifurata_input_temp.append(dr)

    trifurata_input = [(session,dr) for dr in trifurata_input_temp]
    trifulrata_results = list(pool.map(get_page_text,trifurata_input))
        
    print(f"\n-----------Page Extract Total Time: {time.monotonic() - start_time}-----------\n") 
    start_time = time.monotonic()
    
    # sents
    corpuses = [t["extracted_text"] for t in trifulrata_results]
    
    for i,corpus_sents in enumerate(make_sentences.pipe(corpuses,batch_size=30)):
        trifulrata_results[i]["sents"] = [corpus_sent.text for corpus_sent in corpus_sents]

    print(f"\n-------------------Sent Time ALl Web: {time.monotonic() - start_time}--------------\n")
    start_time = time.monotonic()
    
    # chunks
    chunks = list(pool.map(chunking_handler,[t["sents"] for t in trifulrata_results]))
    
    for i in range(len(trifulrata_results)):
        trifulrata_results[i]["chunks"] = chunks[i]

    print(f"--------------CHUNK Time ALl Web: {time.monotonic() - start_time}-----------------\n")
    start_time = time.monotonic()

    # embeddings
    temp_chunks = [chunks[i][j] for i in range(len(chunks)) for j in range(len(chunks[i]))]
    temp_chunks = queries + temp_chunks
    
    web_future = pool.submit(make_embeddings,temp_chunks)
    keyword_future = pool.submit(make_keywords,temp_chunks)

    web_all_embeddings_flattend = web_future.result()
    print(f"------------------Embedding Time: {time.monotonic() - start_time}---------------\n")
    start_time = time.monotonic()

    tuple_keywords = keyword_future.result()
    print(f"---------------Keyword Time ALL Extract: {time.monotonic() - start_time}---------------\n")
    start_time = time.monotonic()
    
    # keywords
    
    if isinstance(tuple_keywords,tuple):
        tuple_keywords = [tuple_keywords]
    
    web_all_keywords_flattend = []
    
    for tuple_keyword in tuple_keywords:
        cur = {k:s for k,s in tuple_keyword}
        web_all_keywords_flattend.append(cur)

    
    query_embeddings = web_all_embeddings_flattend[:number_of_queries]    
    query_keywords = web_all_keywords_flattend[:number_of_queries]

    for query,query_keyword,query_embedding in zip(queries,query_keywords,query_embeddings):
        for tf in trifulrata_results:
            if tf["query_data"]["query"] == query:
                tf["query_data"]["keywords"] = query_keyword
                tf["query_data"]["embedding"] = query_embedding
                continue
            
    web_all_keywords_flattend = web_all_keywords_flattend[number_of_queries:]
    web_all_embeddings_flattend = web_all_embeddings_flattend[number_of_queries:]

    cur_pos = 0
    web_all_embeddings = []
    web_all_keywords = []
        
    for i in range(len(chunks)):
        web_all_embeddings.append(web_all_embeddings_flattend[cur_pos: cur_pos + len(chunks[i])])
        web_all_keywords.append(web_all_keywords_flattend[cur_pos:cur_pos+len(chunks[i])])
        cur_pos += len(chunks[i])
    
    for i in range(len(trifulrata_results)):
        trifulrata_results[i]["embeddings"] = np.stack(web_all_embeddings[i])
        trifulrata_results[i]["keywords"] = web_all_keywords[i]
    
    print(f"----------------WEB KEYWORD TIME MODYFYING: {time.monotonic() - start_time}---------------------\n")
    start_time = time.monotonic()
    
    selected_data = list(pool.map(get_relevant_data_from_page,trifulrata_results))
    print(f"----------Matching Groups Time: {time.monotonic() - start_time}-----------\n")
    
    print(f"-----------------FULL TIME: {time.monotonic() - search_started_time}----------------\n")

if __name__ == "__main__":
    queries = ["GPT 6 Astra ","Claude Fable 5"]
    web_search(queries)