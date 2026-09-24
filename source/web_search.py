import time
import requests
import numpy as np
import trafilatura
from ddgs import DDGS
from utils import make_chunks,make_groups
from call_model import make_embeddings,make_keywords

def get_page_text(url):
    try:
        response = requests.get(url,timeout=10,headers={"User-Agent": "Mozilla/5.0"})

        return trafilatura.extract(response.text)

    except requests.exceptions.Timeout:
        print(f"Timeout: {url}")
        return None

def search(queries):
    main_time = time.monotonic()
    start_time=time.monotonic()
    embeddings = make_embeddings(queries)
    tuple_keywords = make_keywords(queries)
    
    if isinstance(tuple_keywords[0],tuple):
        tuple_keywords = [tuple_keywords]
    
    keywords = [{m:c for m,c in tuplee} for tuplee in tuple_keywords]
    print(f"Keyword embedding time: {time.monotonic() - start_time}")
    start_time = time.monotonic()
    
    retrieved_info = []
    
    with DDGS() as ddgs:
        for query in queries:
            results = ddgs.text(query,max_results=2)
            print(f"DDGS extract time time: {time.monotonic() - start_time}")
            start_time = time.monotonic()
            
            for result in results:
                text = get_page_text(result["href"])

                print(f"page extract time: {time.monotonic() - start_time}")
                start_time = time.monotonic()
    
                if text is None:
                    continue
                
                chunks = make_chunks(text)

                print(f"Chunking time: {time.monotonic()-start_time}")
                start_time = time.monotonic()

                groups, web_keywords, web_embeddings, _ = make_groups(chunks)
                grouped_embeddings = [[web_embeddings[g] for g in group] for group in groups]
                grouped_keywords_unpacked = [[web_keywords[g] for g in group] for group in groups]
                                
                web_grouped_keywords = []
                                    
                for group_k in grouped_keywords_unpacked:
                    to_add = {}
                    
                    for dictt in group_k:
                        for keyword,value in dictt.items():
                            to_add[keyword] = to_add.get(keyword,0) + value
                    
                    web_grouped_keywords.append(to_add)    
                
                print(f"Grouping Time: {time.monotonic() - start_time}")
                start_time = time.monotonic()

                web_group_mean = [np.stack(g_e).mean(axis=0) for g_e in grouped_embeddings]
                web_group_mean = np.stack(web_group_mean)
                embedding_sim = embeddings @ web_group_mean.T

                print(f"Embedding sim time: {time.monotonic() - start_time}")
                start_time = time.monotonic()
                
                key_score = []

                for query_dict in keywords:
                    scores = []
                
                    for group_dict in web_grouped_keywords:
                        score = sum(value + group_dict[keyword] for keyword, value in query_dict.items()if keyword in group_dict)
                        scores.append(score)
                
                    key_score.append(scores)
                
                key_score = np.array(key_score)
                
                print(f"keyword Score Time: {time.monotonic() - start_time}")
                start_time = time.monotonic()
                
                sims = embedding_sim * 0.6 + 0.4 * np.log1p(key_score)
                print(f"Sims Shpe: {sims.shape}")
                selected_groups = set()
                k = 4
                
                for sim in sims:
                    ids = np.argsort(sim)[-min(k,len(sim)):]
                    
                    for i in ids:
                        selected_groups.add(i)
                
                print(f"Selectedx dGroups: {selected_groups}")
                selected_chunks = [g for i,g in enumerate(chunks) if i in selected_groups]
                
                if selected_chunks:
                    retrieved_info.append({"query":query,"page_title":result["title"],"content":selected_chunks})

                print(f"Comparision Time {time.monotonic() - start_time}")
                start_time = time.monotonic()
    
    print(f"Full TIme: {time.monotonic() - main_time}")
    return retrieved_info

if __name__ == "__main__":
    print(search(["GPT 6 Astra ","GPT 6 News"]))