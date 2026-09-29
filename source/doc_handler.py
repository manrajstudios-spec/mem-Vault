import cv2
import json
import time
import pickle
import pymupdf
import camelot
import subprocess
import numpy as np
from itertools import repeat
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor
from utils import make_groups,make_chunks,make_sentences
from call_model import make_embeddings,make_keywords,table_model

path = "Data/doc_data/attention.pdf"
parent_temp_pdf_images_path = "Data/temp_pdf_images/"
loaded_docs = []

def read_doc_page(info):
    page=info[1]
    i=info[0]

    text_extracted = page.get_text("text")
    pix = page.get_pixmap()
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    cv_im = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        
    return {"page_num":i,"page_text":text_extracted,"have_table":False,"cv_im":cv_im}

def open_pdf(path):
    return pymupdf.open(path)

def chunk_each_page(sents):
    return make_chunks(auto=False,sents=sents)

def add_doc(path):
    start_time = time.monotonic()
    super_start_time=time.monotonic()

    doc = pymupdf.open(path)
    
    print(f"----------Pdf Opening Time: {time.monotonic() - start_time}----------------------")
    start_time = time.monotonic()
    
    doc_name = path.split("/")[-1].split(".")[0]
    text_per_page = []
    tabel_pages = []
    pool = ThreadPoolExecutor(max_workers=8)
    
    pages_data = list(pool.map(read_doc_page,[(i,page) for i,page in enumerate(doc)]))
    text_per_page = [page["page_text"] for page in pages_data]
    
    print(f"----------Extract Text And Pixmap : {time.monotonic() - start_time}---------------------")
    start_time = time.monotonic()
    
    all_images_cv = [p["cv_im"] for p in pages_data]
    yolo_output = table_model(all_images_cv,batch=16,verbose=False)
    tabel_pages = []
    
    for i,yo in enumerate(yolo_output):
        if len(yo.boxes) > 0:
                tabel_pages.append(i)
    
    print(f"--------------------Detecting Tabel From Pixmap Time: {time.monotonic() - start_time}---------------------")
    start_time = time.monotonic()
        
    if tabel_pages:
        extracted_tables = camelot.read_pdf(path,pages=",".join([str(p+1) for p in tabel_pages]),flavor="stream",parallel=True,cpu_count=4)
    else:
        extracted_tables = []
        
    doc_tables = []
    
    for tabel in extracted_tables:
        tabel_dict = tabel.df.to_dict(orient="records")
        table_str = json.dumps(tabel_dict)
        doc_tables.append(table_str)
    
    text = "\n".join(text_per_page)
    
    print(f"--------------------Extract Tabels Data: {time.monotonic() - start_time}-------------------------------------")
    start_time = time.monotonic()

    chunks = make_chunks(text=text)

    print(f"--------Chunksing Time: {time.monotonic() - start_time}---------------------------")
    start_time = time.monotonic()
    
    old_len = len(chunks)
    
    chunks.extend(doc_tables)
    
    doc_embeddings= make_embeddings(chunks)
    doc_tuple_keywords= make_keywords(chunks)
    
    print(f"-----------------Embedding KeyBert Time: {time.monotonic() - start_time}-----------------------")
    start_time = time.monotonic()

    doc_tabel_embeds = doc_embeddings[old_len:]
    doc_embeddings = doc_embeddings[:old_len]
    doc_tuple_keywords = doc_tuple_keywords[:old_len]
    
    doc_embeddings = np.stack(doc_embeddings)
    
    if isinstance(doc_tuple_keywords,tuple):
        doc_tuple_keywords = [doc_tuple_keywords]
    
    doc_keywords = [{m:c for m,c in tuplee} for tuplee in doc_tuple_keywords]
    
    print(f"---------------------Keyword Setting Time: {time.monotonic() - start_time}---------------------")
    start_time = time.monotonic()
     
    groups,_,_,_ = make_groups(chunks=chunks,threshold=0.6,tabels=doc_tables,auto=False,embeddings=doc_embeddings,keywords=doc_keywords)    
    
    doc_grouped_embeddings = [[doc_embeddings[g] for g in group] for group in groups]
    doc_grouped_keywords_unpacked = [[doc_keywords[g] for g in group] for group in groups]
    
    # Grouping Keywords So We Get One Dict Per Group
    doc_groups_keywords = []
    
    for group_k in doc_grouped_keywords_unpacked:
        to_add = {}
        
        for dictt in group_k:
            for keyword,value in dictt.items():
                to_add[keyword] = to_add.get(keyword,0) + value
        
        doc_groups_keywords.append(to_add)    

    groups_mean = [np.stack(g_e).mean(axis=0) for g_e in doc_grouped_embeddings]
    
    print(f"------------------Grouping Time: {time.monotonic() - start_time}--------------------------")
    
    print(f"chunks: {len(chunks)}, groups: {len(groups)}")
    
    print(f"------------------------------TOTAL TIME TAKEN: {time.monotonic() - super_start_time}---------------------------------")
    
    return {"doc_name":doc_name,"chunks":chunks,"groups_mean":groups_mean,"groups_keywords":doc_groups_keywords,"tables":doc_tables,"table_embeds":doc_tabel_embeds,"groups":groups,"embeddings":doc_embeddings,"keywords":doc_keywords}

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

def get_relevant_data(query_emebddings,query_keywords,doc,select_tables=False):
    start_time = time.monotonic()
    super_start_time = time.monotonic()
    
    doc_groups = doc["groups"]
    doc_chunks = doc["chunks"]
    doc_tables = doc["tables"]
    doc_keywords = doc["keywords"]
    doc_embeddings = doc["embeddings"]
    doc_groups_means = doc["groups_mean"]
    doc_table_emebeds = doc["table_embeds"]
    doc_grouped_keywords = doc["groups_keywords"]
    
    doc_groups_means = np.stack(doc_groups_means)
    
    query_emebddings_sims = query_emebddings @ doc_groups_means.T
    
    print(f"--------------Embedding Sim Time: {time.monotonic() - start_time}-------------------------")
    start_time = time.monotonic()
    
    query_keywords_score = []
    
    for query_dict in query_keywords:
        scores = []
    
        for group_dict in doc_grouped_keywords:
            score = sum(value + group_dict[keyword]for keyword, value in query_dict.items() if keyword in group_dict)
            scores.append(score)

        query_keywords_score.append(scores)
    
    query_keywords_score = np.array(query_keywords_score)
    
    print(f"--------------Keyword Score Time: {time.monotonic() - start_time}-------------------------")
    start_time = time.monotonic()
    
    sims = query_emebddings_sims * 0.6 + 0.4 * np.log1p(query_keywords_score)
    
    selected_groups_ids = []
    
    n = 5
    for sim in sims:
        ids = np.argsort(sim)[-min(n,len(sim)):]
        selected_groups_ids.extend(ids)
    
    selected_groups_ids = set(selected_groups_ids)  
    selected_chunks = []
    selected_embeddings = []
    selected_keywords = []
    
    for cur_selected_group_id in selected_groups_ids:
        cur_group = doc_groups[cur_selected_group_id]

        selected_embeddings.extend([doc_embeddings[cur_id] for cur_id in cur_group])
        selected_chunks.extend([doc_chunks[cur_id] for cur_id in cur_group])
        selected_keywords.extend([doc_keywords[cur_id] for cur_id in cur_group])

    print(f"--------------Selecting Matching Chunks Time: {time.monotonic() - start_time}-------------------------")
    start_time = time.monotonic()

    selected_tables = []
    selected_table_ids = []

    if select_tables:
        table_sims = query_emebddings @ doc_table_emebeds.T
        
        for sim in table_sims:
            selected_ids = np.argwhere(sim>=0.4).flatten()
            selected_table_ids.extend(selected_ids)
        
        selected_table_ids = set(selected_table_ids)

        if doc_tables:
            selected_tables = [doc_tables[i] for i in selected_table_ids]
    
        print(f"--------------Table Comparing And Extracting Time Time: {time.monotonic() - start_time}-------------------------")
        start_time = time.monotonic()
    
    # rerank queries

    reranked_ids = rerank_selected_data(embeddings=selected_embeddings,query_embeddings=query_emebddings,query_keywords=query_keywords,keywords=selected_keywords)
    
    print(f"--------------Re Ranking Time: {time.monotonic() - start_time}-------------------------")
    start_time = time.monotonic()
    
    selected_chunks = [doc_chunks[i] for i in reranked_ids]

    if selected_tables:
        selected_chunks += selected_tables

    print(f"--------------Total Time Taken In Retriving: {time.monotonic() - super_start_time}-------------------------")
    start_time = time.monotonic()

    return selected_chunks

def rerank_selected_data(embeddings,query_embeddings,query_keywords,keywords,top_k=10):
    embeddings = np.stack(embeddings)
    emebedding_sims = query_embeddings @ embeddings.T
    
    key_score=[]
    
    for query_dict in query_keywords:
        scores = []
    
        for e_dict in keywords:
            score = sum(value + e_dict[keyword]for keyword, value in query_dict.items() if keyword in e_dict)
            scores.append(score)

        key_score.append(scores)
    
    key_score = np.array(key_score)
        
    weighted_combined_sims = 0.6 * emebedding_sims + 0.4 * np.log1p(key_score)
    
    selected = []
    
    for sim in weighted_combined_sims:
        selected_ids = sim.argsort()[-min(top_k,len(sim)):]
        selected.extend(selected_ids)
    
    selected = set(selected)
    
    return selected

if __name__ == "__main__":
    load_docs()
    start_time = time.monotonic()
    
    queries = [
    "How were neurons reconstructed and identified in the FlyWire connectome?",
    "What methods did FlyWire use to trace individual neurons and determine their identities and classifications?"]

    query_embeddings = make_embeddings(queries)
    query_keywords_tuple = make_keywords(queries)
    
    if isinstance(query_keywords_tuple,tuple):
        query_keywords_tuple = [query_keywords_tuple]
    
    query_keywords = [{m:c for m,c in tuplee} for tuplee in query_keywords_tuple]

    pool = ThreadPoolExecutor(max_workers=5)
    
    data = list(pool.map(get_relevant_data,repeat(query_embeddings),repeat(query_keywords),loaded_docs,repeat(True)))

    tabel_needed = True
    
    for d in data:
        print(d)
        print()
        print()