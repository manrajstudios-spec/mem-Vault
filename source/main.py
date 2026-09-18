import time
from router import route_msg
from call_model import ask_model
from doc_handler import load_docs
from mem_handler import save_to_mem


def ask_user(to_ask,options=[],empty=False):
    while True:
        user_input = input(to_ask)
        
        if empty and not user_input:
            return ""
        
        if options:
            if user_input in options:
                return user_input    
            
            continue
        
        if user_input:
            return user_input
                
chat_hist = [] 
to_keep = 5
 
def send_chats():
    global chat_hist
    to_send = []
    
    for i in range(0,len(chat_hist),2):
        cur = f"User: {chat_hist[i]["content"]}\nAssistant: {chat_hist[i+1]["content"]}"
        to_send.append(cur)

    save_to_mem(to_send)
    chat_hist = chat_hist[-to_keep * 2:]
    
while True:
    user_input = ask_user("Enter Your Query: ")
    
    if not user_input: 
        continue
    
    if user_input == "q":
        send_chats()
        break
    
    if user_input == "n":
        load_docs()
    
    start_time = time.monotonic()
    
    chat_hist.append({"role":"user","content":user_input})
    web_info,rag_info,doc_info,user_preferences,user_events,user_decisions,user_tasks = route_msg(chat_hist)
    
    print(f"route Time: {time.monotonic() - start_time}")
    start_time = time.monotonic()

    prompt = ""
    
    if web_info:
        prompt += f"This Data Is Received From Internet And Is Somewhat Similar to users query {web_info}"

    if rag_info:
        prompt += f"This Data Is Extracted From Users Old Chats With Assistant And Is Somewhat Similar to users query {rag_info}"

    if doc_info:
        prompt += f"This Data Is Extracted From documents linked by user in this chat With Assistant And Is Somewhat Similar to users query {doc_info}"
        
    if user_events:
        prompt += f"These Are Events Happend with user any point in time {user_events}"
    
    if user_decisions:
        prompt += f"These Are decisions taken by user any point in time {user_decisions}"

    if user_tasks:
        prompt += f"These Are tasks scheduled by user any point in time {user_tasks}"
        
    temp_chat = chat_hist + [{"role":"system","content":prompt},{"role":"user","content":user_input}]
    
    result = ask_model(temp_chat)
    print(f"Assistant: {result}")
    print(f"route Time: {time.monotonic() - start_time}")
    chat_hist.append({"role":"assistant","content":result})
    start_time = time.monotonic()
    
    