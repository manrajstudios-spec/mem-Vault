from ddgs import DDGS


ddgs = DDGS()

print(len(ddgs.text("GPT 6",max_results=1)))
print(type(ddgs.text("GPT 6",max_results=2)))