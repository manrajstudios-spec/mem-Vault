import numpy as np
from call_model import make_embeddings


queries = [
    "How does transformer self-attention work?",
    "Latest developments in open source AI models",
]

extra_sentences = [
    # Matching
    "Transformers use self-attention to understand relationships between tokens.",
    "Attention layers allow neural networks to focus on important parts of the input.",
    "Open-source AI models can be fine-tuned for specific natural language tasks.",
    "Large language models are commonly built using transformer architectures.",
    "Modern AI systems often combine pretrained models with retrieval techniques.",

    # Not matching
    "The Pacific Ocean is the largest ocean on Earth.",
    "Coffee beans are roasted at different temperatures to produce different flavors.",
    "Mount Everest is located in the Himalayan mountain range.",
    "Electric cars use batteries to store energy for their motors.",
    "The human heart pumps blood throughout the circulatory system.",
]

embeds = make_embeddings(queries + extra_sentences)

embeds = np.stack(embeds)

query_embeds = embeds[:len(queries)]

sent_embeds = embeds[len(queries):]


sims = query_embeds @ sent_embeds.T

for sim in sims:
    print(sim)
    sim_ids=  np.argwhere(sim>=0.2)
    print(sim_ids)
    print()