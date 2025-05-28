import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

with open("./dataset.pkl","rb")as f:
    dataset = pkl.load(f)

with open("./model.pkl","rb")as f:
    model = pkl.load(f)

def resume_preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]","",text)
    words = text.split()
    word_vec = [model.wv[word] for word in words if word in model.wv]
    res = np.mean(word_vec,axis=0) if word_vec else np.zeros(model.vector_size)
    return res

def get_recomendation(res):
    reccomendation = []
    for idx,emb in enumerate(dataset["embeddings"]):
        reccomendation.append((cosine_similarity([res],[emb])[0][0],int(idx)))
    
    reccomendation.sort(reverse=True,key=lambda x:x[0])
    seen_titles = set()
    results = []
    for it in reccomendation:
        title = dataset.iloc[it[1]]["title"]
        score = it[0]
        if len(results)==3:
            break
        if title not in seen_titles and score>=.75:
            seen_titles.add(title)
            results.append(title)

    return results