import pandas as pd
import numpy as np
try:
    import faiss
except Exception as e:
    print("Could not load faiss!!",e)
import os

import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='True'
index = None
model = None
data = None
base_path = os.path.abspath("../../")
device = None

def initialize_index(path_to_roc_vecs=base_path+"/localdatasets/rocstories/roc_vecs.hdf", model_type='all-mpnet-base-v2',d = 768, dev=None, howto=False):
    global model,index ,data,story_data, device
    print("Using and initializing the howto dataset?",howto)
    if not howto:
        path_to_roc_vecs = base_path + "/localdatasets/rocstories/roc_vecs.hdf"
    else:
        path_to_roc_vecs = base_path + "/localdatasets/rocstories/roc_howto_vecs.hdf"
    print("Looking at",path_to_roc_vecs)
    if device is None and dev is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif dev== -1:
        "cuda" if torch.cuda.is_available() else "cpu"
    elif dev is not None:
        device = dev
    print("Loading",path_to_roc_vecs)
    data = pd.read_hdf(path_to_roc_vecs, "mat")
    dict_list = []
    for i in tqdm(data.index):
        r = data.iloc[i].item()
        dict_list.append(r)

    data = pd.DataFrame(dict_list)
    model = SentenceTransformer(model_type,device=device)
    model._first_module().max_seq_length = 509
    arr = np.array([i for i in tqdm(data["vectors"].to_numpy())])
    xb = np.ascontiguousarray(arr)
    # make faiss available
    index = faiss.IndexFlatIP(arr.shape[1])
    index = faiss.index_cpu_to_all_gpus(  # build the index
        index
    )
    index.add(xb)  # add vectors to the index
    if howto:
        story_data = pd.read_csv(base_path+"/localdatasets/rocstories/ROCStories_winter2017 - ROCStories_winter2017.csv").append(
            pd.read_csv(base_path+"/localdatasets/rocstories/ROCStories__spring2016 - ROCStories_spring2016.csv")).append(
            pd.read_csv(base_path+"/localdatasets/wikihow/howtostories.tsv",sep="\t"))
    else:
        story_data = pd.read_csv(base_path+"/localdatasets/rocstories/ROCStories_winter2017 - ROCStories_winter2017.csv").append(
        pd.read_csv(base_path+"/localdatasets/rocstories/ROCStories__spring2016 - ROCStories_spring2016.csv"))
    print("Story data:")
    print(story_data)
    return index

def query_stories(queries,k=5):
    global model,index,data,story_data
    try:
        res = np.array(model.encode(queries))
        # if len(queries)==1:
        #     res = np.squeeze(res,0)
        query = res
        D, I = index.search(query, k)  # sanity check
        content = []
        for story in range(I.shape[0]):
            a = I[story,:]
            matching_stories = data.iloc[a]
            for i in range(matching_stories.shape[0]):
                tit = matching_stories.iloc[i]["titles"]
                content_text = matching_stories.iloc[i]["sentences"].replace(tit + '.', "", 1).strip()
                content_sentences = [x+"." for x in content_text.split(".")]
                sent_encode = model.encode(content_sentences)
                content.append({"title":tit,
                                "content":content_text,
                                "content_sentences":content_sentences,
                                "content_relatedness":[[util.pytorch_cos_sim(query[query_idx],sent_encode[sentence]) for sentence in range(sent_encode.shape[0])] for query_idx in range(res.shape[0])],
                                "story_relatedness": D[story,i]
                                },

                               )

        return content
    except Exception as e:
        print("Error when querying stories:",e)
        print(queries)
        raise Exception("Query Stories Exception")

if __name__ == '__main__':
    initialize_index()
    content = query_stories(["This is a story about a girl","This is a story about a cat"])
    print(content)
