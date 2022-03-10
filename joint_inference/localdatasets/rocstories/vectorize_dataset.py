import argparse

import torch.cuda
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm

def convert_to_string(x):
    if isinstance(x,float):
        x = ""
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, default='multi-qa-mpnet-base-dot-v1',
                        help='an integer for the accumulator')

    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    data = pd.read_csv("ROCStories_winter2017 - ROCStories_winter2017.csv").append(pd.read_csv("ROCStories__spring2016 - ROCStories_spring2016.csv"))
    vectors = []
    sentences = []
    titles = []

    for i in tqdm(data.index):
        row = data.iloc[i].apply(convert_to_string)
        content = ' '.join([x if "." in x else x+"." for x in row[1:]])
        title = row[1]
        sentences.append(content)
        titles.append(title)

    vectors = model.encode(sentences=sentences,show_progress_bar=True,batch_size=48, device="cuda"if torch.cuda.is_available() else "cpu")

    outfile = "roc_vecs.hdf"
    pd.DataFrame([{"vectors":vector,"sentences":sentence,"titles":title}] for vector,sentence,title in zip(vectors,sentences,titles)).to_hdf("roc_vecs.hdf","mat")