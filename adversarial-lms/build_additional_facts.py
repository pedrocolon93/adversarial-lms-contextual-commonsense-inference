import argparse
import json
import os
from multiprocessing import Queue
from multiprocessing.pool import ThreadPool

import numpy as np
import torch.cuda
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss  # make faiss available
from tqdm import tqdm

from assertion import Assertion

devices = ["cuda:" + str(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
models = [SentenceTransformer('all-mpnet-base-v2') for device in devices]


def init(model_queue):
    global mq
    mq = model_queue


def process(chunk):
    global mq
    i = mq.get()
    res = models[i].encode(chunk,
                         show_progress_bar=True,
                         device="cuda:"+str(i) if torch.cuda.is_available() else "cpu",
                         batch_size=64)
    mq.put(i)
    return res

if __name__ == '__main__':

    print("loading")
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input_file', default="train_united_seq2seq.json")
    parser.add_argument('output_additional_facts', default="additional_facts.json")
    parser.add_argument('output_fact_list', default="fact_list.json")
    args = parser.parse_args()
    data = []
    file = args.input_file
    print("opening")
    limit = 30000000
    count = 0
    with open(file) as f:
        for line in tqdm(f):
            if count == limit:
                break
            data.append(json.loads(line))
            count+=1
    # model_queue = Queue()
    # [model_queue.put(i) for i in range(len(models))]
    print("Converting")
    facts = list(set([str(Assertion.parse_from_string(d['relation'])) for d in tqdm(data)][0:limit]))
    print("Starting sentence transformers")
    model = SentenceTransformer('all-mpnet-base-v2')
    p = model.start_multi_process_pool()
    print("Multi-process encode")
    encoded_facts = model.encode_multi_process(sentences=facts, pool=p, batch_size= 128)
    d = 768
    print("Initi index")
    # index_c = faiss.IndexFlatIP(d)  # build the index
    index = faiss.IndexFlatIP(d)  # build the index
    print("Index to gpu")
    res = faiss.StandardGpuResources()  # use a single GPU
    index = faiss.index_cpu_to_gpu(res, 0, index)
    # index = faiss.index_cpu_to_all_gpus(index_c)
    print("done")
    print(index.is_trained)
    index.add(encoded_facts)  # add vectors to the index
    print(index.ntotal)
    encoded_facts = {}
    print("Starting the alignment")
    cache = {}
    for d_idx, d in tqdm(enumerate(data)):
        story, sentence = d["text"].split("<sentence>")
        story = story.replace("<story>","").strip()
        sentence = sentence.strip()
        sentences = sent_tokenize(story)
        # T sentences will contain the sentences we need to find data on.
        target_sentences = []
        for s in sentences:
            target_sentences.append(s)
            if s == sentence:
                break
        all_related_facts = []
        k = 10
        if len(target_sentences) == 0:
            target_sentences.append(str(Assertion.parse_from_string(d["relation"])))
        cat_list = []
        s_v = model.encode_multi_process(sentences=target_sentences, pool=p)
        D, I = index.search(s_v, k)  # sanity check

        for idx, s in enumerate(target_sentences):
            D_2 = D.tolist()[idx]
            I_2 = I.tolist()[idx]

            tgt_fact = str(Assertion.parse_from_string(d["relation"]))
            for z in zip(I_2,D_2):
                if z[0] == d_idx:
                    continue
                found = False
                for memory in all_related_facts:
                    if z[0]==memory[0] or facts[z[0]]==tgt_fact or d_idx==z[0]:
                        found = True
                        break
                if found:
                    continue
                else:
                    all_related_facts.append(z)
        if d_idx not in encoded_facts.keys():
            encoded_facts[d_idx] = []
        for i in all_related_facts:
            encoded_facts[d_idx].append(i)
        if d_idx %10000==0:
            print("Dumping cache facts")
            json.dump(encoded_facts, open(args.output_additional_facts, "w"))
            print("Dumping cache fact list")
            json.dump([d['relation'] for d in data], open(args.output_fact_list, "w"))

    print("Dumping facts")
    json.dump(encoded_facts,open(args.output_additional_facts,"w"))
    print("Dumping fact list")
    json.dump([d['relation'] for d in data],open(args.output_fact_list,"w"))

