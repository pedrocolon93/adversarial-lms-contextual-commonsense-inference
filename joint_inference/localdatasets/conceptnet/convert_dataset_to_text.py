import argparse
import json
import sys
from math import floor
sys.path.insert(0,"../../")

from relations import relation_map


from dataclasses import asdict
from multiprocessing import Pool, Queue
from assertion import Assertion

import torch

import pandas as pd
from tqdm import tqdm
import language_tool_python

from localdatasets.rocstories.distant_supervision import initialize_index, query_stories


def init(df,tot,q):
    global rephrasing_model, rephrase_tokenizer, tool,cn,device, tq
    cn = df
    tq = tqdm(total=tot)
    device = "cuda:"+str(q.get()) if torch.cuda.is_available() else "cpu"
    # rephrasing_model, rephrase_tokenizer = initialize_rephrase_model(device)
    tool = language_tool_python.LanguageTool('en-US')
    initialize_index()

def process(i):
    try:
        row = cn.iloc[i]
        # if "/c/en/" not in row["start"] or "/c/en/" not in row["finish"]:  UNCOMMENT FOR ASSERTIONS.CSV
        #     tq.update(1)
        #     return None
        start_c = row["start"].replace("/c/en/", "").replace("_", " ")
        if "/" in start_c:
            start_c = start_c.split("/", maxsplit=1)[0].replace("_", " ")
        end_c = row["finish"].replace("/c/en/", "").replace("_", " ")
        if "/" in end_c:
            end_c = end_c.split("/", maxsplit=1)[0].replace("_", " ")

        addd = json.loads(row["additional"])
        weight = addd["weight"]
        if "surfaceText" in addd.keys():
            st = addd["surfaceText"]
            sentence = tool.correct(st.replace("[[", "").replace("]]", ""))

        else:
            st = tool.correct(start_c + " " +
                              relation_map[row["rel"].replace("/r/", "")] + " " +
                              end_c + ".")
            sentence = st
        if "." not in sentence:
            sentence = sentence + ". "
        sentence = sentence #+ " " + ' '.join(
            # rephrase(rephrase_tokenizer, rephrasing_model, sentence, amount_of_rephrases=1, device=device)[0:3])
        nearest_story = query_stories([sentence], k=5)[0]
        # if nearest_story["story_relatedness"] < 160:
        #     tq.update(1)
        #     return None
        # relatedness.append(nearest_story["story_relatedness"])
        nearest_sentence = nearest_story["content_sentences"][
            nearest_story["content_relatedness"][0].index(max(nearest_story["content_relatedness"][0]))]
        relation = row["rel"].replace("/r/", "")
        ass = Assertion()
        ass.sentence = nearest_sentence.strip()
        ass.subject = start_c
        ass.object = end_c
        ass.relation = relation
        ass.story = nearest_story["content"]
        additional = {"weight": float(weight), "story_alignment_score": float(nearest_story["story_relatedness"]),
                      "sentence_alignment_score":float(max(nearest_story["content_relatedness"][0]))}
        ass.additional = json.dumps(additional)
        tq.update(1)
        return asdict(ass)
    except Exception as e:
        print(e)
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--process_num', type=int, default=1,
                        help='an integer for the accumulator')
    parser.add_argument('--cuda_num', type=int, default=1,
                        help='an integer for the accumulator')
    parser.add_argument('--infile',type=str,default="train600k.txt")
    parser.add_argument('--outfile',type=str,default="cn_processed_train.tsv")

    args = parser.parse_args()
    print("Using the parameters:",args)
    processnum = args.process_num
    cuda_count = args.cuda_num
    outfile = args.outfile
    infile = args.infile

    print("Loading data")
    cn = pd.read_csv(infile,sep="\t")
    print("Done. Changing names to general processing.")
    # summary	rel	start	finish	additional
    # Relation	subject	object	strength
    cn["rel"] = cn["Relation"]
    cn["start"] = cn["subject"]
    cn["finish"] = cn["object"]
    cn = cn[cn["strength"]>=0]
    cn["additional"] = '{"weight":'+cn["strength"].apply(str)+'}'
    # print("removing data") UNCOMMENT FOR ASSERTIONS.CSV
    # cn = cn[(cn["start"].str.contains("/c/en"))&(cn["finish"].str.contains("/c/en"))].reset_index(drop=True)
    print(cn)
    # init()
    final_data = []
    relatedness = []
    results = []
    save = -1
    q = Queue()
    for x in [i % cuda_count for i in range(processnum)]:
        q.put(x)
    with Pool(processnum,initializer=init,initargs=[cn,int(len(cn.index)/processnum),q]) as p:
        for l in p.imap_unordered(process,cn.index):
            if l is not None:
                results.extend([l])
                t = len(results)/10000
                if floor(t) > save:
                    print("Dumping cache!!!")
                    pd.DataFrame(data=results).to_csv(args.outfile, sep="\t", index=False)
                    save = floor(t)

        # print(ass)
    pd.DataFrame(data=results).to_csv(args.outfile, sep="\t", index=False)

