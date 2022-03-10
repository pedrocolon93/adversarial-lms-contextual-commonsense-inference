import argparse
import json
import sys
from math import floor
from multiprocessing import Pool
from multiprocessing import Queue

sys.path.insert(0,"/u/pe25171/kbart/")
sys.path.insert(0,"../../")
from dataclasses import asdict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

from assertion import Assertion
from localdatasets.rocstories.distant_supervision import initialize_index, query_stories, base_path

rels = ["oEffect","oReact","oWant","xAttr","xEffect",
        "xIntent","xNeed","xReact","xWant"]
text_rels = ["has the effect on others of [FILL]",
             "makes others react [FILL]",
             "makes others want to do [FILL]",
             "described as [FILL]",
             "has the effect [FILL]",
             "causes the event because [FILL]",
             "before needs to [FILL]",
             "after feels [FILL]",
             "after wants to [FILL]"]
rels = ["AtLocation",
 "CapableOf",
 "Causes",
 "CausesDesire",
 "CreatedBy",
 "Desires",
"HasA",
 "HasFirstSubevent",
 "HasLastSubevent",
 "HasPrerequisite",
 "HasProperty",
 "HasSubEvent",
 "HinderedBy",
 "InstanceOf",
 "isAfter",
"isBefore",
"isFilledBy",
"MadeOf",
 "MadeUpOf",
 "MotivatedByGoal",
 "NotDesires",
 "ObjectUse",
 "UsedFor",
 "oEffect",
"oReact",
"oWant",
"PartOf",
 "ReceivesAction",
 "xAttr",
"xEffect",
 "xIntent",
 "xNeed",
 "xReact",
 "xReason",
 "xWant"]

text_rels = ["located or found at/in/on",
             "is/are capable of",
"causes",
"makes someone want",
"is created by",
"desires",
"has, possesses or contains",
"begins with the event/action",
"ends with the event/action",
"to do this, one requires",
"can be characterized by being/having","includes the event/action",
"can be hindered by",
"is an example/instance of",
"happens after",
"happens before",
"blank can be filled by",
"is made of",
"made (up) of",
"is a step towards accomplishing the goal","do(es) not desire",
"used for","used for",
"has the effect on others of",
"makes others react",
"makes others want to do",
"is a part of",
"can receive or be affected by the action",
 "described as",
"has the effect",
"causes the event because",
"before needs to",
"after feels",
"because",
"after wants to"]
print(len(text_rels),len(rels))
def init(d,tot,q,howto):
    global unmasker,device,data, mask_,tq
    data = d
    device = q.get() if torch.cuda.is_available() else -1
    tq = tqdm(total=tot)
    initialize_index(dev=device,howto=howto)
    unmasker = pipeline('fill-mask', model='roberta-large', device= device,)
    mask_ = "<mask>"

def process(i):
    try:
        asses = []
        row = data.iloc[i].tolist()
        event = row[1]
        dimension = row[2]
        sample = row[3]
        try:
            # if "___" in event:
            #     print("Event")
            explanation = text_rels[rels.index(dimension)]
            explanation = explanation + " " + sample
            nearest_story = query_stories([explanation])[0]
            story = query_stories([explanation])[0]["content"]
            nearest_sentence = nearest_story["content_sentences"][
                nearest_story["content_relatedness"][0].index(max(nearest_story["content_relatedness"][0]))]

            ass = Assertion()
            replacements = ["PersonX", "PersonY"]
            causal_event = event
            # if "PersonY" in event:
            #     print("Person Y~!!!", event)
            primed = story
            replaced = False
            for replacement in replacements:
                if replacement in causal_event:
                    replaced = True
                    masked_event = causal_event.replace(replacement, "%s" % mask_, 1)
                    # print(masked_event)
                    primed = story
                    stm = primed[0:primed.find(nearest_sentence)+len(nearest_sentence)] + " " + masked_event+ ". "+ primed[primed.find(nearest_sentence)+len(nearest_sentence):]
                    reps = unmasker(stm)
                    for sub in reps:
                        if "_" in sub["token_str"] or sub["token_str"].strip()=="":
                            print("Skipping token:", sub["token_str"],";")
                            continue
                        else:
                            causal_event = sub["sequence"]
                            # if masked_event.count(replacement)>1:
                            causal_event = causal_event.replace(replacement, sub["token_str"])
                            break
                            # print(causal_event)

            # print("Causal")

            if "___" in event:
                # proceed to fill
                explanation = text_rels[rels.index(dimension)]
                explanation = explanation +" "+sample
                causal_event = causal_event.replace("___", mask_)
                causal_event = causal_event + " ; " + explanation
                # print(causal_event)
                causal_event = unmasker(causal_event)[0]["sequence"]
            else:
                print("pass")
            # if "PersonY" in event:
            #     print("Person Y~!!!",event,causal_event)
            # ass.sentence = causal_event.replace(story, "").split(";")[0].strip()+"."
            if replaced:
                ass.sentence = nearest_sentence
                ass.story = story
                ass.subject = event
                ass.general = True
                ass.object = sample
                ass.relation = dimension
                asses.append(asdict(ass))

            ass = Assertion()
            ass.sentence = nearest_sentence
            ass.story = story
            prep = causal_event[primed.find(nearest_sentence)+len(nearest_sentence):]
            prep = prep[:prep.find(".")+1].strip()
            ass.subject = prep
            ass.general = False
            ass.object = sample
            ass.relation = dimension
            asses.append(asdict(ass))
        except Exception as e:
            print(e)
            print(row)
            print()

        tq.update(1)
        return asses
    except:
        tq.update(1)

        return []
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--process_num', type=int, default=1,
                        help='an integer for the accumulator')
    parser.add_argument('--cuda_num', type=int, default=1,
                        help='an integer for the accumulator')
    parser.add_argument('--howto', action="store_true", default=False,
                        help='an integer for the accumulator')
    args = parser.parse_args()
    process_num = args.process_num
    cuda_count = args.cuda_num
    data = pd.read_csv("train.tsv",sep="\t").drop_duplicates().reset_index()

    save = -1
    results = []
    howto=args.howto
    q = Queue()
    for x in [i % cuda_count for i in range(process_num)]:
        q.put(x)
    with Pool(process_num,initializer=init,initargs=[data,int(len(data.index)/process_num),q,howto]) as p:
        # final_data = p.map(process,data.index) #list of lists that may be null
        for l in p.imap_unordered(process,data.index):
            results.extend([x for x in l if x is not None])
            t = len(results)/10000
            if floor(t)>save:
                print("Dumping cache!!!")
                pd.DataFrame(data=results).to_csv("temp_atomic2020_processed.tsv", sep="\t", index=False)
                save = floor(t)
    if howto:
        pd.DataFrame(data=results).to_csv("atomic2020_processed_howto.tsv", sep="\t", index=False)
    else:
        pd.DataFrame(data=results).to_csv("atomic2020_processed_nohowto.tsv", sep="\t", index=False)
