import json
import os
import argparse
import random
import ast

from datasets import load_metric
from nltk.translate.bleu_score import SmoothingFunction
from nltk import bleu
import numpy as np
from filelock import FileLock
def add_template(rel, dim, kg_type='atomic'):
    if len(rel) == 0:
       rel = 'none.'
    if rel[-1] != '.':
       rel += '.'

    if 'xEffect' in dim: 
       return 'PersonX is likely: ' + rel 

    if 'oEffect' in dim: 
       return 'PersonY is likely: ' + rel 

    if 'xWant' in dim: 
       return 'PersonX wants: ' + rel 

    if 'oWant' in dim: 
       return 'PersonY wants: ' + rel

    if 'xIntent' in dim: 
       return 'PersonX wanted: ' + rel 

    if 'oIntent' in dim:
       return 'PersonY wanted: ' + rel

    if 'xAttr' in dim: 
       return 'PersonX is seen as: ' + rel

    if 'xNeed' in dim:
       return 'PersonX needed: ' + rel 

    if 'xReact' in dim: 
       return 'PersonX then feels: ' + rel

    if 'oReact' in dim:
       return 'Others then feel: ' + rel
    return rel

def reverse_template(rel):
    prefix = rel.split(':')[0]
    if 'PersonY/Others want' in prefix:
       return 'oWant'
    if 'PersonX wants' in prefix:
       return 'xWant'
    if 'PersonY/Others are likely' in prefix:
       return 'oEffect'
    if 'PersonY/Others then feel' in prefix:
       return 'oReact'
    if 'PersonX then feels' in prefix:
       return 'xReact'
    if 'PersonX is likely' in prefix:
       return 'xEffect'
    if 'PersonX is seen as' in prefix:
       return 'xAttr'
    if 'PersonX needed' in prefix:
       return 'xNeed'
    if 'PersonX wanted' in prefix:
       return 'xIntent'
random.seed(0)


parser = argparse.ArgumentParser(description='Evaluate bleu')
parser.add_argument('--decoded_file',type=str,default='../../data/gen_data/beam_outputs.jsonl')
parser.add_argument('--gold_file',type=str,default='../../data/gold_set.jsonl')
args = parser.parse_args()

original_data = open(args.gold_file)
original_data = [json.loads(l) for l in original_data.readlines()] 
data = [json.loads(l) for l in open(args.decoded_file).readlines()]
dims = ["xNeed","xIntent","xWant","oEffect","xReact","oWant","oReact","xEffect","xAttr"]


hyps = []
refs = []
stories = []
dim_rels = []
content_count = 0
bleu_metric = load_metric("sacrebleu")
rougemetric = load_metric("rouge")
meteor_metric = load_metric("meteor")

for l in original_data:
    stories.append(l['story'])
    d_ = [entry for entry in data if entry['story'] == l['story']]
    if len(d_) == 0:
       continue
    content_count+=1
    d_ = d_[0]
    dim = reverse_template(l['prefix'])
    dim_rels.append(dim)
    gold_rel = add_template(l['rel'],dim)
    gen_rel = d_['<|sent' + str(l['sentID']) + '|>_generated_relations'][dims.index(dim)]
    gen_rel = [add_template(g, dim) for g in gen_rel]
    bleu_metric.add_batch(predictions=gen_rel, references=[[gold_rel] for x in range(len(gen_rel))])
    meteor_metric.add_batch(predictions=gen_rel, references=[gold_rel for x in range(len(gen_rel))])


    for i in range(len(gen_rel)):
        rougemetric.add(prediction=gen_rel[i],reference=gold_rel)
    hyps.extend(gen_rel)
    refs.extend([gold_rel] * len(gen_rel))
# print("Samples evaluated",content_count)
# print('num unique stories: ' + str(len(set(stories))))
bleu_result = bleu_metric.compute()["score"]
roughe_res = rougemetric.compute(use_stemmer=True)
rouge_result = {key: value.mid.fmeasure * 100 for key, value in roughe_res.items()}
meteor_result = meteor_metric.compute()

with FileLock("running.lock"):
    with open("results.txt","a+") as results:
        res_dict = {
            "file":args.decoded_file,
            "bleu":bleu_result,
            "meteor":meteor_result["meteor"]*100
        }
        res_dict.update(rouge_result)
        print(res_dict)
        results.write(json.dumps(res_dict)+"\n")

# print("Bleu",metric.compute())
# print("Rouge",rouge_result)


