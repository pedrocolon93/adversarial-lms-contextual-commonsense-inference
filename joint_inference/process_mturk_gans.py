import json
import os
import re

import scipy as scipy
from lxml import etree
from scipy.stats import spearmanr


def count_ratios(lst):
    # global itm
    tot = 0
    pos = 0
    neg = 0
    for itm in lst:
        if itm:
            pos += 1
        else:
            neg += 1
        tot += 1
    return tot,pos,neg


if __name__ == '__main__':
    dir = "mechturkfilesgans/results/"
    model_dict = {}
    for file in os.listdir(dir):
        txt = ''.join(open(os.path.join(dir,file)).readlines())
        q = txt.split("BODY:")[1]
        src = re.findall("<details>(.*?)</details>", txt)[-1]
        model,hint,score = src.split(";")
        score = float(score.split("=")[1])
        ANS = re.findall("<FreeText>(.*?)</FreeText>", txt)[0]
        stmnt =  re.findall("<strong>(.*?)</strong>", q)[1]

        _,hitid,workerid = file.split("_")
        if model not in model_dict.keys():
            model_dict[model] = {"hint=False":[],"hint=True":[]}
        ans_eval = json.loads(ANS)[0]
        ans_eval["workerid"]=workerid
        entry = None
        for itm_idx,itm in enumerate(model_dict[model][hint]):
            if itm["hitid"]==hitid:
                entry = itm
                break
        if entry is None:
            entry = {
                "hitid":hitid,
                "assignments":[],
                "stmnt":stmnt,
                "score":score
            }
        entry["assignments"].append(ans_eval)
        found = False
        for itm_idx,itm in enumerate(model_dict[model][hint]):
            if itm["hitid"]==hitid:
                model_dict[model][hint][itm_idx]=entry
                found = True
        if not found:
            model_dict[model][hint].append(entry)
    glucose_stmnts = []
    atomic_stmnts = []
    conceptnet_stmnts = []
    model = "mechturk_output/results_model_7.csv" #atomic
    print(model)
    print("atomic")
    for hinttype in model_dict[model]:
        if len(model_dict[model][hinttype])>0:
            print(hinttype+"\n")
            acceptables = []
            context_acceptables = []
            alignment_acceptables = []
            scores = []
            statements = []
            for hit in model_dict[model][hinttype]:
                atomic_stmnts.append(hit["stmnt"])
                acceptable = hit["assignments"][0]["general_valid"]['Yes'] and hit["assignments"][1]["general_valid"]['Yes']
                acceptables.append(acceptable)
                context_acceptable = hit["assignments"][0]["context_valid"]['Yes'] and hit["assignments"][1]["context_valid"]['Yes']
                context_acceptables.append(context_acceptable)
                alignment_acceptable = hit["assignments"][0]["gold_context_valid"]['Yes'] and hit["assignments"][1]["gold_context_valid"]['Yes']
                alignment_acceptables.append(alignment_acceptable)
                scores.append(hit["score"])

            tot,pos,neg = count_ratios(acceptables)
            print("acceptable","Total",tot,"pos_rat",pos/tot,"neg_rat",neg/tot,"correlation",spearmanr(scores,[1 if a else 0 for a in acceptables]))
            tot,pos,neg = count_ratios(context_acceptables)
            print("context_acceptable","Total",tot,"pos_rat",pos/tot,"neg_rat",neg/tot,spearmanr(scores,[1 if a else 0 for a in context_acceptables]))
            tot,pos,neg = count_ratios(alignment_acceptables)
            print("alignment_acceptable","Total",tot,"pos_rat",pos/tot,"neg_rat",neg/tot,spearmanr(scores,[1 if a else 0 for a in alignment_acceptables]))
    model = "mechturk_output/results_model_8.csv" #conceptnet
    print(model)
    print("conceptnet")

    for hinttype in model_dict[model]:
        if len(model_dict[model][hinttype])>0:
            print(hinttype+"\n")
            acceptables = []
            context_acceptables = []
            alignment_acceptables = []
            scores = []
            statements = []
            for hit in model_dict[model][hinttype]:
                conceptnet_stmnts.append(hit["stmnt"])

                # if hit["stmnt"] not in statements:
                #     statements.append(hit["stmnt"])
                # else:
                #     print("Skupping dupe")
                #     print(hit["stmnt"])
                #     continue
                acceptable = hit["assignments"][0]["general_valid"]['Yes'] and hit["assignments"][1]["general_valid"]['Yes']
                acceptables.append(acceptable)
                context_acceptable = hit["assignments"][0]["context_valid"]['Yes'] and hit["assignments"][1]["context_valid"]['Yes']
                context_acceptables.append(context_acceptable)
                alignment_acceptable = hit["assignments"][0]["gold_context_valid"]['Yes'] and hit["assignments"][1]["gold_context_valid"]['Yes']
                alignment_acceptables.append(alignment_acceptable)
                scores.append(hit["score"])

            tot,pos,neg = count_ratios(acceptables)
            print("acceptable","Total",tot,"pos_rat",pos/tot,"neg_rat",neg/tot,"correlation",spearmanr(scores,[1 if a else 0 for a in acceptables]))
            tot,pos,neg = count_ratios(context_acceptables)
            print("context_acceptable","Total",tot,"pos_rat",pos/tot,"neg_rat",neg/tot,spearmanr(scores,[1 if a else 0 for a in context_acceptables]))
            tot,pos,neg = count_ratios(alignment_acceptables)
            print("alignment_acceptable","Total",tot,"pos_rat",pos/tot,"neg_rat",neg/tot,spearmanr(scores,[1 if a else 0 for a in alignment_acceptables]))

