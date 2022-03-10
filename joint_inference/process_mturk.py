import json
import os
import re

from lxml import etree


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
    dir = "mechturkfiles/results/"
    model_dict = {}
    for file in os.listdir(dir):
        txt = ''.join(open(os.path.join(dir,file)).readlines())
        q = txt.split("BODY:")[1]
        src = re.findall("<details>(.*?)</details>", txt)[-1]
        model,hint = src.split(";")
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
                "stmnt":stmnt
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
    model = "mechturk_output/results_model_0.csv" #atomic
    print(model)
    print("atomic")
    for hinttype in model_dict[model]:
        if len(model_dict[model][hinttype])>0:
            print(hinttype+"\n")
            acceptables = []
            context_acceptables = []
            alignment_acceptables = []
            statements = []
            for hit in model_dict[model][hinttype]:
                atomic_stmnts.append(hit["stmnt"])
                acceptable = hit["assignments"][0]["general_valid"]['Yes'] and hit["assignments"][1]["general_valid"]['Yes']
                acceptables.append(acceptable)
                context_acceptable = hit["assignments"][0]["context_valid"]['Yes'] and hit["assignments"][1]["context_valid"]['Yes']
                context_acceptables.append(context_acceptable)
                alignment_acceptable = hit["assignments"][0]["gold_context_valid"]['Yes'] and hit["assignments"][1]["gold_context_valid"]['Yes']
                alignment_acceptables.append(alignment_acceptable)
            tot,pos,neg = count_ratios(acceptables)
            print("acceptable","Total",tot,"pos_rat",pos/tot,"neg_rat",neg/tot)
            tot,pos,neg = count_ratios(context_acceptables)
            print("context_acceptable","Total",tot,"pos_rat",pos/tot,"neg_rat",neg/tot)
            tot,pos,neg = count_ratios(alignment_acceptables)
            print("alignment_acceptable","Total",tot,"pos_rat",pos/tot,"neg_rat",neg/tot)
    model = "mechturk_output/results_model_1.csv" #conceptnet
    print(model)
    print("conceptnet")

    for hinttype in model_dict[model]:
        if len(model_dict[model][hinttype])>0:
            print(hinttype+"\n")
            acceptables = []
            context_acceptables = []
            alignment_acceptables = []
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
            tot,pos,neg = count_ratios(acceptables)
            print("acceptable","Total",tot,"pos_rat",pos/tot,"neg_rat",neg/tot)
            tot,pos,neg = count_ratios(context_acceptables)
            print("context_acceptable","Total",tot,"pos_rat",pos/tot,"neg_rat",neg/tot)
            tot,pos,neg = count_ratios(alignment_acceptables)
            print("alignment_acceptable","Total",tot,"pos_rat",pos/tot,"neg_rat",neg/tot)

    model = "mechturk_output/results_model_2.csv"  # glucose
    print(model)
    print("glucose")

    for hinttype in model_dict[model]:
        if len(model_dict[model][hinttype]) > 0:
            print(hinttype + "\n")
            acceptables = []
            context_acceptables = []
            alignment_acceptables = []
            statements = []
            for hit in model_dict[model][hinttype]:
                glucose_stmnts.append(hit["stmnt"])

                acceptable = hit["assignments"][0]["general_valid"]['Yes'] and hit["assignments"][1]["general_valid"][
                    'Yes']
                acceptables.append(acceptable)
                context_acceptable = hit["assignments"][0]["context_valid"]['Yes'] and \
                                     hit["assignments"][1]["context_valid"]['Yes']
                context_acceptables.append(context_acceptable)
                alignment_acceptable = hit["assignments"][0]["gold_context_valid"]['Yes'] and \
                                       hit["assignments"][1]["gold_context_valid"]['Yes']
                alignment_acceptables.append(alignment_acceptable)
            tot,pos,neg = count_ratios(acceptables)
            print("acceptable","Total",tot,"pos_rat",pos/tot,"neg_rat",neg/tot)
            tot,pos,neg = count_ratios(context_acceptables)
            print("context_acceptable","Total",tot,"pos_rat",pos/tot,"neg_rat",neg/tot)
            tot,pos,neg = count_ratios(alignment_acceptables)
            print("alignment_acceptable","Total",tot,"pos_rat",pos/tot,"neg_rat",neg/tot)

    model = "mechturk_output/results_model_6.csv"  # glucose
    print(model)
    for hinttype in model_dict[model]:
        if len(model_dict[model][hinttype]) > 0:
            print(hinttype + "\n")
            acceptables = []
            context_acceptables = []
            alignment_acceptables = []
            statements = []
            for hit in model_dict[model][hinttype]:
                kb = ""
                if hit["stmnt"] in atomic_stmnts:
                    kb = "atomic"
                elif hit["stmnt"] in conceptnet_stmnts:
                    kb = "conceptnet"
                elif hit["stmnt"] in glucose_stmnts:
                    kb = "glucose"
                else:
                    kb = "unk"
                # if hit["stmnt"] not in statements:
                #     statements.append(hit["stmnt"])
                # else:
                #     print("Skupping dupe")
                #     print(hit["stmnt"])
                #     continue
                acceptable = hit["assignments"][0]["general_valid"]['Yes'] and hit["assignments"][1]["general_valid"][
                    'Yes']
                acceptables.append((acceptable,kb))
                context_acceptable = hit["assignments"][0]["context_valid"]['Yes'] and \
                                     hit["assignments"][1]["context_valid"]['Yes']
                context_acceptables.append((context_acceptable,kb))
                alignment_acceptable = hit["assignments"][0]["gold_context_valid"]['Yes'] and \
                                       hit["assignments"][1]["gold_context_valid"]['Yes']
                alignment_acceptables.append((alignment_acceptable,kb))
            for kg in ["atomic","glucose","conceptnet"]:
                tot, pos, neg = count_ratios([temp[0] for temp in filter(lambda x: x[1]==kg,acceptables)])
                print(kg,"acceptable", "Total", tot, "pos_rat", pos / tot, "neg_rat", neg / tot)
                tot, pos, neg = count_ratios([temp[0] for temp in filter(lambda x: x[1]==kg,context_acceptables)])
                print(kg,"context_acceptable", "Total", tot, "pos_rat", pos / tot, "neg_rat", neg / tot)
                tot, pos, neg = count_ratios([temp[0] for temp in filter(lambda x: x[1]==kg,alignment_acceptables)])
                print(kg,"alignment_acceptable", "Total", tot, "pos_rat", pos / tot, "neg_rat", neg / tot)



