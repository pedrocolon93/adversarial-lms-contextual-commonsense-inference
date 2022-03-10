import json
from copy import deepcopy
import numpy as np
import random
np.random.seed(1)





def hint_supply(src, tgt, use_hints=False):
    """
    For a given entry, if use_hints is False, give the model data_without_hints.
    Otherwise, do random sampling of hint components on data_with_hints on the spot,
    and give the model the resulting data_with_hints.
    """
    # if not use_hints:
    #     print("Not giving hint...")
    flag = np.random.choice(5) + 1
    story = src
    hints = tgt
    assertions = deepcopy(tgt)
    hints = hints.strip()
    s_hint, g_hint = hints.split(' ** ')
    ## specific hint
    s_hint = s_hint.strip()
    ss, sr, so = s_hint.split('>')
    ss = ss.strip()
    sr = '>' + sr + '>'
    so = so.strip()
    ## general hint
    g_hint = g_hint.strip()
    gs, gr, go = g_hint.split('>')
    gs = gs.strip()
    gr = '>' + gr + '>'
    go = go.strip()
    general_token = "<|general|>"
    specif = "<|specific|>"

    ## collect
    # hint_components = [" subject: "+ss, " relation: "+sr, " object: "+so, " general subject: "+gs, " general relation: "+gr, " general object: "+go]
    hint_components = [specif+"<|subj|>"+ss, specif+"<|rel|>"+sr, specif+"<|obj|>"+so, general_token+"<|subj|>"+gs,
                       general_token+"<|rel|>"+gr, general_token+"<|obj|>"+go]
    hint_components_idx = [i for i in range(len(hint_components))]

    ## random sampling
    ## (1) make four subsets, each of 1, 2, 3, and 4 components
    subset1 = random.sample(hint_components_idx, 1)
    subset2 = random.sample(hint_components_idx, 2)
    subset3 = random.sample(hint_components_idx, 3)
    subset4 = random.sample(hint_components_idx, 4)
    subset1 = sorted(subset1)
    subset2 = sorted(subset2)
    subset3 = sorted(subset3)
    subset4 = sorted(subset4)
    subset1 = np.array(hint_components)[subset1].tolist()
    subset2 = np.array(hint_components)[subset2].tolist()
    subset3 = np.array(hint_components)[subset3].tolist()
    subset4 = np.array(hint_components)[subset4].tolist()
    ## (2) with 25% chance, choose only one of these subsets
    if use_hints:
        if flag == 1:
            sampled_hints = subset1
        elif flag == 2:
            sampled_hints = subset2
        elif flag == 3:
            sampled_hints = subset3
        elif flag == 4:  # i.e. flag == 4
            sampled_hints = subset4
        else:
            sampled_hints = []
    else:
        sampled_hints = []
    return [story, sampled_hints,assertions]

def hint_supply2(src, tgt, use_hints=False,probability=0.5):
    """
    For a given entry, if use_hints is False, give the model data_without_hints.
    Otherwise, do random sampling of hint components on data_with_hints on the spot,
    and give the model the resulting data_with_hints.
    """
    # if not use_hints:
    #     print("Not giving hint...")
    probability = float(probability)
    hint = np.random.binomial(1, probability)
    sampled_hints = []
    story = src
    hints = tgt
    assertions = deepcopy(tgt)

    if hint:
        flag = np.random.choice(4) + 1
        hints = hints.strip()
        s_hint, g_hint = hints.split(' ** ')
        ## specific hint
        s_hint = s_hint.strip()
        ss, sr, so = s_hint.split('>')
        ss = ss.strip()
        sr = '>' + sr + '>'
        so = so.strip()
        ## general hint
        g_hint = g_hint.strip()
        gs, gr, go = g_hint.split('>')
        gs = gs.strip()
        gr = '>' + gr + '>'
        go = go.strip()
        general_token = "<|general|>"
        specif = "<|specific|>"

        ## collect
        # hint_components = [" subject: "+ss, " relation: "+sr, " object: "+so, " general subject: "+gs, " general relation: "+gr, " general object: "+go]
        hint_components = [specif + "<|subj|>" + ss, specif + "<|rel|>" + sr, specif + "<|obj|>" + so,
                           general_token + "<|subj|>" + gs,
                           general_token + "<|rel|>" + gr, general_token + "<|obj|>" + go]
        hint_components_idx = [i for i in range(len(hint_components))]

        ## random sampling
        ## (1) make four subsets, each of 1, 2, 3, and 4 components
        subset1 = random.sample(hint_components_idx, 1)
        subset2 = random.sample(hint_components_idx, 2)
        subset3 = random.sample(hint_components_idx, 3)
        subset4 = random.sample(hint_components_idx, 4)
        subset1 = sorted(subset1)
        subset2 = sorted(subset2)
        subset3 = sorted(subset3)
        subset4 = sorted(subset4)
        subset1 = np.array(hint_components)[subset1].tolist()
        subset2 = np.array(hint_components)[subset2].tolist()
        subset3 = np.array(hint_components)[subset3].tolist()
        subset4 = np.array(hint_components)[subset4].tolist()
        ## (2) with 25% chance, choose only one of these subsets
        if use_hints:
            if flag == 1:
                sampled_hints = subset1
            elif flag == 2:
                sampled_hints = subset2
            elif flag == 3:
                sampled_hints = subset3
            else:  # i.e. flag == 4
                sampled_hints = subset4
    return [story, sampled_hints,assertions]
import pickle
if __name__ == '__main__':
    # with open('../../data/t5_training_data.tsv', 'r') as f:
    #     data = f.readlines()
    # data_without_hints = []
    # for entry in data_without_hints:
    #     src,tgt = entry.split("\t")
    #     tgt = tgt.strip()
    #     data_without_hints.append([src,tgt])
    # data_with_hints = []
    # for item in data:
    #     src,tgt = item.split("\t")
    #     tgt = tgt.strip()
    #     new_entry = hint_supply(src, tgt, use_hints=True)
    #     data_without_hints.append(new_entry)

    with open('../../data/t5_training_data.tsv', 'r') as f:
        train_data = f.readlines()
    d = []
    with open('../../data/t5_training_data_all.json', 'w') as f:
        for line in train_data:
            src, tgt = line.split("\t")
            tgt = tgt.strip()
            item = {"source":src,"target":tgt}
            d.append(item)
            json.dump(item,f)
            f.write("\n")
    # with open('../../data/t5_test_data.tsv', 'r') as f:
    #     test_data = f.readlines()
    # with open('../../data/t5_test_data.jsonl', 'w') as f:
    #     for line in test_data:
    #         src, tgt = line.split("\t")
    #         tgt = tgt.strip()
    #         json.dump({"source":src,"target":tgt},f)
    #         f.write("\n")


