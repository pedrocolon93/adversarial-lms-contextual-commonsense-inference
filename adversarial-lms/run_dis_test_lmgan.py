import datetime
import json
import sys

import torch.cuda
from datasets import load_metric
from numpy import average, median, mean
from tqdm import tqdm

from kbs_to_text_unite_summary import rel_to_text
from model.kbart.kgcbartgannormalgan import BartGAN


def generate_sequence(input_text, model, tokenizer, device, top_k=155,p=False,max_length=24,num_beams=1,sample=False):
    print("Generating using device",device)
    if not isinstance(input_text,list):
        input_text = [input_text]
    s = datetime.datetime.now()
    source = tokenizer(input_text, max_length=max_length, pad_to_max_length=True,
                                         return_tensors='pt')
    source_ids = source['input_ids']  # .unsqueeze()
    source_mask = source['attention_mask']  # .unsqueeze()

    if device is not None:
        source_ids = source_ids.to(device)  # .unsqueeze()
        source_mask = source_mask.to(device)  # .unsqueeze()

    generated_ids = model.generate(
        input_ids=source_ids,
        attention_mask=source_mask,
        max_length=max_length,
        num_beams=num_beams,
        do_sample=sample,
        top_k=top_k
    )
    f = datetime.datetime.now()
    print(f-s)
    res = []
    for i in range(generated_ids.shape[0]):
        gids = generated_ids[i, :].tolist()
        s = tokenizer.decode(gids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        if s not in res:
            res.append(s)
        if p:
            print(s)
    return res


def evaluate_gan_model(model):
    ctext = "<story> "
    acc = load_metric("accuracy")
    # with open("localdatasets/conceptnet/test.txt") as testfile:
    #     tb = []
    #     for line in tqdm(testfile):
    #         lines = line.split("\t")
    #         b = {
    #             "sources":"<story><sentence>",
    #             "target":"<specific> <relation> "+rel_to_text[lines[0]]+" <subj> "+lines[0]+" <obj> "+lines[1],
    #             "score":int(lines[-1].strip())
    #         }
    #         tb.append(b)
    #         if len(tb)==8:
    #             res = model.eval_assertion(tb,threshold=0.6)
    #             acc.add_batch(predictions=res,references=[it["score"] for it in tb])
    #             tb = []
    print("loading")
    allscores = []
    bn = 0
    with open("localdatasets/conceptnet/concepntet_aligned_test_full.tsv") as testfile:
        tb = []
        skip_first = True
        for line in tqdm(testfile):
            if skip_first:
                skip_first = False
                continue
            lines = line.split("\t")
            b = {
                "sources": "<story>" + lines[0] + " <sentence>" + lines[1],
                "target": "<specific> <relation> " + rel_to_text[lines[4]] + " <subj> " + lines[2] + " <obj> " + lines[
                    3] + "</relation>",
                "score": int(lines[6][14:15])
            }
            tb.append(b)
            if len(tb) == 8:
                bn += 1
                if bn == 150:
                    print("Halfway")
                threshold = 0.5
                res = model.eval_assertion(tb)
                final_out = []
                for o in res:
                    allscores.append(o)
                    if o > threshold:
                        final_out.append(1)
                    else:
                        final_out.append(0)
                acc.add_batch(predictions=final_out, references=[it["score"] for it in tb])
                tb = []
    print("ACC", acc.compute(), mean(allscores))

def evaluate_gan_model(model):
    ctext = "<story> "
    acc = load_metric("accuracy")
    # with open("localdatasets/conceptnet/test.txt") as testfile:
    #     tb = []
    #     for line in tqdm(testfile):
    #         lines = line.split("\t")
    #         b = {
    #             "sources":"<story><sentence>",
    #             "target":"<specific> <relation> "+rel_to_text[lines[0]]+" <subj> "+lines[0]+" <obj> "+lines[1],
    #             "score":int(lines[-1].strip())
    #         }
    #         tb.append(b)
    #         if len(tb)==8:
    #             res = model.eval_assertion(tb,threshold=0.6)
    #             acc.add_batch(predictions=res,references=[it["score"] for it in tb])
    #             tb = []
    print("loading")
    allscores = []
    bn = 0
    with open("localdatasets/conceptnet/concepntet_aligned_test_full.tsv") as testfile:
        tb = []
        skip_first = True
        for line in tqdm(testfile):
            if skip_first:
                skip_first = False
                continue
            lines = line.split("\t")
            b = {
                # "sources": "<story>" + lines[0] + " <sentence>" + lines[1],
                "sources": "<story>" + "" + " <sentence>" + "",
                "target": "<specific> <relation> " + rel_to_text[lines[4]] + " <subj> " + lines[2] + " <obj> " + lines[
                    3] + "</relation>",
                "score": int(lines[6][14:15])
            }
            tb.append(b)
            if len(tb) == 8:
                bn += 1
                if bn == 150:
                    print("Halfway")
                threshold = 0.6
                res = model.eval_assertion(tb)
                final_out = []
                for o in res:
                    allscores.append(o)
                    if o > threshold:
                        final_out.append(1)
                    else:
                        final_out.append(0)
                acc.add_batch(predictions=final_out, references=[it["score"] for it in tb])
                tb = []
    print("ACC", acc.compute(), mean(allscores))

def evaluate_dis_model(model):
    ctext = "<story> "
    acc = load_metric("accuracy")
    # with open("localdatasets/conceptnet/test.txt") as testfile:
    #     tb = []
    #     for line in tqdm(testfile):
    #         lines = line.split("\t")
    #         b = {
    #             "sources":"<story><sentence>",
    #             "target":"<specific> <relation> "+rel_to_text[lines[0]]+" <subj> "+lines[0]+" <obj> "+lines[1],
    #             "score":int(lines[-1].strip())
    #         }
    #         tb.append(b)
    #         if len(tb)==8:
    #             res = model.eval_assertion(tb,threshold=0.6)
    #             acc.add_batch(predictions=res,references=[it["score"] for it in tb])
    #             tb = []
    print("loading")
    allscores = []
    bn = 0
    with open("localdatasets/conceptnet/concepntet_aligned_test_full.tsv") as testfile:
        tb = []
        skip_first = True
        for line in tqdm(testfile):
            if skip_first:
                skip_first = False
                continue
            lines = line.split("\t")
            b = {
                "sources": "<story>" + lines[0] + " <sentence>" + lines[1],
                "target": "<specific> <relation> " + rel_to_text[lines[4]] + " <subj> " + lines[2] + " <obj> " + lines[
                    3] + "</relation>",
                "score": int(lines[6][14:15])
            }
            tb.append(b)
            if len(tb) == 8:
                bn += 1
                if bn == 150:
                    print("Halfway")
                threshold = 0.5
                #TODO CHECK ME
                res = model.eval_assertion(tb)
                final_out = []
                for o in res:
                    allscores.append(o)
                    if o > threshold:
                        final_out.append(1)
                    else:
                        final_out.append(0)
                acc.add_batch(predictions=final_out, references=[it["score"] for it in tb])
                tb = []
    print("ACC", acc.compute(), mean(allscores))


if __name__ == '__main__':
    model_path = sys.argv[1]
    print("Loading",model_path)
    model = BartGAN.load_from_checkpoint(model_path)
    # tokenizer = model.generator_tok
    # model = model.generator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ctext = "<story> It was raining all day. " \
    #         "<sentence> It was raining all day. ( <relation> is located at) "
    model.to(device)
    evaluate_gan_model(model)

