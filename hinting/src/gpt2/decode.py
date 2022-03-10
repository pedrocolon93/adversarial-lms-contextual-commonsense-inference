import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.insert(1, '../utils')
from text_utils import TextEncoder, fix_malformed

from decoding import beam_search
import pickle
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformer_models import GPT2MemModel
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_multigpu',action="store_true",default=False)
    parser.add_argument('--n_gpu',type=int,default=1)
    parser.add_argument('--load_epoch',type=str,default='2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kg_type',type=str,default='atomic')
    parser.add_argument('--use_mem',type=bool,default=False)
    parser.add_argument('--comet',type=bool, default=False)
    parser.add_argument('--gen_len',type=int, default=50)
    parser.add_argument('--model_type',type=str,default='./models/model/') #specify model path
    parser.add_argument('--save_filename',type=str,default='outputs.jsonl')
    parser.add_argument('--save_dir',type=str,default='../../data/gen_data/gpt')
    parser.add_argument('--original_file',type=str, default='examples.jsonl')
    parser.add_argument('--data_dir',type=str,default='../../data')
    parser.add_argument('--n_batch',type=int,default=1)
    parser.add_argument('--beam',type=int,default=10)
    parser.add_argument('--filter_decode',type=bool,default=True)
    parser.add_argument('--mem_k',type=int,default=1)
    parser.add_argument('--limit',type=int, default=None)

args = parser.parse_args()
print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
use_mem = args.use_mem
device = torch.device(device)
text_encoder = GPT2Tokenizer.from_pretrained('gpt2')
add_toks = ['<|sent0|>', '<|sent1|>', '<|sent2|>', '<|sent3|>', '<|sent4|>',
            '<|xNeed|>', '<|xIntent|>', '<|xWant|>', '<|oEffect|>', '<|xReact|>', '<|oWant|>',
            '<|oReact|>', '<|xEffect|>', '<|xAttr|>', '<|PAD|>',
            "<|subj|>", "<|rel|>", "<|obj|>", "<|general|>", "<|specific|>"]

special_tokens_dict = {'additional_special_tokens': add_toks}
num_added_toks = text_encoder.add_special_tokens(special_tokens_dict)
n_vocab = len(text_encoder)

sent_ids = [text_encoder.encode('<|sent0|>')[0],text_encoder.encode('<|sent1|>')[0],
            text_encoder.encode('<|sent2|>')[0],text_encoder.encode('<|sent3|>')[0],
            text_encoder.encode('<|sent4|>')[0]]



best_model = 'best_params_' + args.load_epoch
model_path = os.path.join(args.model_type, best_model)

if use_mem:
   model = GPT2MemModel.from_pretrained('gpt2')
else:
   model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(n_vocab)
if args.use_multigpu:
   model = nn.DataParallel(model).to(device)
model = model.to(device)
model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))
model.eval()

print("device", device, "n_gpu", args.n_gpu)
gen_len = args.gen_len

def get_token(next_idx, tokenizer=text_encoder):
    try:
       return tokenizer.decode(next_idx)
    except:
       return next_idx

if not os.path.exists(args.save_dir):
   os.makedirs(args.save_dir)
gen_file = open(os.path.join(args.save_dir, 'beam_' + args.save_filename),'w')

dims_ = {'atomic':["<|xNeed|>","<|xIntent|>","<|xWant|>","<|oEffect|>","<|xReact|>","<|oWant|>","<|oReact|>","<|xEffect|>","<|xAttr|>"]}[args.kg_type]
dims = [text_encoder.encode(d)[0] for d in dims_]

def clean_gen(gen):
    if not isinstance(gen,list):
        gen = gen.tolist()

    gen = [w for w in gen if w != text_encoder.encode('<|PAD|>')[0]]
    gen = [get_token(idx) for idx in gen]
    if '<unk>' in gen:
       gen = [t for t in gen if t != '<unk>']
    gen = "".join([word.replace("Ġ", " ") for word in gen])
    gen = gen.replace("<|endoftext|>","")
    try:
        gen.index("(")
        gen.index(")")
        gen = gen[gen.index(")")+1:]
    except:
        pass
    gen = gen.replace("<|obj|>","").strip()
    if len(gen) > 0 and gen[-1] == ' ':
       gen = gen[:-1]
    return fix_malformed(gen)

def pad_rels(relation, pad_len=100):
    return relation[:100] + [text_encoder.encode('<|PAD|>')[0]] * (100-len(relation[:100]))

if use_mem:
   external_mem = {}

print("Working from...",os.path.abspath("."))
iter_count = 0
gens = []
skipped = 0
max_decode_length = 256
for line_ in tqdm([json.loads(l) for l in open(args.original_file).readlines()]):
    try:
        if args.limit is not None:
            if iter_count == args.limit:
                break
        iter_count+=1
        id = line_["storyid"]
        if use_mem:
           if id not in external_mem.keys():
              external_mem[id] = []
              size_mem = 0
           else:
              continue
        original = [line_['sentence1'],line_['sentence2'],line_['sentence3'],line_['sentence4'],line_['sentence5']]
        save_output = {}
        save_output["storyid"] = id
        save_output["story"] = " ".join(original)
        save_output["gold_relations"] = line_["distance_supervision_relations"]
        for sent_id in tqdm(sent_ids):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    sid = text_encoder.decode(sent_id)
                    for d in tqdm(range(len(dims))):
                         XMB = [text_encoder.encode('<|PAD|>')[0]] * 384
                         if args.filter_decode and ('xIntent' in dims_[d] or 'xNeed' in dims_[d] or 'xAttr' in dims_[d]):
                             context = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(
                                 ' '.join(original[:int(sid.split('<|sent')[1].replace('|>', '')) + 1])))
                         else:
                             context = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(' '.join(original)))
                         hint = (text_encoder.convert_tokens_to_ids(
                         text_encoder.tokenize(line_["hint"])) if "hint" in line_ else [])
                         save_output["hint"]=(line_["hint"]) if "hint" in line_ else False
                         context = context + [sent_id, dims[d]] + hint

                         i_1 = len(context) - 1
                         XMB[:len(context)] = context
                         XMB = torch.tensor(XMB, dtype=torch.long).to(device)
                         XMB = XMB.unsqueeze(0)
                         with autocast():
                            gen = beam_search(model, text_encoder, XMB, i_1, num_beams=args.beam)
                            gen1 = [clean_gen(g) for g in gen]
                         # with autocast():
                         #    gen = model.generate(torch.tensor(context, dtype=torch.long).to(device).unsqueeze(0),max_length=128,num_beams=args.beam,early_stopping=True,num_return_sequences=args.beam,pad_token_id=text_encoder.encode('<|PAD|>')[0])
                         #    gen2 = [clean_gen(g) for g in gen.tolist()]
                         # print(gen)
                         gen = gen1
                         print(gen)
                         if use_mem:
                            mem_gen = gen[0]
                            size_mem += 1
                            external_mem[id].append(text_encoder.convert_tokens_to_ids(text_encoder.tokenize(mem_gen)))
                         if text_encoder.decode(sent_id) + '_' + "generated_relations" in save_output.keys():
                            save_output[text_encoder.decode(sent_id) + '_' + "generated_relations"].append(gen)
                            save_output[text_encoder.decode(sent_id) + '_' + "generated_dims"].append([text_encoder.decode(dims[d])] * len(gen))
                         else:
                            save_output[text_encoder.decode(sent_id) + '_' + "generated_relations"] = [gen]
                            save_output[text_encoder.decode(sent_id) + '_' + "generated_dims"] = [[text_encoder.decode(dims[d])] * len(gen)]
        gens.append(save_output)
        if not os.path.exists(os.path.join(args.save_dir, 'beam_' + args.save_filename)):
            gen_file = open(os.path.join(args.save_dir, 'beam_' + args.save_filename), 'w')
            for save_output in gens:
                gen_file.write(json.dumps(save_output) + '\n')
            gen_file.close()
    except Exception as e:
        print("Problem",e,line_)
        skipped+=1
gen_file = open(os.path.join(args.save_dir, 'beam_' + args.save_filename), 'w')
for save_output in gens:
    gen_file.write(json.dumps(save_output) + '\n')
gen_file.close()
print(skipped)
