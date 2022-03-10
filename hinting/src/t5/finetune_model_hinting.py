import os
import argparse
import random
from uuid import uuid1

import numpy as np
import torch
import torch.nn as nn
import math
import json
import sys
from pathlib import Path
from sklearn.utils import shuffle
sys.path.insert(1, '../utils')
from datasets import roc_stories
from transformers import T5Tokenizer, T5ForConditionalGeneration
from opt import OpenAIAdam
from utils import (encode_dataset4, iter_data2, ResultLogger, make_path)
from loss import LossComputeSeq2Seq
import pickle
from tensorboard_logger import configure, log_value

configure("./log"+str(uuid1()), flush_secs=5)

def transform_story(X1, X2):
    n_batch = len(X1)
    n_ctx = 512
    end_token = [text_encoder.encoder["<|endoftext|>"]]
    xmb_kg = np.tile(np.array([text_encoder.encode('<|PAD|>')[0]] * n_ctx,dtype=np.int32),(n_batch,1))
    xmb_mk = np.ones((n_batch,n_ctx))
    for i, (x1,x2), in enumerate(zip(X1,X2)):
        new_x1 = x1
        new_x2 = x2
        x12 = new_x1
        x13 = new_x2 + end_token
        x14 = new_x2
        x15 = new_x1[-2:] + new_x1[:-2] + end_token
        xmb_kg[i,:len(x12)] = x12
        xmb_kg[i,len(x12):len(x12)+len(x13)] = x13
        xmb_mk[i,len(x14)+len(x15):] = 1
    return xmb_kg, xmb_mk

def iter_apply(X1s, X2s, v_idxs):
    logits = []

    cost = 0
    losses = []
    with torch.no_grad():
        model.eval()
        for b_idx in iter_data2(*v_idxs, n_batch=n_batch_train, truncate=False, verbose=True):
            b_idx = np.array(list(b_idx)).squeeze()
            labels = {}
            for key in X2s:
                labels[key] = X2s[key][b_idx]
            model_inputs = {}
            for key in X1s:
                model_inputs[key] = X1s[key][b_idx]
            model_inputs["labels"] = labels["input_ids"]
            if n_updates == 0:
                past_loss = math.inf
            model.train()
            # XMB_KG = torch.tensor(xmb_kg, dtype=torch.long).to(device)
            # XMB_ST = torch.tensor(xmb_st, dtype=torch.long).to(device)
            for key in model_inputs:
                model_inputs[key] = model_inputs[key].to(device)
            for key in model_inputs:
                if len(model_inputs[key].shape)==1:
                    model_inputs[key] = model_inputs[key].unsqueeze(0)
            lm_logits = model(**model_inputs) #TODO FIX
            loss = compute_loss_fct(train_loss=lm_logits["loss"], only_return_losses=True)
            losses.append(loss)
    return np.sum(losses), np.mean(losses)


def log(save_dir, desc='model', iter=0,save='',save_model=True):
    global best_score
    print("Logging")
    l = len(trX_kg[list(trX_kg.keys())[0]])
    training_idxs = [[x] for x in range(min(l,n_valid))]
    training_idxs = np.array(training_idxs)
    l = len(vaX_kg[list(vaX_kg.keys())[0]])
    val_idxs = [[x] for x in range(min(l,n_valid))]
    val_idxs = np.array(val_idxs)

    for key in trX_kg:
        trX_kg[key] = torch.tensor(trX_kg[key], dtype=torch.long)
    for key in trX_st:
        trX_st[key] =  torch.tensor(trX_st[key], dtype=torch.long)
    for key in vaX_kg:
        vaX_kg[key] = torch.tensor(vaX_kg[key], dtype=torch.long)
    for key in vaX_st:
        vaX_st[key] =  torch.tensor(vaX_st[key], dtype=torch.long)

    tr_sum_loss, tr_mean_loss = iter_apply(trX_kg,trX_st,training_idxs)
    va_sum_loss, va_mean_loss = iter_apply(vaX_kg,vaX_st, val_idxs)
    try:
        log_value('va_sum_loss',va_sum_loss,n_updates)
        log_value('va_mean_loss',va_mean_loss,n_updates)
    except:
        pass
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=float(tr_sum_loss), va_cost=float(va_sum_loss), tr_acc=float(tr_mean_loss), va_acc=float(va_mean_loss))
    print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_sum_loss, va_sum_loss, tr_mean_loss, va_mean_loss))
    path = os.path.join(save_dir, desc, 'best_params_' + str(iter+1) + save)
    if save_model:
        torch.save(model.state_dict(), make_path(path))
    return va_mean_loss

def pad_rels(relation, pad_len=100):
    return relation[:100] + [text_encoder.encode('<|PAD|>')[0]] * (100-len(relation[:100]))

def handle_empty(list_of_rels):
    if len(list_of_rels) == 0:
        return [[] for i in range(args.max_mem_size*5)]
    if len(list_of_rels) < args.max_mem_size*5:
        list_of_rels.extend([[] for i in range(args.max_mem_size*5 - len(list_of_rels))])
    return list_of_rels

def run_epoch(iter):
    losses = []
    i = 0
    l = len(trX_kg[list(trX_kg.keys())[0]])
    for key in trX_kg:
        trX_kg[key] = torch.tensor(trX_kg[key], dtype=torch.long)
    for key in trX_st:
        trX_st[key] =  torch.tensor(trX_st[key], dtype=torch.long)
    idxs = np.array([[i] for i in range(l)])
    for idx in iter_data2(*shuffle(idxs, random_state=np.random),
                                         n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        global past_loss
        # idx should be a batch of indices.
        # need to find those in trxkg
        # also need to convert those found to tensors.
        # print(idx)
        idx = np.array(list(idx)).squeeze()
        tid = np.array(trIds)[idx]
        labels = {}
        for key in trX_st:
            labels[key] = trX_st[key][idx]
        model_inputs = {}
        batch_size = 0
        for key in trX_kg:
            model_inputs[key] = trX_kg[key][idx]
            if batch_size == 0:
                batch_size = trX_kg[key][idx].shape[0]
        model_inputs["labels"] = labels["input_ids"]

        rows_with_hints = []

        for row in range(batch_size):
            dec = text_encoder.decode(model_inputs["input_ids"][row, :].tolist(),skip_special_tokens=False).replace("<s>","").replace("</s>","").replace("<pad>","")
            with text_encoder.as_target_tokenizer():
                dec_tgt = text_encoder.decode([x if x!=-100 else text_encoder.pad_token_id for x in model_inputs["labels"][row,:].tolist()],skip_special_tokens=True)
            tstdc = text_encoder.decode([x if x!=-100 else text_encoder.pad_token_id for x in model_inputs["labels"][row,:].tolist()],skip_special_tokens=False)

            d = json.load(open(os.path.join(args.data_dir, args.kg_type, tid[row] + ".jsonl")))

            story = "source: "+" ".join([d["sentence" + str(i + 1)] for i in range(5)])

            # print(story)
            sent = dec[dec.index("<|") + 2:dec.index("|>")]
            idx_of_period = int(sent.replace("sent", ""))
            sent = d["sentence" + str(idx_of_period + 1)]

            dim = dec[dec.index("<|", dec.index("|>")):dec.index("|>", dec.index("|>") + 1) + 2]
            obj = dec_tgt
            obj = "<|obj|>" + obj
            hint = np.random.binomial(1, 0.5)
            hint_content = ["<|subj|>"+sent, "<|rel|>"+dim, obj]
            if hint == 1 and args.use_hint:
                amount_in_hint = random.sample([1, 2], 1)[0]
                hint_items = sorted(random.sample([0, 1, 2], amount_in_hint))
                # print("here")
                src = story
                hint_txt = "hint: ( " + ','.join([hint_content[i] for i in hint_items]) + " ) "

                hint_enc = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(hint_txt))
                x = src
                special = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(" <|sent" + str(idx_of_period) + "|> "+dim))

                stry = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(story))  + special + hint_enc + [text_encoder.eos_token_id]
                att_mask = [1 for i in range(len(stry))]+[0 for i in range(max_source_length-len(stry))]
                stry += [text_encoder.pad_token_id for i in range(max_source_length-len(stry))]
                tgt = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(obj))


                additionals = [text_encoder.eos_token_id]
                tst1 = text_encoder.decode(tgt+additionals)
                tst2 = text_encoder.decode(stry)
                final_tgt = tgt+additionals+[-100 for i in range(max_target_length-len(tgt+additionals))]
                new_row = stry + tgt + additionals
                new_entry = {}
                new_entry["input_ids"] = stry
                new_entry["attention_mask"] = att_mask
                new_entry["labels"] = final_tgt
                for key in model_inputs:
                    model_inputs[key][row, :] = torch.tensor(new_entry[key])
            else:
                amount_in_hint = random.sample([1, 2], 1)[0]
                hint_items = sorted(random.sample([0, 1, 2], amount_in_hint))
                # print("here")
                src = story

                x = src
                special = text_encoder.convert_tokens_to_ids(
                    text_encoder.tokenize(" <|sent" + str(idx_of_period) + "|> " + dim))

                stry = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(story)) + special + [
                    text_encoder.eos_token_id]
                att_mask = [1 for i in range(len(stry))] + [0 for i in range(max_source_length - len(stry))]
                stry += [text_encoder.pad_token_id for i in range(max_source_length - len(stry))]
                tgt = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(obj))

                additionals = [text_encoder.eos_token_id]
                tst1 = text_encoder.decode(tgt + additionals)
                tst2 = text_encoder.decode(stry)
                final_tgt = tgt + additionals + [-100 for i in range(max_target_length - len(tgt + additionals))]
                new_row = stry + tgt + additionals
                new_entry = {}
                new_entry["input_ids"] = stry
                new_entry["attention_mask"] = att_mask
                new_entry["labels"] = final_tgt
                for key in model_inputs:
                    model_inputs[key][row, :] = torch.tensor(new_entry[key])
        if n_updates == 0:
            past_loss = math.inf
        model.train()
        # XMB_KG = torch.tensor(xmb_kg, dtype=torch.long).to(device)
        # XMB_ST = torch.tensor(xmb_st, dtype=torch.long).to(device)
        for key in model_inputs:
            model_inputs[key] = model_inputs[key].to(device)
        lm_logits = model(**model_inputs) #TODO FIX
        loss = compute_loss_fct(train_loss=lm_logits["loss"], encoder=text_encoder, batch_num=n_updates, accum_steps=int(16/args.n_batch))
        loss = float(loss)
        wandb.log({"loss":loss})

        losses.append(loss)
        n_updates += 1
        if (n_updates + 1) % 20000 == 0:
            try:
                va_loss = log(save_dir,desc, iter,save='_'+str(n_updates),save_model=False)
            except:
                pass
        try:
            log_value('batch_train_loss',loss,n_updates)
            log_value('mean_train_loss',np.mean(losses),n_updates)
            log_value('total_train_loss',np.sum(losses),n_updates)
        except:
            pass
argmax = lambda x: np.argmax(x, 1)
import wandb
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_mem_size',type=int,default=45)
    parser.add_argument('--use_multigpu',action="store_true",default=False)
    parser.add_argument('--use_pretrain',action="store_true",default=False)
    parser.add_argument('--use_filter',type=bool,default=True)
    parser.add_argument('--mem_k',type=int,default=1)
    parser.add_argument('--use_mem',type=bool,default=False)
    parser.add_argument('--desc',type=str,default='model',help="Description")
    parser.add_argument('--comet',action="store_true",default=False)
    parser.add_argument('--kg_type',type=str,default='atomic')
    parser.add_argument('--log_dir', type=str, default='h_log_nopre/')
    parser.add_argument('--model_dir', type=str, default='h_models_nopre/')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=40)
    parser.add_argument('--n_batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_ctx', type=int, default=1024)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--use_hint',action="store_true",default=False)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Constants
    desc = args.desc
    n_ctx = args.n_ctx
    save_dir = args.model_dir
    data_dir = args.data_dir
    log_dir = args.log_dir

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    text_encoder = T5Tokenizer.from_pretrained('t5-base')
    add_toks = ['<|sent0|>', '<|sent1|>', '<|sent2|>', '<|sent3|>', '<|sent4|>',
                '<|xNeed|>', '<|xIntent|>', '<|xWant|>', '<|oEffect|>', '<|xReact|>', '<|oWant|>',
                '<|oReact|>', '<|xEffect|>', '<|xAttr|>', '<|PAD|>',
                "<|subj|>", "<|rel|>", "<|obj|>", "<|general|>", "<|specific|>"]

    special_tokens_dict = {'additional_special_tokens': add_toks}
    num_added_toks = text_encoder.add_special_tokens(special_tokens_dict)
    n_vocab = len(text_encoder)
    print("Encoding dataset...")
    max_source_length = 256
    max_target_length = 128
    try:
        trX_kg, trX_st, trMem, trIds, vaX_kg, vaX_st, vaMem, vaIds =  pickle.load(open(data_dir + '/' + 't_' + 'c' * args.comet + 'h' * (1-args.comet) + '_' + args.use_filter * 'filtered_' + args.kg_type + '_' + 'datat5.pkl','rb'))
    except:
        try:
            data_dump = pickle.load(open(data_dir + '/' + 'c' * args.comet + 'h' * (1-args.comet) + '_' + args.use_filter * 'filtered_' + args.kg_type + '_' + 'datat5.pkl','rb'))
            trX1, trX2, trMem, trIds = data_dump[0]
            vaX1, vaX2, vaMem, vaIds = data_dump[1]
        except:
            STORIES = roc_stories(data_dir, args.comet, args.kg_type)
            print("Encoding dataset...")
            ((trX1, trX2, trMem, trIds),
             (vaX1, vaX2, vaMem, vaIds)) = encode_dataset4(*STORIES,encoder=text_encoder,max_source_length = max_source_length, max_target_length = max_target_length)
            print("Dumping...")
            pickle.dump([(trX1,trX2, trMem, trIds), (vaX1, vaX2, vaMem, vaIds)], open(data_dir + '/' + 'c' * args.comet + 'h' * (1-args.comet) + '_' + args.use_filter * 'filtered_' +  args.kg_type + '_' + 'datat5.pkl','wb'))
            print("Done....")
        # trX_kg, trX_st = transform_story(trX1, trX2)
        # vaX_kg, vaX_st = transform_story(vaX1, vaX2)
        trX_kg = trX1
        trX_st = trX2
        vaX_kg = vaX1
        vaX_st = vaX2

        pickle.dump((trX_kg, trX_st, trMem, vaX_kg, vaX_st, vaMem), open(data_dir + '/t_' + 'c' * args.comet + 'h' * (1-args.comet) + '_' + args.use_filter * 'filtered_' + args.kg_type + '_' + 'data.pkl','wb'))

    # n_train = len(trX_kg)
    # n_valid = len(vaX_kg)
    n_train = len(trX_kg[list(trX_kg.keys())[0]])
    n_valid = len(vaX_kg[list(vaX_kg.keys())[0]])

    n_batch_train = args.n_batch * max(n_gpu, 1)
    n_updates_total = (n_train // n_batch_train) * args.n_iter
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    if not args.use_pretrain:
        model.init_weights()
    model.resize_token_embeddings(len(text_encoder))
    if args.use_multigpu:
        model = nn.DataParallel(model)
    model = model.to(device)
    model_opt = OpenAIAdam(model.parameters(),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)
    compute_loss_fct = LossComputeSeq2Seq(None,model_opt)


    n_updates = 0
    n_epochs = 0
    best_score = 0
    wandb.init(project="hinting",notes=args.desc,config=args)

    for i in range(n_epochs, args.n_iter):
        print("running epoch", i)
        run_epoch(n_epochs)
        n_epochs += 1
        log(save_dir,iter=i)
