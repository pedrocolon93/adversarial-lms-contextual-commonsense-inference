import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration, AutoTokenizer
from datasets import load_metric

import relations


class LitSeq2Seq(pl.LightningModule):
    def __init__(self, model_name, model_type, learning_rate,tokenizer,max_target_length=256,hint_subject=False, hint_object=False, hint_specificity=False,hint_relation=False,):
        super().__init__()
        self.lr = learning_rate
        self.tokenizer = tokenizer
        self.encoder = T5ForConditionalGeneration.from_pretrained(model_type) if model_name=="t5" else \
            BartForConditionalGeneration.from_pretrained(model_type)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.max_target_length = max_target_length
        self.test_metric = load_metric("sacrebleu")
        self.test_metric_2 = load_metric('rouge')
        self.test_metric_3 = load_metric('meteor')
        self.val_metric = load_metric("sacrebleu")
        self.hint_subject = hint_subject
        self.hint_object = hint_object
        self.hint_relation = hint_relation
        self.hint_specificity = hint_specificity

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        z = self.encoder(**batch)
        loss = z["loss"]
        self.log("train_loss", loss)
        return loss

    def validation_step(self,batch,batch_idx):
        gen = self.encoder.generate(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    max_length=self.max_target_length,
                                    num_beams=5,early_stopping=True,repetition_penalty=1.2,
                                    top_k=120,top_p=0.9)
        bl = batch["labels"].tolist()
        # Fix for pad token not being accounted in loss
        fix_bl = []
        for b in bl:
            fix_bl.append([bli if bli!=-100 else self.tokenizer.pad_token_id for bli in b])

        decodings = self.tokenizer.batch_decode(fix_bl, skip_special_tokens=False)
        decodings = [[d.replace("<s>","").replace("</s>","").replace("<pad>","")] for d in decodings]

        gen_rel = self.tokenizer.batch_decode(gen.tolist(),skip_special_tokens=False)
        gen_rel = [d.replace("<s>","").replace("</s>","").replace("<pad>","") for d in gen_rel]

        if batch_idx%10==0:
            print("Generations",gen_rel)
            print("Gold",decodings)
        self.val_metric.add_batch(predictions=gen_rel, references=decodings)

    def test_step(self,batch,batch_idx):
        gen = self.encoder.generate(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    max_length=self.max_target_length,
                                    num_beams=5,early_stopping=True,repetition_penalty=1.2,
                                    top_k=120,top_p=0.9)
        bl = batch["labels"].tolist()
        # Fix for pad token not being accounted in loss
        fix_bl = []
        for b in bl:
            fix_bl.append([bli if bli!=-100 else self.tokenizer.pad_token_id for bli in b])

        decodings = self.tokenizer.batch_decode(fix_bl, skip_special_tokens=False)
        decodings = [[d.replace("<s>","").replace("</s>","").replace("<pad>","")] for d in decodings]

        gen_rel = self.tokenizer.batch_decode(gen.tolist(),skip_special_tokens=False)
        gen_rel = [d.replace("<s>","").replace("</s>","").replace("<pad>","") for d in gen_rel]
        inputs_to_model = self.tokenizer.batch_decode(batch["input_ids"].tolist(),skip_special_tokens=True)
        # if batch_idx%10==0:
        print("Originals",inputs_to_model)
        print("*"*100)
        print("Generations",gen_rel)
        print("*"*100)
        print("Gold",decodings)
        print("*"*100)

        self.test_metric.add_batch(predictions=gen_rel, references=decodings)
        for g, d in zip(gen_rel,[decodings[i][0] for i in range(len(decodings))]):
            self.test_metric_2.add_batch(predictions=[g], references=[d])
            self.test_metric_3.add_batch(predictions=[g], references=[d])
    def on_validation_epoch_end(self):
        s = self.val_metric.compute()["score"]
        print("val_bleu",s)
        self.log("val_bleu",s,sync_dist=True)

    def on_test_epoch_end(self):
        s = self.test_metric.compute()["score"]
        r = self.test_metric_2.compute()["rouge1"].mid.fmeasure
        m = self.test_metric_3.compute()['meteor']
        print("test_bleu",s)
        print("test_rouge",r)
        print("test_meteor",m)
        self.log("test_bleu", s, sync_dist=True)
        self.log("test_rouge_mid_f1",r, sync_dist=True)
        self.log("test_meteor",m, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class ContextualizedRelations(Dataset):
    def __init__(self, file, tokenizer, subject_delimiter="<subject>", relation_delimiter="<relation>",
                 object_delimiter="<object>", end_of_relation_delimiter="</relation>",
                 story_delimiter="<story>", sentence_delimiter="<sentence>", specific_delimiter="<specific>",general_delimiter="<general>",
                 max_source_length=512, max_target_length=256, ignore_pad_for_loss=True, random_hints=True, test=False,
                 hint_subject=False, hint_object=False, hint_relation=False,hint_specificity=False,limit=-1,
                 test_with_hints=False,sample_limit=False):
        self.ignore_pad_for_loss = ignore_pad_for_loss
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.test = test
        self.random_hints = random_hints

        self.hint_subject = hint_subject
        self.hint_object = hint_object
        self.hint_relation = hint_relation
        self.hint_specificity = hint_specificity
        self.limit=limit
        self.test_with_hints = test_with_hints
        f = pd.read_csv(file,delimiter="\t")

        def convert_relation(rel):
            return relations.relation_map[rel]
        def convert_to_type(rel_type_bool):
            return specific_delimiter if rel_type_bool else general_delimiter
        print("Converting relationships...")
        f["text_relation"] = f["relation"].apply(convert_relation)
        f["text_rel_type"] = f["general"].apply(convert_to_type)
        print("Joining subject,rel,objects...")
        print("Dumping nans...")
        print(f)
        f["target"] = f["text_rel_type"]+" "+subject_delimiter+" "+f["subject"]+" "+relation_delimiter+" "+f["text_relation"]+" "+object_delimiter+" "+f["object"]+" "+end_of_relation_delimiter+" "
        f = f[~f["target"].str.contains("nan",na=True)]
        print(f)
        print("Joining stories and sentences...")
        f["source"] = story_delimiter+" "+f["story"]+" "+sentence_delimiter+" "+f["sentence"]
        self.data = f
        print("Amount of data...:",len(self.data))
        print("Is test?",self.test)
        print("Data will contain:","random_hints", self.random_hints,
        self.hint_subject , "<-hint_subject",
        self.hint_object ,"<-hint_object",
        self.hint_relation , "<-hint_relation")

        if sample_limit:
            self.data = self.data.sample(limit,axis=0)
        print(self.data)

    def __len__(self):
        if self.limit>0 and self.limit!=len(self.data):
            return min(self.limit,len(self.data))
        return len(self.data)

    def __getitem__(self, idx):
        train_batch = self.data.iloc[idx]
        padding = "max_length"
        train_batch["target"] = str(train_batch["target"])
        train_batch["source"] = str(train_batch["source"])
        targets = train_batch["target"]
        if not self.test:
            if self.random_hints or self.hint_subject or self.hint_relation or self.hint_object: # Load up all the hint components
                hint_components = train_batch["target"].split("<")[1:-1]
                hint_components = ["<"+itm for itm in hint_components]
                hst = [self.hint_specificity, self.hint_subject,self.hint_relation,self.hint_object]
                hint_components = [hint_components[hidx] for hidx,hint_part in enumerate(hst) if hint_part]


            if self.random_hints: #Randomly select these components
                random_hint = np.random.binomial(1, 0.5, 1)[0]
                amount_to_hint = random.sample([1, 2, 3], 1)[0]
                random_hint_components = random.sample([t for t in range(len(hint_components))], amount_to_hint)
                random_hint_components = sorted(random_hint_components)
                random_hint_components = [hint_components[t] for t in random_hint_components]
                train_batch["source"] += ( " ( "+';'.join(random_hint_components)+" ) "if random_hint==1 else "")
            elif self.hint_subject or self.hint_relation or self.hint_object or self.hint_specificity: #no randomization always supply those components
                hst = [self.hint_specificity, self.hint_subject,self.hint_relation,self.hint_object]
                tst = [hint_components[hidx] for hidx,hint_part in enumerate(hst) if hint_part]
                train_batch["source"] += " ( " + ';'.join(tst) + ";) "
        else:
            hint_components = train_batch["target"].split("<")[1:-1]
            hint_components = ["<" + itm for itm in hint_components]
            hst = [self.hint_specificity, self.hint_subject, self.hint_relation, self.hint_object]
            hint_components = [hint_components[hidx] for hidx, hint_part in enumerate(hst) if hint_part]
            if len(hint_components)>0:
                train_batch["source"] += " ( " + ';'.join(hint_components) + " ) "

        only_text = self.tokenizer(train_batch["source"],
                                       max_length=self.max_source_length,
                                       padding="max_length",
                                       truncation=True,
                                       return_tensors="pt")
        for key in only_text:
            only_text[key] = only_text[key].squeeze(0)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length,
                                        padding=padding, truncation=True)

        if padding == "max_length" and self.ignore_pad_for_loss:
            labels["input_ids"] = [
                (label if label != self.tokenizer.pad_token_id else -100) for label in labels["input_ids"]
            ]

        only_text["labels"] = torch.tensor(labels["input_ids"])
        only_text["target"] = train_batch["target"]
        return only_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_processes', type=int, default=1,
                        help='an integer for the accumulator')

    parser.add_argument('--cuda', action="store_true", default=False)
    parser.add_argument('--do_train', action="store_true", default=False)
    parser.add_argument('--do_test', action="store_true", default=False)
    parser.add_argument('--hint_subject', action="store_true", default=False)
    parser.add_argument('--hint_object', action="store_true", default=False)
    parser.add_argument('--hint_relation', action="store_true", default=False)
    parser.add_argument('--hint_random', action="store_true", default=False)
    parser.add_argument('--hint_specificity', action="store_true", default=False)
    parser.add_argument('--test_with_hints', action="store_true", default=False)

    parser.add_argument('--train_file',type=str,default="localdatasets/conceptnet/conceptnet_aligned_train600k.tsv")
    parser.add_argument('--dev_file',type=str,default=None)
    parser.add_argument('--test_file',type=str,default="localdatasets/conceptnet/conceptnet_aligned_test.tsv")
    parser.add_argument('--load_checkpoint',type=str,default=None)

    parser.add_argument('--model_name',type=str,default="bart")
    parser.add_argument('--run_name',type=str,required=True)
    parser.add_argument('--model_type',type=str,default="facebook/bart-base")
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='an integer for the accumulator')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='an integer for the accumulator')
    parser.add_argument('--epochs', type=int, default=3,
                        help='an integer for the accumulator')
    parser.add_argument('--train_data_limit', type=int, default=-1,
                        help='an integer for the accumulator')

    parser.add_argument('--log_every_n_steps', type=int, default=5,
                        help='an integer for the accumulator')
    parser.add_argument('--fp16', action="store_true", default=False,
                        help='an integer for the accumulator')
    parser.add_argument('--seed', default=42,type=int,
                        help='an integer for the accumulator')

    args = parser.parse_args()
    seed = args.seed
    seed_everything(seed)
    batch_size = args.batch_size


    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    tokenizer.add_tokens(relations.additional_tokens)
    tokenizer.add_special_tokens({"additional_special_tokens": [relations.tokens[key] for key in relations.tokens.keys()]})
    if args.dev_file is not None:
        dev_data = ContextualizedRelations(args.dev_file,tokenizer=tokenizer,hint_specificity=args.hint_specificity,random_hints=args.hint_random,hint_subject=args.hint_subject,hint_relation=args.hint_relation)
        val_loader = DataLoader(dev_data,batch_size=batch_size)
    else:
        val_loader = None


    logger = WandbLogger(project="joint_inference",config=args,name=args.run_name)
    model = LitSeq2Seq(model_name=args.model_name,model_type=args.model_type,
                       learning_rate=args.learning_rate,tokenizer=tokenizer,hint_subject=args.hint_subject,
                       hint_relation=args.hint_relation,hint_specificity=args.hint_specificity,)
    chkpt = ModelCheckpoint(dirpath=args.run_name, filename=None, monitor="val_bleu", verbose=False,
                            save_last=True, save_top_k=1, save_weights_only=False, mode='max',
                            auto_insert_metric_name=True, every_n_train_steps=None, train_time_interval=None,
                            every_n_epochs=None, save_on_train_epoch_end=True, period=None, every_n_val_epochs=None)
    trainer = pl.Trainer(gpus=torch.cuda.device_count() if torch.cuda.is_available() else None,
                         logger=logger,
                         max_epochs=args.epochs,
                         precision=16 if args.fp16 else 32,
                         log_every_n_steps=args.log_every_n_steps,
                         num_processes=args.num_processes,
                         val_check_interval=0.5,callbacks=[chkpt],
                         accelerator="ddp_sharded"
                         )

    if args.load_checkpoint is not None:
        print('Loading checkpoint',args.load_checkpoint)
        model = model.load_from_checkpoint(args.load_checkpoint,model_name=args.model_name,model_type=args.model_type,
                       learning_rate=args.learning_rate,tokenizer=tokenizer)
    if args.do_train:
        if "," in args.train_file:
            print("Multiple training files detected.")
            dss = []
            for train_file in args.train_file.split(","):
                print("Loading", train_file)
                training_data = ContextualizedRelations(train_file, tokenizer=tokenizer, random_hints=args.hint_random,
                                                        hint_subject=args.hint_subject,
                                                        hint_specificity=args.hint_specificity,
                                                        hint_relation=args.hint_relation, limit=args.train_data_limit)
                dss.append(training_data)
            training_data = ConcatDataset(dss)
        else:
            training_data = ContextualizedRelations(args.train_file, tokenizer=tokenizer, random_hints=args.hint_random,
                                                    hint_subject=args.hint_subject,
                                                    hint_relation=args.hint_relation,
                                                    hint_specificity=args.hint_specificity, limit=args.train_data_limit)
        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        trainer.fit(model, train_loader,val_dataloaders=val_loader)
        trainer.save_checkpoint("final_models/"+str(args.run_name)+"/final.ckpt")
    if args.do_test:
        test_data = ContextualizedRelations(args.test_file, tokenizer=tokenizer, hint_specificity=args.hint_specificity,
                                            hint_subject=args.hint_subject, hint_relation=args.hint_relation,
                                            random_hints=args.hint_random,test_with_hints=args.test_with_hints,test=True)

        test_loader = DataLoader(test_data, batch_size=batch_size)
        trainer.test(model,test_loader)

