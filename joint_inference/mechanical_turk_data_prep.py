# for each model type = 7 models
# sample 100 stories from the test set for each of the trained knowledge bases
# 100 * (3 + 2(3) + 3) = 100 * 12 = 1200
# run each sample without hints to see what it produces
# put to rate on a 5 scale whether it valid
import argparse
import csv
import os
import random

import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import relations
from train_individual_model import ContextualizedRelations, LitSeq2Seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_processes', type=int, default=1,
                        help='an integer for the accumulator')

    parser.add_argument('--cuda', action="store_true", default=False)
    parser.add_argument('--hint_subject', action="store_true", default=False)
    parser.add_argument('--hint_object', action="store_true", default=False)
    parser.add_argument('--hint_relation', action="store_true", default=False)
    parser.add_argument('--hint_random', action="store_true", default=False)
    parser.add_argument('--hint_specificity', action="store_true", default=False)
    parser.add_argument('--test_with_hints', action="store_true", default=False)

    parser.add_argument('--model_name', type=str, default="bart")
    parser.add_argument('--model_type', type=str, default="facebook/bart-base")
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='an integer for the accumulator')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='an integer for the accumulator')
    parser.add_argument('--limit', type=int, default=50,
                        help='an integer for the accumulator')

    parser.add_argument('--epochs', type=int, default=3,
                        help='an integer for the accumulator')
    parser.add_argument('--train_data_limit', type=int, default=-1,
                        help='an integer for the accumulator')

    parser.add_argument('--log_every_n_steps', type=int, default=5,
                        help='an integer for the accumulator')
    parser.add_argument('--fp16', action="store_true", default=False,
                        help='an integer for the accumulator')
    parser.add_argument('--seed', default=142, type=int,
                        help='an integer for the accumulator')
    args = parser.parse_args()
    print("Setting seed",args.seed)
    random.seed(args.seed)
    pytorch_lightning.seed_everything(args.seed)
    models_available = os.listdir("final_models")
    test_files = ["localdatasets/atomic2020/contextualized_test.tsv",
                  "localdatasets/conceptnet/conceptnet_aligned_test.tsv",
                  "localdatasets/glucose/glucose_testing_processed.tsv"]
    test_kbs = ["atomic2020", "conceptnet", "glucose"]
    kb_test_file_map = dict(zip(test_kbs, test_files))
    kb_test_data_map = dict(zip(test_kbs,[None,None,None]))
    combinations = [["atomic2020"], ["conceptnet"], ["glucose"],
                    ["conceptnet", "glucose"], ["atomic2020", "glucose"], ["atomic2020", "conceptnet"],
                    ["atomic2020", "conceptnet", "glucose"]]
    combination_to_model_map = {
        0: "final_models/atomic_only",
        1: "final_models/conceptnet_only",
        2: "final_models/glucose_only",
        3: "final_models/conceptnet_glucose",
        4: "final_models/atomic_glucose",
        5: "final_models/conceptnet_atomic",
        6: "final_models/atomic_glucose_conceptnet"
    }
    for key in combination_to_model_map:
        os.makedirs(combination_to_model_map[key],exist_ok=True)
        combination_to_model_map[key]+="/final.ckpt"

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    tokenizer.add_tokens(relations.additional_tokens)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [relations.tokens[key] for key in relations.tokens.keys()]})
    model = LitSeq2Seq(model_name=args.model_name, model_type=args.model_type,
                       learning_rate=args.learning_rate, tokenizer=tokenizer, hint_subject=args.hint_subject,
                       hint_relation=args.hint_relation, hint_specificity=args.hint_specificity, )

    batch_size = 8
    for c_idx, combination in enumerate(combinations):
        model_file = combination_to_model_map[c_idx]
        model = model.load_from_checkpoint(model_file, model_name=model_file, model_type=args.model_type,
                                           learning_rate=1e-5, tokenizer=tokenizer)
        results = []
        for kb in combination:
            hints = False
            test_file = kb_test_file_map[kb]
            if kb_test_data_map[kb] is None:
                test_data = ContextualizedRelations(test_file, tokenizer=tokenizer, hint_specificity=hints,
                                                    hint_subject=hints, hint_relation=hints,
                                                    random_hints=hints, test_with_hints=hints, test=True,
                                                    limit=args.limit,
                                                    sample_limit=True)
                kb_test_data_map[kb] = test_data
            else:
                test_data = kb_test_data_map[kb]

            test_data.hint_subject = False
            test_data.hint_relation = False
            test_data.hint_specificity = False
            test_data.random_hints = False
            test_data.random_hints = False


            if args.cuda:
                model = model.to("cuda")
            test_loader_no_hints = DataLoader(test_data, batch_size=batch_size)
            for batch in tqdm(test_loader_no_hints):
                tgt = batch.pop("target")
                if args.cuda:
                    for key in batch:
                        batch[key] = batch[key].to("cuda")
                gen = model.encoder.generate(input_ids=batch["input_ids"],
                                            attention_mask=batch["attention_mask"],
                                            max_length=model.max_target_length,
                                            num_beams=5, early_stopping=True, repetition_penalty=1.2,
                                            top_k=120, top_p=0.9)
                bl = batch["labels"].tolist()
                # Fix for pad token not being accounted in loss
                fix_bl = []
                for b in bl:
                    fix_bl.append([bli if bli != -100 else tokenizer.pad_token_id for bli in b])

                gen_rel = tokenizer.batch_decode(gen.tolist(), skip_special_tokens=False)
                gen_rel = [d.replace("<s>", "").replace("</s>", "").replace("<pad>", "") for d in gen_rel]
                [results.append((x,"no_hint",model_file,test_file,tokenizer.decode(batch["input_ids"][x_idx,:].tolist()).replace("<pad>",""),tgt[x_idx])) for x_idx,x in enumerate(gen_rel)]
            hint_test_data = test_data
            hint_test_data.hint_subject = True
            hint_test_data.hint_relation = True
            hint_test_data.hint_specificity = True
            hint_test_data.random_hints = True
            hint_test_data.random_hints = True

            test_loader_hints = DataLoader(hint_test_data, batch_size=batch_size)

            for batch in tqdm(test_loader_hints):
                tgt = batch.pop("target")
                if args.cuda:
                    for key in batch:
                        batch[key] = batch[key].to("cuda")

                gen = model.encoder.generate(input_ids=batch["input_ids"],
                                             attention_mask=batch["attention_mask"],
                                             max_length=model.max_target_length,
                                             num_beams=5, early_stopping=True, repetition_penalty=1.2,
                                             top_k=120, top_p=0.9)
                bl = batch["labels"].tolist()
                # Fix for pad token not being accounted in loss
                fix_bl = []
                for b in bl:
                    fix_bl.append([bli if bli != -100 else tokenizer.pad_token_id for bli in b])

                gen_rel = tokenizer.batch_decode(gen.tolist(), skip_special_tokens=False)
                gen_rel = [d.replace("<s>", "").replace("</s>", "").replace("<pad>", "") for d in gen_rel]
                [results.append((x,"hint",model_file,test_file,tokenizer.decode(batch["input_ids"][x_idx,:].tolist()).replace("<pad>",""),tgt[x_idx])) for x_idx,x in enumerate(gen_rel)]


            results = [["generation","hint","model_file","test_file","input","gold"]]+results
            os.makedirs("mechturk_output",exist_ok=True)
            with open('mechturk_output/results_model_'+str(c_idx)+'.csv', 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile)
                for line in results:
                    spamwriter.writerow(line)

