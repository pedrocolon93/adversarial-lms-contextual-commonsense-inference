import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from localdatasets.atomic2020.convert_dataset_to_text import rels,text_rels

tokens = {
    "story_start": "<story>",
    "story_end": "</story>",
    "relation_start": "<relation>",
    "relation_end": "</relation>",
    "sentence_start": "<sentence>",
    "sentence_end": "</sentence>",
    "general": "<general>",
    "specific": "<specific>"
}
additional_tokens = ["PersonX", "PersonY",
                     "Something_A", "Something_B", "Something_C", "Something_D", "Something_E",
                     "People_A", "People_B", "People_C", "People_D", "People_E",
                     "Someone_A", "Someone_B", "Someone_C", "Someone_D",
                     "Someone_E",
                     "Some_Event_A", "Some_Event_B", "Some_Event_C", "<end>", "<subj>", "<obj>"]
# atomic_relations = ["oEffect","oReact","oWant","xAttr","xEffect",
#         "xIntent","xNeed","xReact","xWant"]
# atomic_text_relations = ["has the effect on others of",
#              "makes others react",
#              "makes others want to do",
#              "described as",
#              "has the effect",
#              "causes the event because",
#              "before needs to",
#              "after feels",
#              "after wants to"]
atomic_relations = rels
atomic_text_relations = text_rels
atomic_relation_map = dict(zip(atomic_relations,atomic_text_relations))
cn_relation_map = {
    "HasFirstSubevent":"has the first sub event",
    "HasLastSubevent":"has the last sub event",
    "FormOf": "is a form of",
    "IsA": "is a",
    "NotDesires":"does not desire",
    "RelatedTo": "is related to",
    "HasProperty": "has the property",
    "HasContext": "has the context",
    "DerivedFrom": "is derived from",
    "DefinedAs":"is defined as",
    "UsedFor": "is used for",
    "Causes": "causes",
    "Synonym": "is a synonym of",
    "Antonym": "is a antonym of",
    "CapableOf": "is capable of",
    "HasA": "has a",
    "Desires": "desires",
    "AtLocation": "is located at",
    "ReceivesAction": "receives the action",
    "SimilarTo": "is similar to",
    "CausesDesire": "causes a desire for",
    "DistinctFrom": "is distinct from",
    "PartOf": "is a part of",
    "HasSubevent": "has the subevent",
    "HasPrerequisite": "has the prerequisite",
    "MannerOf": "is a manner of",
    "MotivatedByGoal": "is motivated by the goal",
    "MadeOf":"is made of",
    "NotCapableOf":"not capable of",
    "NotIsA":"is not a",
    "NotHasProperty":"does not have the property",
}
glucose_rel_map = {
    "Causes/Enables":"Causes/Enables",
    "Causes":"Causes",
    "Results in":"Results in",
    "Motivates":"Motivates",
    "Enables":"Enables",
}
rel_to_text = {}
rel_to_text.update(atomic_relation_map)
rel_to_text.update(cn_relation_map)
rel_to_text.update(glucose_rel_map)
print("Relations...:",len(rel_to_text.keys()))
for k in rel_to_text.keys():
    print(k)
print("*"*100)
def get_rel_tokens(data):
    toks = data["relation"].unique().tolist()
    return toks



def load_json(x):
    if str(x).lower() != "nan" and str(x).lower() != "":return json.loads(x)
    else: return str(x).lower()
def get_sentence_sim(x):
    return x["sentence_alignment_score"]
def get_story_sim(x):
    return x["story_alignment_score"]

def filter_fun(d):
    d["sentence_alignment_score"] = d["additional"].apply(get_sentence_sim)
    d["story_alignment_score"] = d["additional"].apply(get_story_sim)
    avg_sentence_al_score = np.average(d["sentence_alignment_score"].tolist())
    avg_story_al_score = np.average(d["story_alignment_score"].tolist())
    return d["sentence_alignment_score"]>avg_sentence_al_score

def load_dbs(dbs):
    data = pd.DataFrame()
    for db in dbs:
        d = pd.read_csv(db, sep="\t")
        # try:
        #     d["additional"] = d["additional"].apply(load_json)
        #     d = d[filter_fun(d)]
        # except:
        #     pass
        data = data.append(d, ignore_index=True)
    return data

def replace_person_x(row):
    row = str(row)
    targets = ["PersonX","PersonY"]
    replacements = ["Someone_A","Someone_B"]
    for target in targets:
        if target in row:
            row = row.replace(target,replacements[targets.index(target)])
    return row
import random
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)


    #TODO FIX
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('output_train_file', type=str, default="train_united_seq2seq.json",
                        help='an integer for the accumulator')
    parser.add_argument('output_test_file', type=str, default="test_united_seq2seq.json",
                        help='an integer for the accumulator')

    parser.add_argument('-db', '--database', action='store', dest='dbs',
                        type=str, nargs='*', default=["localdatasets/conceptnet/cn_processed_nohowto.tsv",
                                                      "localdatasets/glucose/glucose_processed.tsv",
                                                      "localdatasets/atomic2020/atomic2020_processed_nohowto.tsv"
            # "localdatasets/wikihow/cn_processed.tsv"
                                                      ],help="Examples: -db item1 item2, -db item3")
    args = parser.parse_args()
    train_output_file = args.output_train_file
    test_output_file = args.output_test_file
    dbs = args.dbs

    print("Loading",dbs)
    data = load_dbs(dbs)
    rel_toks = get_rel_tokens(data)
    train_sentences = []
    test_sentences = []
    n, p = 1, .95  # n = coins flipped, p = prob of success
    p2 = 0.5
    print("Unifying as someone_n")
    for key in data:
        if key=="general":continue
        data[key] = data[key].apply(replace_person_x)
    story_alignment_scores_total = []
    sentence_alignment_scores_total = []
    # limit = 500000
    misaligned = 0
    for i in tqdm(data.index):
        # if i == limit:
        #     break
        try:
            row = data.iloc[i]
            s = np.random.binomial(n, p)
            if row["additional"] != "nan":
                add = json.loads(row["additional"])
                story_alignment_scores_total.append(add['story_alignment_score'])
                sentence_alignment_scores_total.append(add['sentence_alignment_score'])
                if add['story_alignment_score'] < 195:# and add["sentence_alignment_score"]<0.5:
                    misaligned+=1
                    datapoint = {
                        "text": "<story><sentence>",
                        "relation": ' '.join(
                            [str(tokens["general"] if row["general"] else tokens["specific"]),
                             tokens["relation_start"],
                             (rel_to_text[row["relation"]] if row["relation"] in rel_to_text.keys() else row["relation"]),
                             "<subj>", row["subject"],
                             "<obj>", row["object"],
                             tokens["relation_end"]])
                    }
                # Add the suggestion for the relation type and general
                # hintparts = []
                # hintparts.append((tokens["general"] if row["general"] else tokens["specific"]) if np.random.binomial(n,0.75) == 1 else None)
                # hintparts.append((tokens["relation_start"] + " " + (rel_to_text[row["relation"]] if row["relation"] in rel_to_text.keys() else row["relation"])) if np.random.binomial(n,0.75) == 1 else None)
                # hintparts.append((("<subj>" + " " + row["subject"]) if np.random.binomial(n,0.6) == 1 else ("<obj>" + " " + row["object"])) if np.random.binomial(n,0.75) == 1 else None)
                # hint = "( "+' , '.join([hintpart for hintpart in hintparts if hintpart is not None])+" )"
                # if hint == "(  )":
                #     hint = ""
                # hint_datapoint = {
                #     "text": ' '.join([tokens["story_start"], str("" if str(row["story"]) == "nan" else row["story"]),
                #                       tokens["sentence_start"], row["sentence"], hint]),
                #     "relation": ' '.join(
                #         [str(tokens["general"] if row["general"] else tokens["specific"]),
                #          tokens["relation_start"], (rel_to_text[row["relation"]] if row["relation"] in rel_to_text.keys() else row["relation"]),
                #          "<subj>", row["subject"], "<obj>", row["object"], tokens["relation_end"]])
                # }
                # else:
                #     test_sentences.append(datapoint)
                # Add the normal without the suggestion
                else:
                    datapoint = {
                        "text": ' '.join([tokens["story_start"], str("" if str(row["story"]) == "nan" else row["story"]),
                                          tokens["sentence_start"], row["sentence"]]),
                        "relation": ' '.join(
                            [str(tokens["general"] if row["general"] else tokens["specific"]),
                             tokens["relation_start"], (rel_to_text[row["relation"]] if row["relation"] in rel_to_text.keys() else row["relation"]),
                             "<subj>", row["subject"],
                             "<obj>", row["object"],
                             tokens["relation_end"]])
                    }
            else:
                datapoint = {
                    "text": ' '.join([tokens["story_start"],
                                      str("" if str(row["story"]) == "nan" else row["story"]),
                                      tokens["sentence_start"],
                                      row["sentence"]]),
                    "relation": ' '.join(
                        [str(tokens["general"] if row["general"] else tokens["specific"]),
                         tokens["relation_start"],
                         (rel_to_text[row["relation"]] if row["relation"] in rel_to_text.keys() else row["relation"]),
                         "<subj>", row["subject"],
                         "<obj>", row["object"],
                         tokens["relation_end"]])
                }
            if s==1:
                train_sentences.append(datapoint)
            else:
                test_sentences.append(datapoint)
        except Exception as e:
            print(e, i, row)
    final_df = pd.DataFrame(train_sentences)
    print("Avg alignment_scores", np.median(story_alignment_scores_total),np.median(sentence_alignment_scores_total))
    print("Amount misaligned?",misaligned)

    final_df.to_json(train_output_file, lines=True, orient="records")
    test_final_df = pd.DataFrame(test_sentences)
    test_final_df.to_json(test_output_file, lines=True, orient="records")
    print(final_df)
    print(test_final_df)
