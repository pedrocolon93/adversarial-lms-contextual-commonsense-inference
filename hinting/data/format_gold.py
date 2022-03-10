import json
from nltk import sent_tokenize
from tqdm import tqdm

original_file = "gold_set.jsonl"
print("Opening")
with open('gold_fixed.jsonl', 'w') as outfile:
    print("Going for it")
    for line_ in tqdm([json.loads(l) for l in open(original_file).readlines()]):
        sentences = [x for x in sent_tokenize(line_["story"])]
        storyid = ""
        item = {"story":line_["story"],"storyid":storyid}
        item["distance_supervision_relations"] = []
        for i in range(len(sentences)):
            item["sentence"+str(i+1)] = sentences[i]
        json.dump(item, outfile)
        outfile.write('\n')

