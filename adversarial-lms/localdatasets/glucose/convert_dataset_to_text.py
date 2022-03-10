from dataclasses import asdict

import pandas as pd
from tqdm import tqdm

from assertion import Assertion


def process_glucose():
    global story, sentence, point
    gc = pd.read_csv("GLUCOSE_training_data_final.csv")
    final_data = []
    for i in tqdm(range(len(gc.index))):
        entry = gc.iloc[i]
        story = entry["story"]
        sentence = entry["selected_sentence"]
        for j in range(10):
            updated_index = j + 1
            specific_sentence = entry[str(updated_index) + "_specificNL"]
            specific_split = specific_sentence.split(">")

            if specific_sentence != "escaped":
                point = Assertion()
                point.subject = specific_split[0]
                point.relation = specific_split[1]
                point.object = specific_split[2]
                point.general = False
                point.story = story
                point.sentence = sentence
                final_data.append(asdict(point))
            gen_sentence = entry[str(updated_index) + "_generalNL"]
            gen_sentence_split = gen_sentence.split(">")

            if gen_sentence != "escaped":
                point = Assertion()
                point.story = story
                point.sentence = sentence
                point.subject = gen_sentence_split[0]
                point.relation = gen_sentence_split[1]
                point.object = gen_sentence_split[2]
                point.general = True
                final_data.append(asdict(point))
        # break
    final_data = pd.DataFrame(data=final_data)
    final_data.to_csv("glucose_processed.tsv", sep="\t", index=False)
    return final_data

if __name__ == '__main__':
    process_glucose()
