

import csv
import json

from nltk import sent_tokenize

if __name__ == '__main__':
    t_file = "../../data/test_set_answer_key.csv"
    r = csv.reader(open(t_file))
    headers = None
    final_data = []
    for row in r:
        if headers == None:
            headers=row
            continue
        story = row[1].replace("****"," ")
        if not row[2].strip() in story:
            print("prblem")
        story = story.replace(row[2].strip(),"*"+row[2].strip()+"*")
        story_with_highlighted = story[0:story.index(row[2])]+"*"+row[2]+"*"+story[story.index(row[2])+len(row[2]):]
        start_idx = 4
        for i in range(10):
            idx = start_idx+(2*i)
            if row[idx] == 'escaped': continue
            alts = [ spec+" ** "+gen for spec,gen in zip(row[idx].split("****"),row[idx+1].split("****"))]
            final_data.append([str(i+1)+": "+story,alts])
            # print("here")
    json.dump(final_data,open("../../data/t5_testing_data.json","w"))



