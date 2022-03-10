import argparse
import csv

if __name__ == '__main__':
    headers = ["Relation","subject","object","strength"]
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--infile',type=str,default="train600k.txt")
    args = parser.parse_args()
    j1 = csv.reader(open(args.infile),delimiter="\t")
    dev_facts = []
    for row in j1:
        dev_facts.append(row)
    if dev_facts[0]!=headers:
        print("File is missing headers. Adding the headers",headers)
        dev_facts = [headers]+dev_facts
        writer = csv.writer(open(args.infile, "w"), delimiter="\t")
        writer.writerows(dev_facts)

