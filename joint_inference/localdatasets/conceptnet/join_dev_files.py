import csv


def get_real_facts():
    dev_facts = []
    for row in j1:
        if row[3] == "1":
            dev_facts.append(row)
    return  dev_facts


if __name__ == '__main__':
    j1 = csv.reader(open("dev1.txt"),delimiter="\t")
    j1_facts = get_real_facts()
    j1 = csv.reader(open("dev2.txt"),delimiter="\t")
    j2_facts = get_real_facts()
    final_facts = [["Relation","subject","object","strength"]]+j1_facts+j2_facts
    writer = csv.writer(open("dev.txt","w"),delimiter="\t")
    writer.writerows(final_facts)