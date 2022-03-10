import csv


def get_real_facts():
    dev_facts = []
    for row in j1:
        if row[3] == "1":
            dev_facts.append(row)
    return  dev_facts


if __name__ == '__main__':
    j1 = csv.reader(open("test.txt"),delimiter="\t")
    j1_facts = get_real_facts()
    writer = csv.writer(open("test.txt","w"),delimiter="\t")
    writer.writerows(j1_facts)