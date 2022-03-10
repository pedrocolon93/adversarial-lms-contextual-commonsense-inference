import os
import json
import sys
from tqdm import tqdm
sys.path.insert(1, '../utils')
from utils import write_items

dir = '../../data/atomic'
save_dir = '../../data'
files = os.listdir(dir)

data = []
max_amount = -1
count = 0
for f in tqdm(files):
    if f.endswith('.jsonl') and 'train' not in f:
        try:
            if max_amount == count:
                break
            data.append(json.dumps(json.load(open(dir + '/' + f))))
            count+=1
        except Exception as e:
            print(f,"has error",e)
new_file = os.path.join(save_dir, 'all_data.jsonl')
write_items(data, new_file)
