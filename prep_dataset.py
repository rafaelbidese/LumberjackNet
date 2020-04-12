import json, os
import random
from collections import namedtuple
import numpy as np
random.seed(0)


def _json_object_hook(d):
    return namedtuple('tree', d.keys())(*d.values())

def json2obj(data):
    return json.loads(data, object_hook=_json_object_hook)

def _obj_to_dict(obj):
    return obj._asdict()

def getDataset(json_path, split):
    with open(json_path,'r') as json_file:
        json_data = json_file.read()
        trees = json.loads(json_data)
    raw_data = json2obj(json.dumps(trees['trees']))
    random.shuffle(raw_data)
    val_length = int(np.floor(len(raw_data) * split))
    val, train  = raw_data[:val_length], raw_data[val_length+1:] 
    return [train, val]

def main(json_path,split):
    datasets = getDataset(json_path,split)
    outputs = ['train_split.json', 'val_split.json']
    for output, dataset in zip(outputs, datasets):
        with open(output, 'w') as json_file:
            json.dump([obj._asdict() for obj in dataset],json_file)

if __name__ == '__main__':
    json_path = r'data/sounds_DBH_inches.json'
    split = 0.3
    main(json_path,split)
    print("Dataset splits successfully created")