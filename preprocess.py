import numpy as np
import os
import os.path
from collections import defaultdict
import re

<<<<<<< HEAD
dataset_path = 'dataset\\'
=======
path = 'dataset\\'
>>>>>>> 4da4654d1f22aaac7da8fe8859d21d963a7f9a2d
processed_path = 'processed\\'

def gather_dataset(path):
    assert os.path.exists(path)
    dirs = [path + dirname + '\\' for dirname in os.listdir(path) if not os.path.isfile(path + dirname)]
    train_dir = dirs[0] if 'train' in dirs[0].lower() else dirs[1]
    test_dir = dirs[0] if 'test' in dirs[0].lower() else dirs[1]
    topics_list = [topic for topic in os.listdir(train_dir)]
        
    return train_dir, test_dir, topics_list

def collect_data_from(dir, topics_list):
    data = []
    for topic_id, topic in enumerate(topics_list):
        label = topic_id
        dir_path = dir + topic + '/'
        files = [(filename, dir_path + filename) for filename in os.listdir(dir_path) if os.path.isfile(dir_path + filename)]
        files.sort()

        for filename, filepath in files:
            with open(filepath, encoding = 'utf-16') as f:
                text = f.read().lower()
                words = [word for word in re.split('\W+', text) if word not in stop_words]
                content = ' '.join(words)
                assert len(content.splitlines()) == 1
                data.append(str(label) + '<fff>' + filename + '<fff>' + content)
        print('topic ' + str(label) + ': done')
    print('----------------------')
    return data

if not os.path.exists(processed_path):
    os.mkdir(processed_path)

def write_data(path, data):
    if not os.path.exists(path):
        with open(path, 'w', encoding = 'utf-16') as f:
            f.write('\n'.join(data))

train_dir, test_dir, topics_list = gather_dataset(path = dataset_path)

with open(dataset_path + 'vietnamese-stopwords.txt', encoding = 'utf-8') as f:
    stop_words = [stop_word for stop_word in f.read().splitlines()]

train_data = collect_data_from(dir = train_dir, topics_list = topics_list)
test_data = collect_data_from(dir = test_dir, topics_list = topics_list)

write_data(processed_path + 'processed_train_data.txt', train_data)
write_data(processed_path + 'processed_test_data.txt', test_data)
