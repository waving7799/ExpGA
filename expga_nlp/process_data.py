import pandas as pd
import os, sys
sys.path.insert(0, '../')
import numpy as np
from utils.config_nlp import config
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import OpenAttack as oa



pd.set_option('display.max_colwidth', 1000)

DATA_DIR = '../data_set/Wiki/'
SEED = 12

def word_wiki_process(train_texts, train_labels, test_texts, test_labels, dataset):

    tokenizer = Tokenizer(num_words=config.num_words[dataset])
    tokenizer.fit_on_texts(train_texts)

    maxlen = config.word_max_len[dataset]

    x_train_seq = tokenizer.texts_to_sequences(train_texts)
    x_test_seq = tokenizer.texts_to_sequences(test_texts)
    x_train = sequence.pad_sequences(x_train_seq, maxlen=maxlen, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test_seq, maxlen=maxlen, padding='post', truncating='post')
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    return x_train, y_train, x_test, y_test


def read_wiki_files():
    print("Processing Wiki dataset")
    toxicity_annotated_comments = pd.read_csv(os.path.join(DATA_DIR, 'toxicity_annotated_comments.tsv'), sep = '\t')
    toxicity_annotations = pd.read_csv(os.path.join(DATA_DIR, 'toxicity_annotations.tsv'), sep = '\t')

    annotations_gped = toxicity_annotations.groupby('rev_id', as_index=False).agg({'toxicity': 'mean'})
    all_data = pd.merge(annotations_gped, toxicity_annotated_comments, on = 'rev_id')

    all_data['comment'] = all_data['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    all_data['comment'] = all_data['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
    all_data['comment'] = all_data['comment'].apply(lambda x: x.replace("::", " "))

    all_data['is_toxic'] = all_data['toxicity'] > 0.5

    wiki_splits = {}
    for split in ['train', 'test', 'dev']:
        wiki_splits[split] = all_data.query('split == @split')

    train_text = np.array(wiki_splits["train"]["comment"])
    test_text = np.array(wiki_splits["test"]["comment"])
    train_label = []
    test_label = []
    for item in wiki_splits["train"]["toxicity"]:
        if item >0.5:
            train_label.append([0,1])
        else:
            train_label.append([1,0])
    for item in wiki_splits["test"]["toxicity"]:
        if item >0.5:
            test_label.append([0,1])
        else:
            test_label.append([1,0])
    return train_text, train_label, test_text, test_label

def read_sst_files():
    print("Processing SST dataset")
    dataset = oa.DataManager.load("Dataset.SST")
    train_text = []
    test_text = []
    train_label = []
    test_label = []
    for item in dataset[0]:
        train_text.append(item.x)
        if item.y ==1:
            train_label.append([0,1])
        else:
            train_label.append([1,0])
    for item in dataset[1]:
        test_text.append(item.x)
        if item.y ==1:
            test_label.append([0,1])
        else:
            test_label.append([1,0])
    for item in dataset[2]:
        test_text.append(item.x)
        if item.y == 1:
            test_label.append([0, 1])
        else:
            test_label.append([1, 0])
    train_text = np.array(train_text)
    test_text = np.array(test_text)
    return train_text, train_label, test_text, test_label


if __name__ == '__main__':
    dataset = "sst"
    print(dataset)
    train_texts, train_labels, test_texts, test_labels = read_sst_files()
    x_train, y_train, x_test, y_test = word_wiki_process(train_texts, train_labels, test_texts, test_labels, dataset)
    print(x_train, y_train, x_test, y_test)
    print(x_train.shape, x_test.shape)

