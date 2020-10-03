import time
import numpy as np
from collections import defaultdict
import re
import csv
from bert_serving.client import BertClient
import joblib
import pandas as pd


def build_data_cv(datafile, cv=10, clean_string=True):
    """
    Loads data
    """
    revs = []
    vocab = defaultdict(float)
    with open(datafile, "r", errors='ignore') as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        first_line = True
        for line in csvreader:
            if first_line:
                first_line = False
                continue
            status = []
            sentences = line[1].strip()
            if clean_string:
                orig_rev = clean_str(sentences.strip())
                if orig_rev == '':
                    continue
                words = set(orig_rev.split())
                splitted = orig_rev.split()

                if len(splitted) > 250:
                    # chunk huge sentences to small ones.
                    orig_rev = []
                    splits = int(np.floor(len(splitted) / 250))
                    for index in range(splits):
                        orig_rev.append(' '.join(splitted[index * 250:(index + 1) * 250]))
                    if len(splitted) > splits * 250:
                        orig_rev.append(' '.join(splitted[splits * 250:]))
                    status.extend(orig_rev)
                else:
                    status.append(orig_rev)
            else:
                orig_rev = sentences.strip().lower()
                words = set(orig_rev.split())
                status.append(orig_rev)

            for word in words:
                vocab[word] += 1

            datum = {"y0": 1 if line[2].lower() == 'y' else 0,
                     "y1": 1 if line[3].lower() == 'y' else 0,
                     "y2": 1 if line[4].lower() == 'y' else 0,
                     "y3": 1 if line[5].lower() == 'y' else 0,
                     "y4": 1 if line[6].lower() == 'y' else 0,
                     "text": status,
                     "user": line[0],
                     "num_words": np.max([len(sent.split()) for sent in status]),  # todo: what is this? longest sent?
                     "split": line[7]}  # cv is for determining the cluster
            revs.append(datum)
    return revs, vocab


def get_W_for_bert(word_vecs, k=1024, dtype="float64"):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype=dtype)
    W[0] = np.zeros(k, dtype=dtype)
    i = 1
    for word_idx_pack in word_vecs:
        W[i] = word_vecs[word_idx_pack]
        word_idx_map[word_idx_pack] = i
        i += 1

    print(word_idx_map)
    return W, word_idx_map


def load_bert_vec(revs):
    start_time = time.time()
    now_time = time.time()
    bc = BertClient()

    for rev_idx, rev in enumerate(revs):
        rev_splitted = [orig_rev.split() for orig_rev in rev["text"]]
        print(str((100 * rev_idx + 0.0) / len(revs)) + "% done")
        print(str(time.time() - now_time) + "sec passed")
        print(str((time.time() - start_time) * ((len(revs) - rev_idx) / (rev_idx + 1))) + "sec need to to complete")
        print(str((time.time() - start_time) * (len(revs) / (rev_idx + 1))) + "sec need in total")
        result = bc.encode(rev_splitted, is_tokenized=True)
        rev["embedding"] = result


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s ", string)
    string = re.sub(r"\'ve", " have ", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d", " would ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def get_mairesse_features(file_name):
    feats = {}
    with open(file_name, "r") as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        for line in csvreader:
            feats[line[0]] = [float(f) for f in line[1:]]
    return feats


if __name__ == "__main__":
    data_folder = "essays_200_max_split.csv"
    mairesse_file = "mairesse.csv"
    print("loading data...")
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    num_words = pd.DataFrame(revs)["num_words"]
    max_l = np.max(num_words)
    print("data loaded!")
    print("number of status: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    print("loading word2vec vectors...")
    load_bert_vec(revs)
    print("word2vec loaded!")
    mairesse = get_mairesse_features(mairesse_file)
    filename = 'essays_mairesse_sb_tokenized_200_max_rev_vector.p'
    joblib.dump([revs, vocab, mairesse], filename, protocol=2)
    print("dataset created!")
