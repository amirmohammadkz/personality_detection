import numpy as np
import csv
import re
from collections import defaultdict


def build_data_cv(datafile, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    vocab = defaultdict(float)
    with open(datafile, "r", errors='ignore') as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        first_line = True
        m = []
        for line in csvreader:
            if first_line:
                first_line = False
                continue
            status = []
            sentences = re.split(r'[.]', line[1].strip())
            # print(sentences)
            try:
                sentences.remove('')
            except ValueError:
                pass
                # print("omg")
            sentences = [sent + "." for sent in sentences]
            last_sentences = []
            for i in range(len(sentences)):
                sents = re.split(r'[?]', sentences[i].strip())
                for s in sents:
                    try:
                        if len(s) == 0:
                            pass
                        elif s[-1] == ".":
                            last_sentences.append(s)
                        else:
                            last_sentences.append(s + "?")
                    except Exception as e:
                        print(s)
            sentences = last_sentences
            x = 0
            for sent in sentences:
                if clean_string:
                    orig_rev = sent.strip()
                    if orig_rev == '':
                        continue
                    words = set(orig_rev.split())
                    splitted = orig_rev.split()
                    x += len(splitted)
                    if len(splitted) > 200:
                        # chunk huge sentences to small ones.
                        orig_rev = []
                        splits = int(np.floor(len(splitted) / 200))
                        for index in range(splits):
                            orig_rev.append(' '.join(splitted[index * 200:(index + 1) * 200]))
                        if len(splitted) > splits * 200:
                            orig_rev.append(' '.join(splitted[splits * 200:]))
                        status.extend(orig_rev)
                    else:
                        status.append(orig_rev)
                else:
                    orig_rev = sent.strip().lower()
                    words = set(orig_rev.split())
                    status.append(orig_rev)

                for word in words:
                    vocab[word] += 1

            datum = {"y0": line[2],
                     "y1": line[3],
                     "y2": line[4],
                     "y3": line[5],
                     "y4": line[6],
                     "text": status,
                     "user": line[0],
                     "num_words": np.max([len(sent.split()) for sent in status]),  # todo: what is this? longest sent?
                     "split": line[7]}  # cv is for determining the cluster
            revs.append(datum)
            m.append(x)
    print(len(revs))
    return revs, vocab


def build_new_data(revs):
    with open("essays_200_max_split.csv", "w") as output:
        output.write("#AUTHID,TEXT,cEXT,cNEU,cAGR,cCON,cOPN,split\n")
        x = False
        xs = []
        rs = []
        for rev in revs:
            if rev["user"] == "1999_474029.txt":
                pass
            now_len = 0
            text = []
            for idx, sent in enumerate(rev["text"]):
                now_len += len(sent.split())
                if now_len <= 200:
                    text.append(sent)
                if now_len > 200 or idx == len(rev["text"]) - 1:
                    x = True
                    txt = " ".join(text)
                    output.write(
                        rev["user"] + ",\"" + txt.replace("\"", "\"\"") + "\"," + rev['y0'] + "," + rev['y1'] + "," +
                        rev['y2'] + "," + rev[
                            'y3'] + "," + rev['y4'] + "," + rev["split"] + "\n")
                    text = [sent]
                    now_len = len(sent.split())
            xs.append(x)
            if x == False:
                rs.append(rev["user"])
            x = False
    pass


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s ", string)
    string = re.sub(r"\'ve", " have ", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d", " would ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " \? ", string)
    #    string = re.sub(r"[a-zA-Z]{4,}", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


if __name__ == "__main__":
    x, y = build_data_cv("essays_original_splitted.csv")
    build_new_data(x)
