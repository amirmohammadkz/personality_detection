import re
import time
import svm

import numpy as np
import joblib
from bert_serving.client import BertClient
import os.path
from joblib import dump, load


def shrink_text(text, clean_string=True):
    status = []
    sentences = re.split(r'[.]', text.strip())
    # print(sentences)
    try:
        sentences.remove('')
    except ValueError:
        pass
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
            splitted = orig_rev.split()
            x += len(splitted)
            if len(splitted) > 200:
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
            status.append(orig_rev)
    return status


def build_new_chunks(sentences):
    now_len = 0
    text = []
    text_chunks = []
    for idx, sent in enumerate(sentences):
        now_len += len(sent.split())
        if now_len <= 200:
            text.append(sent)
        if now_len > 200 or idx == len(sentences) - 1:
            x = True
            txt = " ".join(text)
            text_chunks.append(txt.replace("\"", "\"\""))
            text = [sent]
            now_len = len(sent.split())
    return text_chunks


def preocess_chunks(text_chunks, clean_string=True):
    revs = []
    for chunk in text_chunks:
        status = []
        sentences = chunk.strip()
        if clean_string:
            orig_rev = clean_str(sentences.strip())
            if orig_rev == '':
                continue
            splitted = orig_rev.split()
            if len(splitted) > 250:
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
            status.append(orig_rev)

        revs.append(status)
    return revs


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
    #    string = re.sub(r"[a-zA-Z]{4,}", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def load_bert_vec(processed_chunks):
    bc = BertClient()
    embeddings = []

    for rev_idx, rev in enumerate(processed_chunks):
        rev_splitted = [orig_rev.split() for orig_rev in rev]
        result = bc.encode(rev_splitted, is_tokenized=True)
        embeddings.append(result)
    return embeddings


def predict_with_model(embeddings, l=0):
    embeddings4layers = []
    for embedding in embeddings:
        embeddings4layers.append(np.mean(embedding[:, -768 * 4:], axis=0))
    results = []
    for y in range(0, 5):
        model_file_name = 'svm_model_y'+str(y)+'.joblib'
        if os.path.isfile(model_file_name):
            classifier = load(model_file_name)
        else:
             classifier, _, _, _ = svm.classify("essays_mairesse_sb_tokenized_200_max_rev_vector.p", y, -1, 0, add_mairesse=False)
             dump(classifier, model_file_name)
        predicts = classifier.predict(embeddings4layers)
        mean = np.mean(predicts)
        if mean != 0.5:
            results.append(np.round(np.mean(mean)))
        else:
            results.append(classifier.predict([np.mean(embeddings4layers, axis=0)])[0])

    return results


def predict(text):
    sentences = shrink_text(text)
    chunks = build_new_chunks(sentences)
    processed_chunks = preocess_chunks(chunks)
    embeddings = load_bert_vec(processed_chunks)
    results = predict_with_model(embeddings)
    return results


if __name__ == "__main__":
    x = input("Enter a new text:")
    predictions = predict(x)
    print("The prediction for EXT, NEU, AGR, CON, OPN : ", predictions)
