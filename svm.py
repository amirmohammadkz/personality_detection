import time

import joblib
import numpy as np
from sklearn import svm
from sklearn.ensemble import BaggingClassifier


def read_bert_mode(addr, attr, fold, l, add_mairesse):
    revs, vocab, mairesse = joblib.load(addr)

    for rev in revs:
        rev["split"] = int(rev["split"])
        if l != 0:
            rev["embedding"] = np.mean(rev["embedding"][:, (l - 1) * 768:l * 768], axis=0)
        else:
            rev["embedding"] = np.mean(rev["embedding"][:, -768 * 4:], axis=0)
    if add_mairesse:
        X_train = [np.concatenate([rev["embedding"], mairesse[rev["user"]]]) for rev in revs if int(rev["split"]) != fold]
        X_test = [np.concatenate([rev["embedding"], mairesse[rev["user"]]]) for rev in revs if int(rev["split"]) == fold]
    else:
        X_train = [rev["embedding"] for rev in revs if int(rev["split"]) != fold]
        X_test = [rev["embedding"] for rev in revs if int(rev["split"]) == fold]
    y_train = [rev["y" + str(attr)] for rev in revs if rev["split"] != fold]
    y_test = [rev["y" + str(attr)] for rev in revs if rev["split"] == fold]
    y_test_names = [rev["user"] for rev in revs if rev["split"] == fold]

    return X_train, X_test, y_train, y_test, y_test_names


def classify(path, y, cv, layer=0, add_mairesse=True):
    print("Classifying...")
    start = time.time()
    X_train, X_test, y_train, y_test, y_test_names = read_bert_mode(path, y, cv, layer, add_mairesse)

    classifier = BaggingClassifier(svm.SVC(gamma="scale"), n_jobs=-1)  # this line is for BB-SVM
    # classifier = svm.SVC(gamma="scale")  # this line is for BB-SVM without bagging
    classifier.fit(X_train, y_train)
    end = time.time()
    print("Bagging SVC", end - start)
    return classifier, X_test, y_test, y_test_names
