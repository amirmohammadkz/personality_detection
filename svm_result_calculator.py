import pandas as pd
import svm
import numpy as np


def label_converter(row):
    if row == 'y':
        return 1
    else:
        return 0


def truth_determiner(column1, column2, equal=True):
    if equal:
        if column1 == column2:
            return 1
        else:
            return 0
    else:
        if column1 != column2:
            return 1
        else:
            return 0


def label_assigner(column1, column2):
    if column1 >= column2:
        return 0
    else:
        return 1


def add_max_200_svm(path, l=0):
    classifier, X_test, y_test, _ = svm.classify(path, y, cv, l)
    test_df["svm_predict"] = classifier.predict(X_test)
    test_df["vector"] = X_test
    test_df["is_svm_true"] = test_df.apply(lambda row: truth_determiner(row["y" + str(y)], row["svm_predict"]),
                                           axis=1)
    determined = test_df.groupby("#AUTHID").mean()
    determined["vector"] = test_df.groupby("#AUTHID")["vector"].apply(np.mean)
    dont_know = determined[determined["is_svm_true"] == 0.5].copy()
    test = dont_know["vector"].to_list()
    if not dont_know.empty:
        dont_know["svm_predict"] = classifier.predict(test)
        dont_know["is_svm_true"] = dont_know.apply(
            lambda row: truth_determiner(row["y" + str(y)], row["svm_predict"]), axis=1)
        acc2 = len(dont_know[dont_know["is_svm_true"] == 1])
    else:
        acc2 = 0
    acc1 = len(determined[determined["is_svm_true"] > 0.5])

    acc = (acc2 + acc1) / len(determined)
    return acc1, acc2, acc


def add_max_200_svm_probability(path, l=0):
    classifier, X_test, y_test, _ = svm.classify(path, y, cv, l)
    x = classifier.predict_proba(X_test)
    test_df["svm_predict0"] = x[:, 0]
    test_df["svm_predict1"] = x[:, 1]
    determined = test_df.groupby("#AUTHID").mean()
    determined["predicted_label"] = determined.apply(
        lambda row: label_assigner(row["svm_predict0"], row["svm_predict1"]), axis=1)
    determined["is_svm_true"] = determined.apply(
        lambda row: truth_determiner(row["y" + str(y)], row["predicted_label"]), axis=1)
    acc1 = len(determined[determined["is_svm_true"] == 1])
    acc = acc1 / len(determined)
    return acc


def process_raw_data(path):
    df = pd.read_csv(path)
    df["y0"] = df.apply(lambda row: label_converter(row["cEXT"]), axis=1)
    df["y1"] = df.apply(lambda row: label_converter(row["cNEU"]), axis=1)
    df["y2"] = df.apply(lambda row: label_converter(row["cAGR"]), axis=1)
    df["y3"] = df.apply(lambda row: label_converter(row["cCON"]), axis=1)
    df["y4"] = df.apply(lambda row: label_converter(row["cOPN"]), axis=1)
    return df


if __name__ == "__main__":
    whole_df = process_raw_data("essays_200_max_split.csv")
    l = 0
    f = open("svm_" + str(l) + "_layer_with_mairesse_rev200_.csv", "w")
    min_accs = []
    vote_accs = []
    doc_accs = []
    max_accs = []
    for y in range(0, 5):
        accs = []
        min_accs.append([])
        vote_accs.append([])
        doc_accs.append([])
        max_accs.append([])
        print("Y" + str(y) + ", L= " + str(l))
        for cv in range(0, 10):
            test_df = whole_df[whole_df["split"] == cv].copy()
            acc1, acc2, acc = add_max_200_svm("essays_mairesse_sb_tokenized_200_max_rev_vector.p", l)
            f.write(str(acc) + ",")
            accs.append(acc)
            print(acc)
        print("###")
        f.write(str(sum(accs) / len(accs)) + "\n")
        print(str(sum(accs) / len(accs)))
    f.close()
