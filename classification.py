import argparse
import csv
from itertools import islice
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score


def feature_extraction_all(feature_csv):
    features = []

    with open(feature_csv, 'r') as f:
        data = csv.reader(f)
        for line in islice(data, 0, None):
            if line[0] != 'file1':
                feature = [float(i) for i in line[2:]]
                features.append(feature)
    print('length:')
    print(len(features))
    return features


def obtain_dataset(dir_path):
    if dir_path[-1] == '/':
        clone_featureCSV = dir_path + 'BCB_clone_4_dis.csv'
        nonclone_featureCSV = dir_path + 'BCB_nonclone_4_dis.csv'

    else:
        clone_featureCSV = dir_path + '/GCJ_clone_4_dis.csv'
        nonclone_featureCSV = dir_path + '/GCJ_nonclone_4_dis.csv'

    Vectors = []
    Labels = []
    clone_features = feature_extraction_all(clone_featureCSV)
    nonclone_features = feature_extraction_all(nonclone_featureCSV)

    Vectors.extend(clone_features)
    Labels.extend([1 for i in range(len(clone_features))])
    Vectors.extend(nonclone_features)
    Labels.extend([0 for i in range(len(nonclone_features))])
    print('len of Vectors:')
    print(len(Vectors))
    print('len of Labels:')
    print(len(Labels))
    return Vectors, Labels


def random_features(vectors, labels):
    Vec_Lab = []

    for i in range(len(vectors)):
        vec = vectors[i]
        lab = labels[i]
        vec.append(lab)
        Vec_Lab.append(vec)

    random.shuffle(Vec_Lab)

    return [m[:-1] for m in Vec_Lab], [m[-1] for m in Vec_Lab]


from sklearn.ensemble import RandomForestClassifier
def randomforest(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    F1s = []
    Precisions = []
    Recalls = []

    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = RandomForestClassifier(max_depth=64, random_state=0)
        clf.fit(train_X, train_Y)

        #joblib.dump(clf, 'clf_randomforest.pkl')
        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)

    print(np.mean(F1s), np.mean(Precisions), np.mean(Recalls))
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls)]


from sklearn.neighbors import KNeighborsClassifier
def knn_1(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)

    F1s = []
    Precisions = []
    Recalls = []

    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(train_X, train_Y)

        #joblib.dump(clf, 'clf_knn_1.pkl')
        y_pred = clf.predict(test_X)


        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)

    print(np.mean(F1s), np.mean(Precisions), np.mean(Recalls))
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls)]


def knn_3(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)

    F1s = []
    Precisions = []
    Recalls = []

    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(train_X, train_Y)

        #joblib.dump(clf, 'clf_knn_3.pkl')
        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)

    print(np.mean(F1s), np.mean(Precisions), np.mean(Recalls))
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls)]


from sklearn import tree
def decision_tree(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)

    F1s = []
    Precisions = []
    Recalls = []

    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = tree.DecisionTreeClassifier()
        clf.fit(train_X, train_Y)

        # save model
        #joblib.dump(clf, 'clf_decision_tree.pkl')
        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)

    print(np.mean(F1s), np.mean(Precisions), np.mean(Recalls))
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls)]


def main():
    dir_path = './'  # BCB
    # dir_path = '.'  # GCJ
    Vectors, Labels = obtain_dataset(dir_path)
    vectors, labels = random_features(Vectors, Labels)
    print('randomforest')
    randomforest(vectors, labels)
    # print('decisiontree')
    # decision_tree(vectors, labels)
    # print('knn_1')
    # knn_1(vectors, labels)
    # print('knn_3')
    # knn_3(vectors, labels)


if __name__ == '__main__':
    main()
