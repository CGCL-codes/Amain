import numpy as np
import csv
import os
import time
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from itertools import islice


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)


def get_distance(ori, out):
    existnpy = []
    listdir('./npy', existnpy)
    j = 0
    Cos = []
    E = []
    M = []
    C = []
    exc = []

    reader = csv.reader(open(ori, 'r'))

    for r in reader:
        f1 = r[0].split('.java')[0]
        f2 = r[1].split('.java')[0]
        file1 = './npy/' + f1 + '.npy'
        file2 = './npy/' + f2 + '.npy'

        if file1 in existnpy and file2 in existnpy:
            matrix1 = np.load(file1)
            matrix2 = np.load(file2)
            cos = cosine_similarity(matrix1, matrix2)
            euc = pairwise_distances(matrix1, matrix2)
            man = pairwise_distances(matrix1, matrix2, metric='manhattan')
            che = pairwise_distances(matrix1, matrix2, metric='chebyshev')
            cosine = []
            euclidean = []
            manhattan = []
            chebyshev = []
            for i in range(len(cos[0])):
                cosine.append(1-cos[i][i])
                euclidean.append(euc[i][i])
                manhattan.append(man[i][i])
                chebyshev.append(che[i][i])
            data = [f1, f2]
            data.extend(cosine)
            data.extend(euclidean)
            data.extend(manhattan)
            data.extend(chebyshev)

            co = [f1, f2]
            co.extend(cosine)

            e = [f1, f2]
            e.extend(euclidean)

            m = [f1, f2]
            m.extend(manhattan)

            ch = [f1, f2]
            ch.extend(chebyshev)

            print(j)
            j += 1

            exc.append(data)
            Cos.append(co)
            E.append(e)
            M.append(m)
            C.append(ch)

    print(len(exc[0]))

    with open(out + '_4_dis.csv', 'w', newline='') as csvfile0:
        writer = csv.writer(csvfile0)
        for row in exc:
            writer.writerow(row)

    # with open(out + '_cos_dis.csv', 'w', newline='') as csvfile1:
    #     writer = csv.writer(csvfile1)
    #     for row in Cos:
    #         writer.writerow(row)
    # with open(out + '_euc_dis.csv', 'w', newline='') as csvfile2:
    #     writer = csv.writer(csvfile2)
    #     for row in E:
    #         writer.writerow(row)
    # with open(out + '_man_dis.csv', 'w', newline='') as csvfile3:
    #     writer = csv.writer(csvfile3)
    #     for row in M:
    #         writer.writerow(row)
    # with open(out + '_che_dis.csv', 'w', newline='') as csvfile4:
    #     writer = csv.writer(csvfile4)
    #     for row in C:
    #         writer.writerow(row)


if __name__ == '__main__':

    get_distance('./GCJ_clone.csv', 'GCJ_clone')
    get_distance('./BCB_clone.csv', 'BCB_clone')
    get_distance('./GCJ_nonclone.csv', 'GCJ_nonclone')
    get_distance('./BCB_nonclone.csv', 'BCB_nonclone')



