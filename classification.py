import pandas as pd
import numpy as np
import numpy.matlib
import math
import time

import scipy.io
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_alg
from scipy.sparse import csgraph
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
import sys 
from scipy.sparse import csr_matrix

COLUMNS = ["user_id", "timestamp", "longitude", "latitude", "video_id"]


def preprocessing(file_name):
    data = pd.read_csv(file_name, sep=',', names=COLUMNS)
    data_size = len(data)
    #uid_to_index = dict([val,key] for key,val in data["user_id"].to_dict().items())
    #print(uid_to_index)
    #return
    uid_to_index = {}
    index_to_uid = []
    vid_to_index = {}
    index_to_vid = []
    
    for i in range(0, data_size):
        if data["user_id"][i] not in uid_to_index:
            index_to_uid.append(data["user_id"][i])
            uid_index = len(index_to_uid) - 1
            uid_to_index[data["user_id"][i]] = uid_index
        if data["video_id"][i] not in vid_to_index:
            index_to_vid.append(data["video_id"][i])
            vid_index = len(index_to_vid) - 1
            vid_to_index[data["video_id"][i]] = vid_index

    user_num = len(index_to_uid)
    video_num = len(index_to_vid)
    
    value = []
    row = []
    col = []

    flag = {}
    for i in range(0, data_size):
        uid_index = uid_to_index[data["user_id"][i]]
        vid_index = vid_to_index[data["video_id"][i]]
        h = vid_index * user_num + uid_index
        if h not in flag:
            flag[h] = 1
        else:
            continue
        row.append(vid_index)
        col.append(uid_index)
        value.append(1)

    x = csr_matrix((value, (row, col)), shape=(video_num, user_num))

    return x, user_num, video_num, uid_to_index, index_to_uid, vid_to_index, index_to_vid

'''
def cal_sim(x):
    rows, cols = x.shape
    sp = sparse.csr_matrix(x)
    sp_t = sparse.csr_matrix(x.transpose())
    sim = sp * sp_t.toarray()
    linalg_norm = np.zeros([rows])
    for i in range(0, rows):
        linalg_norm[i] = np.linalg.norm(x[i])

    inalg_norm_xy = linalg_norm.reshape([-1, 1]).dot(linalg_norm.reshape([1, -1]))
    sim = numpy.ones(sim.shape) - sim / inalg_norm_xy
    print(sim)
''' 



if __name__ == "__main__":
    start_time = time.time()
    x, user_num, video_num, uid_to_index, index_to_uid, vid_to_index, index_to_vid = preprocessing(sys.argv[1])
    preprocess_time = time.time() - start_time
    print("preprocess finished, time %.1f" % (preprocess_time))
    clustering = SpectralClustering(n_clusters=8, eigen_solver='amg', gamma=1.0, affinity='rbf', assign_labels='kmeans').fit(x)
    cluster_time = time.time() - start_time
    print("SpectralClustering finished, time %.1f" % (cluster_time))
    with open("./result", 'w') as fp:
        fp.write("video_id\ttype\n")
        for i, j in enumerate(clustering.labels_):
            fp.write("%d\t%d\n" % (index_to_vid[i], j))
