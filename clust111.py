# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>, Brian Cheung
# License: BSD 3 clause

import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cv2

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

from sklearn.cluster import MiniBatchKMeans, KMeans

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

from pickle import load, dump

from wriply import write_plyTri

def loadPoClo():

# load the raccoon face as a numpy array
#try:  # SciPy >= 0.16 have face in misc
#    from scipy.misc import face
#    face = face(gray=True)
#except ImportError:
#    face = sp.face(gray=True)
#print('load image...')
#face = cv2.pyrDown(cv2.imread ('stul/disp/1d_2208_2b3(2).jpg'))
#face = cv2.imread ('stul/disp/1d_2208_2b3(2).jpg')
    try:
        pointsC = load(open("pointsC.p11","rb"))
        print('PointC loaded from pickle')
    except FileNotFoundError:
        print('load data from file...')
        points2 = np.loadtxt("out3", delimiter=" ")
        pointsC = points2[::,0:6]
        dump(pointsC,open("pointsC.p","wb"))
    return pointsC
#!print('invert X Y...')
#exit(0)
#max2 = pointsC[:,0].max()
#pointsC[:,0]= max2 - pointsC[:,0]
#max2 = pointsC[:,1:2].max()
#pointsC[:,1:2]= max2 - pointsC[:,1:2]
#pointsC = points2[:,0:2]
#print(pointsC[0])
#exit(0)
# Resize it to 10% of the original size to speed up the processing
#print('resize image...')
#face = sp.misc.imresize(face, 0.10) / 255.
#cv2.imshow('face', face)
#cv2.waitKey()

def scalePoClo(in_pointsC):
    print('scaller...')
    out_X = in_pointsC#StandardScaler().fit_transform(pointsC)#[:100000])
    maxX=out_X[::,0:3].max()
    # print(out_X[::,0:3].max())
    # print(out_X[100])
    out_X[::,0] = out_X[::,0]/maxX
    out_X[::,1] = out_X[::,1]/maxX
    out_X[::,2] = out_X[::,2]/maxX

    #maxX=out_X[::,3].max()
    #out_X[::,3] = out_X[::,3]/maxX
    # print(out_X[100])
    # print(out_X[::,0:3].max())
    return out_X

def clustDBSCAN(in_X):
    try:
        out_db = load(open("dbs.pickle1","rb"))
        print('DB loaded from pickle')
    except FileNotFoundError:
        print('DBSCAN...')
        t0 = time.time()
        #XR=np.zeros(X[:,3].shape)
        #XR = (X[:,3]/30) + (X[:,4]/30) + (X[:,5]/30)
        #XR[::1] = 1
        #print(X[:,1].min(), X[:,1].max())
        #exit(0)
        #cnt = 0
        #minXy = X[:,1].min()
        #for i in range(0,X.shape[0]):
        #    if (X[i,1] < (minXy + 0.27)):
        #        XR[i] = 0
        #        cnt += 1
        #print(cnt)
        out_db = DBSCAN(eps=0.01, min_samples=100).fit(in_X[::,0:3])
        t_dbs = time.time() - t0
        print("Time: %.2f" % t_dbs)
        dump(out_db,open("dbs.pickle","wb"))
    return out_db

def plotPoClo(X):
    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:,0:1],X[:,1:2],X[:,2:3],c=X[:,3])
    pyplot.show()
    #print(labels.shape)
    #exit(0)

def remNoClust(X,labels):
    try:
        X2 = load(open("X2.pickle1","rb"))
        print('X2 loaded from pickle')
    except FileNotFoundError:
        print('start X2...')
        t0 = time.time()
        #X2=np.array([0,0,0,0,0,0])
        Xt=np.zeros(X.shape)#X.copy()
        cnt = 0
        for i in range(0,labels.shape[0]):
            if labels[i]==-1:
                X[i]=0#[0,0,0]
            else:
                #if labels[i] in range(10,50):
                #X2 = np.vstack((X2,np.array([X[i,0],X[i,1],X[i,2],labels[i],200,100])))
                Xt[cnt] = X[i]
                #Xt[cnt,3] = labels[i]
                Xt[cnt,3] = X[i,3]
                Xt[cnt,4] = X[i,4]#200
                Xt[cnt,5] = X[i,5]#100
                cnt += 1
        X2 = Xt[:cnt]
        t_dbs = time.time() - t0
        dump(X2,open("X2.pickle","wb"))
        print("end X2   Time: %.2f" % t_dbs)
    X3 = np.array([0,0,0,0,0,0])
    X3 = X2
#X2 = X2[::30,::]
#for i in range(0,X2.shape[0]):
    #if X2[i,3] in range(30,31):#int(X2[::,3].max())):
    #    X3 = np.vstack((X3,np.array(X2[i])))
#    X3 = np.vstack((X3,np.array(X2[i])))
    X3[::,0:3] = X3[::,0:3]*100
    return X3

#!exit(0)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f"
#      % metrics.adjusted_rand_score(labels_true, labels))
#print("Adjusted Mutual Information: %0.3f"
#      % metrics.adjusted_mutual_info_score(labels_true, labels))
#print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
# import matplotlib.pyplot as plt
#
# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
#
#     class_member_mask = (labels == k)
#
#     xy = X[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=5)
#
#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=1)
#
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()
# exit(0)
# #----------------------------------------------------------
# print('graph image...')
# graph = image.img_to_graph(face)
#
# # Take a decreasing function of the gradient: an exponential
# # The smaller beta is, the more independent the segmentation is of the
# # actual image. For beta=1, the segmentation is close to a voronoi
# beta = 5
# eps = 1e-6
# print('exp image...')
# graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps
#
# #X=face
# print(face.shape)
# exit(0)
# # k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
# # t0 = time.time()
# # k_means.fit(X)
# # t_batch = time.time() - t0
# #
# # mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size,
# #                       n_init=10, max_no_improvement=10, verbose=0)
# # t0 = time.time()
# # mbk.fit(X)
# # t_mini_batch = time.time() - t0
#
# # Apply spectral clustering (this step goes much faster if you have pyamg
# # installed)
# N_REGIONS = 10
#
# #Visualize the resulting regions
# print('start clustering image...')
# for assign_labels in ('discretize', 'kmeans'):#, 'discretize'):
#     t0 = time.time()
#     print('spectral clustering graph...')
#     labels = spectral_clustering(graph, n_clusters=N_REGIONS,
#                                  assign_labels=assign_labels, random_state=1)
#     t1 = time.time()
#     labels = labels.reshape(face.shape)
#
#     plt.figure(figsize=(5, 5))
#     plt.imshow(face, cmap=plt.cm.gray)
#     for l in range(N_REGIONS):
#         plt.contour(labels == l, contours=1,
#                     colors=[plt.cm.spectral(l / float(N_REGIONS))])
#     plt.xticks(())
#     plt.yticks(())
#     title = 'Spectral clustering: %s, %.2fs' % (assign_labels, (t1 - t0))
#     print(title)
#     plt.title(title)
# plt.show()
#
if __name__ == '__main__':
    #print("Begin")
    PointC = loadPoClo()
    X=scalePoClo(PointC)
#    X[:,3] = X[:,3]*2
    db=clustDBSCAN(X)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    X3 = remNoClust(X,labels)
    #plotPoClo(X3)

    write_plyTri('out114.ply', X3)
    print('%s saved' % 'out114.ply')
#else :
#    print("name=", __name__)
