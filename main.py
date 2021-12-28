import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def grow(img,k,j,i):
    stack = []
    list = []
    (m,n) = np.shape(img)
    result = np.zeros((m,n))
    m = m-1
    n = n-1
    #step 1
    img[k,j] = i
    result[k,j] = 255
    stack.append((k,j))
    stack.append((0,0))

    #step 2
    kl = 0
    while True:
        halo_size = 3

        start_k = max(0,k-halo_size)
        end_k = min(k+halo_size,m)+1

        start_j = max(0, j-halo_size)
        end_j = min(j+halo_size, n)+1

        for ik in range(start_k,end_k):
            if ik == k:
                continue
            for ij in range(start_j,end_j):
                if ij == j:
                    continue
                if img[ik,ij] == 1:
                    img[ik,ij] = i;
                    result[ik,ij] = 255
                    stack.append((ik,ij))
                    list.append((ik,ij))
        (k,j) = stack.pop()
        if (k,j) == (0,0):
            (k,j) = stack.pop()
            return list,result,img,stack



def edge_clustering(img):
    (m,n) = np.shape(img)
    print(n)
    print(m)
    for k in range(m):
        for j in range(n):
            if int(img[k,j]) == 255:
                img[k,j] = 1
    i = 1
    stacked_img = []
    stacked_list = []
    for k in range(m):
        for j in range(n):
            if img[k,j] == 1:
                i = i+1
                res ,img, stack = grow(img,k,j,i)
                if np.count_nonzero(res) > 5:
                    stacked_img.append(res)
    print('Clustering: COMPLETE')
    print('Found '+str(len(stacked_img))+' clusters.')
    return stacked_img


img = cv.imread('./image/1.jpg',0)
img = cv.GaussianBlur(img,(5,5),0)
img = cv.resize(img, dsize=(400, 400), interpolation=cv.INTER_CUBIC)

#img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)




edges = cv.Canny(img,100,200)
edges_cp = np.copy(edges)
clustered_edges = edge_clustering(edges_cp)

for imgs in clustered_edges:
    cv.imshow('Img',imgs)
    cv.waitKey()

comb = sum(clustered_edges)




plt.subplot(121),plt.imshow(comb,cmap = 'gray')
plt.title('Combined Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()