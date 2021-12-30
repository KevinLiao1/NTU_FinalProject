import numpy as np
import cv2 as cv
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import matplotlib.cm as cm

import copy
matplotlib.use("TkAgg")


class trajectory:
    list = []
    def __init__(self,start):
        self.start = start
        list.append(start)
    def add(self,next):
        list.append(next)


def find_direction_index(i,j,dir):
# 0 1 2
# 7   3
# 6 5 4
    if dir == 0:
        return [i-1,j-1]
    if dir == 1:
        return [i-1,j]
    if dir == 2:
        return [i-1,j+1]
    if dir == 3:
        return [i,j+1]
    if dir == 4:
        return [i+1,j+1]
    if dir == 5:
        return [i+1,j]
    if dir == 6:
        return [i+1,j-1]
    if dir == 7:
        return [i,j-1]


def classify(patch):
    neighbours = np.zeros((8,))
    it = 0

    # 0 1 2
    # 7   3
    # 6 5 4

    cr_it = [1, 3, 5, 7]
    co_it = [0, 2, 4, 6]

    neighbours[0] = patch[0][0]
    neighbours[1] = patch[0][1]
    neighbours[2] = patch[0][2]
    neighbours[3] = patch[1][2]
    neighbours[4] = patch[2][2]
    neighbours[5] = patch[2][1]
    neighbours[6] = patch[2][0]
    neighbours[7] = patch[1][0]

    ind = np.nonzero(neighbours)
    nn = len(ind[0])
    directions = []
    if nn == 0:
        return -1,directions           # delete single pixel
    elif nn == 1:
        directions.append(ind[0])
        return 0,directions            # start/end
    else:
        if nn == 2 and abs(ind[0][0]-ind[0][1]) % 6 == 1:
            if ind[0][0] % 2 == 1:
                directions.append(ind[0][0])
            else:
                directions.append(ind[0][1])
            return 0,directions        # start/end
        elif nn == 2 and abs(ind[0][0]-ind[0][1]) % 6 != 1:
            directions.append(ind[0][0])
            directions.append(ind[0][1])
            return 1,directions        # normal
        else:
            found = False
            for ind_nn in ind[0]:
                if ind_nn % 2 == 1: #cross
                    found = True
                    directions.append(ind_nn)
                    if ind_nn + 4 % 6 in ind[0]:
                        directions.append(ind_nn + 4 % 6)
                        if ind_nn + 2 % 6 in ind[0]:
                            directions.append(ind_nn + 2 % 6)
                        if ind_nn - 2 % 6 in ind[0]:
                            directions.append(ind_nn - 2 % 6)
                    else:
                        if ind_nn + 2 % 6 in ind[0]:
                            directions.append(ind_nn + 2 % 6)
                        else:
                            if ind_nn + 3 % 6 in ind[0]:
                                directions.append(ind_nn + 3 % 6)

                        if ind_nn - 2 % 6 in ind[0]:
                            directions.append(ind_nn - 2 % 6)
                        else:
                            if ind_nn - 3 % 6 in ind[0]:
                                directions.append(ind_nn - 3 % 6)
                    break
            if found==False:
                for ind_nn in ind[0]:
                    if ind_nn % 2 == 0:  # corner
                        directions.append(ind_nn)

                        if ind_nn + 4 % 6 in ind[0]:
                            directions.append(ind_nn + 4 % 6)
                            if ind_nn + 2 % 6 in ind[0]:
                                directions.append(ind_nn + 2 % 6)
                            if ind_nn - 2 % 6 in ind[0]:
                                directions.append(ind_nn - 2 % 6)
                        else:
                            if ind_nn + 2 % 6 in ind[0]:
                                directions.append(ind_nn + 2 % 6)
                            else:
                                if ind_nn + 3 % 6 in ind[0]:
                                    directions.append(ind_nn + 3 % 6)

                            if ind_nn - 2 % 6 in ind[0]:
                                directions.append(ind_nn - 2 % 6)
                            else:
                                if ind_nn - 3 % 6 in ind[0]:
                                    directions.append(ind_nn - 3 % 6)
                        break
        if len(directions)>2:
            return 2,directions        # branch
        else:
            return 1,directions        # normal

def trajectory_improved(img):
    n = len(img)  # height
    m = len(img[0])  # width

    orig_img = np.copy(img)
    img = np.pad(img,1,mode='constant',constant_values=0)

    #search start point
    for i in range(1,n):
        for j in range(1,m):
            if img[i][j] == 255:
                sub_img = np.empty((3,3))
                sub_img = img[i-1:i+1+1,j-1:j+1+1]
                c, direction = classify(sub_img)
                if c == 0:
                    start = [i,j]
                    break

    #begin build trjectory
    tr = trajectory(start)
    next = find_direction_index(start[0],start[1],direction)
    sub_img = np.empty((3, 3))
    sub_img = img[next[0] - 1:next[0] + 1 + 1, next[1] - 1:next[1] + 1 + 1]
    c, _ = classify(sub_img)
    while next!=start or c != 0:
        tr.add(next)





