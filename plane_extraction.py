import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import matplotlib.cm as cm
import copy
matplotlib.use("TkAgg")


cb_dim = (6,9)
square_size = 25


def draw_campose(RT_all):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # image
    # c_x = [+square_size * 3.5]
    # c_y = [+square_size * 2]
    # c_z = [0]
    # depth = 0
    # sq_x = [c_x[0] - 3.5 * square_size, c_x[0] + 3.5 * square_size, c_x[0] + 3.5 * square_size,
    #         c_x[0] - 3.5 * square_size, c_x[0] - 3.5 * square_size, c_x[0], c_x[0] + 3.5 * square_size, c_x[0],
    #         c_x[0] + 3.5 * square_size, c_x[0], c_x[0] - 3.5 * square_size]
    # sq_y = [c_y[0] - 2 * square_size, c_y[0] - 2 * square_size, c_y[0] + 2 * square_size, c_y[0] + 2 * square_size,
    #         c_y[0] - 2 * square_size, c_y[0], c_y[0] - 2 * square_size, c_y[0], c_y[0] + 2 * square_size, c_y[0],
    #         c_y[0] + 2 * square_size]
    # sq_z = [c_z[0] - depth, c_z[0] - depth, c_z[0] - depth, c_z[0] - depth, c_z[0] - depth, c_z[0], c_z[0] - depth,
    #         c_z[0], c_z[0] - depth, c_z[0], c_z[0] - depth]
    # img_pos = np.stack((sq_x, sq_y, sq_z, np.ones((11,))))

    # camera
    vsquare_size = 5
    c_x = [0]
    c_y = [0]
    c_z = [0]
    depth = -15
    cm_x = [c_x[0] - 3.5 * vsquare_size, c_x[0] + 3.5 * vsquare_size, c_x[0] + 3.5 * vsquare_size,
            c_x[0] - 3.5 * vsquare_size, c_x[0] - 3.5 * vsquare_size, c_x[0], c_x[0] + 3.5 * vsquare_size, c_x[0],
            c_x[0] + 3.5 * vsquare_size, c_x[0], c_x[0] - 3.5 * vsquare_size]
    cm_y = [c_y[0] - 2 * vsquare_size, c_y[0] - 2 * vsquare_size, c_y[0] + 2 * vsquare_size, c_y[0] + 2 * vsquare_size,
            c_y[0] - 2 * vsquare_size, c_y[0], c_y[0] - 2 * vsquare_size, c_y[0], c_y[0] + 2 * vsquare_size, c_y[0],
            c_y[0] + 2 * vsquare_size]
    cm_z = [c_z[0] - depth, c_z[0] - depth, c_z[0] - depth, c_z[0] - depth, c_z[0] - depth, c_z[0], c_z[0] - depth,
            c_z[0], c_z[0] - depth, c_z[0], c_z[0] - depth]
    cam_pos = np.stack((cm_x, cm_y, cm_z, np.ones((11,))))
    #ax.plot(sq_x, sq_y, sq_z)



    objp = np.zeros((1, cb_dim[0] * cb_dim[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:cb_dim[0], 0:cb_dim[1]].T.reshape(-1, 2) * square_size
    ax.plot(objp[0,:,0], objp[0,:,1], objp[0,:,2],'rx')
    img_pos = np.stack((objp[0,:,0], objp[0,:,1], objp[0,:,2], np.ones((cb_dim[0]*cb_dim[1],))))



    data = np.array(img_pos)
    data = np.hstack((data,np.array([[0],[0],[0],[0]])))

    for RT in RT_all:
        n = 11
        new_pos = []
        for i in range(11):
            pos = RT @ cam_pos[:, i]
            new_pos.append(pos)
        new_pos = np.array(new_pos).T
        tmp = np.array([[RT[0, 3]], [RT[1, 3]], [RT[2, 3]], [RT[3, 3]]])
        data = np.hstack((data, tmp))
        data = np.hstack((data, new_pos))
        ax.plot(new_pos[0, :], new_pos[1, :], -new_pos[2, :])
        ax.plot(RT[0, 3], RT[1, 3], -RT[2, 3], 'bo')



    ax.set_box_aspect(
        (np.ptp(data[0, :]), np.ptp(data[1, :]),np.ptp(data[2, :]) ))  # aspect ratio is 1:1:1 in data space


    plt.show()

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    corner = (int(corner[0]),int(corner[1]))

    corner2 = tuple(imgpts[0].ravel())
    corner2 = (int(corner2[0]),int(corner2[1]))

    corner3 = tuple(imgpts[1].ravel())
    corner3 = (int(corner3[0]),int(corner3[1]))

    corner4 = tuple(imgpts[2].ravel())
    corner4 = (int(corner4[0]),int(corner4[1]))
    img = cv2.line(img, corner, corner2, (255,0,0), 5)  #red
    img = cv2.line(img, corner, corner3, (0,255,0), 5)  #green
    img = cv2.line(img, corner, corner4, (0,0,255), 5)  #blue
    return img


def plane_extraction(img_curr,scale):


    # cb_dim = (12,13)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3) * square_size

    objp = np.zeros((1, cb_dim[0] * cb_dim[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:cb_dim[0], 0:cb_dim[1]].T.reshape(-1, 2) * square_size

    #objp = np.zeros((cb_dim[0] * cb_dim[1], 3), np.float32)
    #objp[:, :2] = np.mgrid[0:cb_dim[1], 0:cb_dim[0]].T.reshape(-1, 2) *square_size


    gray_curr = cv2.cvtColor(img_curr,cv2.COLOR_BGR2GRAY)
    #mtx = np.array([[3.29802586e+03, 0.00000000e+00, 1.00705602e+03], #scale =3
    #                [0.00000000e+00, 3.29545030e+03, 5.09038576e+02],
    #                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    # mtx = np.array([[1.98101340e+03, 0.00000000e+00, 6.03402590e+02], #scale =5
    #       [0.00000000e+00, 1.97947322e+03, 3.07321742e+02],
    #       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    mtx = np.array([[2.07923435e+03, 0.00000000e+00, 5.27413389e+02],
 [0.00000000e+00, 2.06712037e+03, 3.33685933e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])




    ret, corners = cv2.findChessboardCorners(gray_curr, cb_dim, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        corners_refined = cv2.cornerSubPix(gray_curr, corners, (11, 11), (-1, -1), criteria)
        ret,rvecs, tvecs = cv2.solvePnP(objp[0][:], corners_refined, mtx, distCoeffs= None )

        print(corners_refined.T)
        print(objp.T)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, distCoeffs=None)
        img_axis = draw(img_curr, corners_refined, imgpts)
        cv2.imshow('img', img_axis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            cv2.imwrite('test.png', img_axis)


        cv2.drawChessboardCorners(img_curr, (8,6), corners_refined, ret)
        cv2.imshow('img_corners', img_curr)
        cv2.waitKey(1)


    print(ret)
    print(rvecs)
    print(tvecs)

    R, _ = cv2.Rodrigues(rvecs)
    R = R.T
    tvecs = -R @ tvecs



    RT = np.zeros((4,4))
    RT[0:3,0:3] = R[:,:]
    RT[0,3] = tvecs[0]
    RT[1,3] = tvecs[1]
    RT[2,3] = tvecs[2]
    RT[3,3] = 1

    print(RT)
    return RT


RT_all = []
scale = 5
#img_curr = cv2.imread('.\plane_image\center.jpg')
#img_curr = cv2.resize(img_curr, (int(img_curr.shape[1] / scale), int(img_curr.shape[0] / scale)), interpolation=cv2.INTER_AREA)
#RT_all.append(plane_extraction(img_curr,scale))



# img_curr = cv2.imread('.\plane_image_2\LL.jpg')
# img_curr = cv2.resize(img_curr, (int(img_curr.shape[1] / scale), int(img_curr.shape[0] / scale)), interpolation=cv2.INTER_AREA)
# RT_all.append(plane_extraction(img_curr,scale))
#
# img_curr = cv2.imread('.\plane_image_2\LR.jpg')
# img_curr = cv2.resize(img_curr, (int(img_curr.shape[1] / scale), int(img_curr.shape[0] / scale)), interpolation=cv2.INTER_AREA)
# RT_all.append(plane_extraction(img_curr,scale))
#
# img_curr = cv2.imread('.\plane_image_2\/UR.jpg')
# img_curr = cv2.resize(img_curr, (int(img_curr.shape[1] / scale), int(img_curr.shape[0] / scale)), interpolation=cv2.INTER_AREA)
# RT_all.append(plane_extraction(img_curr,scale))
#
# img_curr = cv2.imread('.\plane_image_2\/UL.jpg')
# img_curr = cv2.resize(img_curr, (int(img_curr.shape[1] / scale), int(img_curr.shape[0] / scale)), interpolation=cv2.INTER_AREA)
# RT_all.append(plane_extraction(img_curr,scale))


img_curr = cv2.imread('.\plane_image_2\/1.jpg')
img_curr = cv2.resize(img_curr, (int(img_curr.shape[1] / scale), int(img_curr.shape[0] / scale)), interpolation=cv2.INTER_AREA)
RT_all.append(plane_extraction(img_curr,scale))

img_curr = cv2.imread('.\plane_image_2\/2.jpg')
img_curr = cv2.resize(img_curr, (int(img_curr.shape[1] / scale), int(img_curr.shape[0] / scale)), interpolation=cv2.INTER_AREA)
RT_all.append(plane_extraction(img_curr,scale))

img_curr = cv2.imread('.\plane_image_2\/3.jpg')
img_curr = cv2.resize(img_curr, (int(img_curr.shape[1] / scale), int(img_curr.shape[0] / scale)), interpolation=cv2.INTER_AREA)
RT_all.append(plane_extraction(img_curr,scale))


draw_campose(RT_all)
