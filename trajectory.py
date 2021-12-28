import numpy as np
import cv2 as cv
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
matplotlib.use("TkAgg")

def animate(i,path):

    x_line.append(path[int(i)][1])
    y_line.append(path[int(i)][0])

    line.set_xdata(x_line)
    line.set_ydata(y_line)

    dot.set_xdata(path[int(i)][1])
    dot.set_ydata(path[int(i)][0])
    #return line,



def search_neighbors(op, brmatrix, se_points, visited_matrix):
    x = op[0]
    y = op[1]
    neighbors = []
    neighbors.append([x-1,y-1])
    neighbors.append([x-1,y])
    neighbors.append([x-1,y+1])
    neighbors.append([x,y-1])
    neighbors.append([x,y+1])
    neighbors.append([x+1,y-1])
    neighbors.append([x+1,y])
    neighbors.append([x+1,y+1])
    #print(neighbors)
    for i in range(8):
        if brmatrix[neighbors[i][0]][neighbors[i][1]] == 1:
            if visited_matrix[neighbors[i][0]][neighbors[i][1]] == 0:
                #print("found branch")
                return [neighbors[i][0],neighbors[i][1]],0

    for i in range(8):
        for j in range(len(se_points)):
            if  (neighbors[i][0] == se_points[j][0]) & (neighbors[i][1] == se_points[j][1]):
                return [neighbors[i][0],neighbors[i][1]],1
    return [-1,-1],2
    

#zhang suen thinning
def intarray(binstring):
    '''Change a 2D matrix of 01 chars into a list of lists of ints'''
    return [[1 if ch == '1' else 0 for ch in line] 
            for line in binstring.strip().split()]
 
def chararray(intmatrix):
    '''Change a 2d list of lists of 1/0 ints into lines of 1/0 chars'''
    return '\n'.join(''.join(str(p) for p in row) for row in intmatrix)
 
def toTxt(intmatrix):
    '''Change a 2d list of lists of 1/0 ints into lines of '#' and '.' chars'''
    return '\n'.join(''.join(('#' if p else '.') for p in row) for row in intmatrix)
 
def neighbours(x, y, image):
    '''Return 8-neighbours of point p1 of picture, in order'''
    i = image
    x1, y1, x_1, y_1 = x+1, y-1, x-1, y+1
    #print ((x,y))
    return [i[y1][x],  i[y1][x1],   i[y][x1],  i[y_1][x1],  # P2,P3,P4,P5
            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9
 
def transitions(neighbours):
    n = neighbours + neighbours[0:1]    # P2, ... P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))
 
def zhangSuen(image):
    changing1 = changing2 = [(-1, -1)]
    while changing1 or changing2:
        # Step 1
        changing1 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and    # (Condition 0)
                    P4 * P6 * P8 == 0 and   # Condition 4
                    P2 * P4 * P6 == 0 and   # Condition 3
                    transitions(n) == 1 and # Condition 2
                    2 <= sum(n) <= 6):      # Condition 1
                    changing1.append((x,y))
        for x, y in changing1: image[y][x] = 0
        # Step 2
        changing2 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and    # (Condition 0)
                    P2 * P6 * P8 == 0 and   # Condition 4
                    P2 * P4 * P8 == 0 and   # Condition 3
                    transitions(n) == 1 and # Condition 2
                    2 <= sum(n) <= 6):      # Condition 1
                    changing2.append((x,y))
        for x, y in changing2: image[y][x] = 0
        #print changing1
        #print changing2
    return image     



img = cv.imread('.\image\snoopy.jpg',cv.IMREAD_GRAYSCALE)
#RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.bitwise_not(img)

#canny edge detection
edges = cv.Canny(img,100,200)
#edges = cv.bitwise_not(edges)

"""#skeletonize
img = edges
size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

# Get a Cross Shaped Kernel
element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))

# Repeat steps 2-4
while True:
    #Step 2: Open the image
    opimg = cv.morphologyEx(img, cv.MORPH_OPEN, element)
    #Step 3: Substract open from the original image
    temp = cv.subtract(img, opimg)
    #Step 4: Erode the original image and refine the skeleton
    eroded = cv.erode(img, element)
    skel = cv.bitwise_or(skel,temp)
    img = eroded.copy()
    # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
    if cv.countNonZero(img)==0:
        break

# Displaying the final skeleton
#cv.imshow("Skeleton",skel)
#cv.waitKey(0)
#cv.destroyAllWindows()"""

#skel = zhangSuen(edges)
skel = edges
#cv.imshow("edges",skel)
#cv.waitKey(0)
#cv.destroyAllWindows()

#size of image
n = len(skel) #height
m = len(skel[0]) #width
#print(n,m)

#search for start/end points and branch points
#print(skel[0][0])
se_points = []
branch_points = []
brmatrix = []
#inittialize brmatrix
for i in range(n):
    for j in range(m):
        if j == 0:
            brmatrix.append([0])
        else:
            brmatrix[i].append(0)
#print(brmatrix[0][20])


for i in range(1,n-1):
    for j in range(1,m-1):
        if skel[i][j] == 255:
            #print(skel[i][j])
            #print("hi")
            neighbors = []
            neighbors.append(skel[i-1][j-1])
            neighbors.append(skel[i-1][j])
            neighbors.append(skel[i-1][j+1])
            neighbors.append(skel[i][j-1])
            neighbors.append(skel[i][j+1])
            neighbors.append(skel[i+1][j-1])
            neighbors.append(skel[i+1][j])
            neighbors.append(skel[i+1][j+1])
            amount = 0
            for k in range(8):
                if neighbors[k] == 255:
                    amount += 1
            if amount == 1:
                se_points.append([i,j])
            if amount > 1:
                branch_points.append([i,j])
                brmatrix[i][j] = 1
        


if len(se_points) == 0:
    if len(branch_points) != 0:
        se_points.append(branch_points[0])
    else:
        print("Empty picture")


#write to file to check

"""fp = open("check_startend.txt","w")
msg = ""
for i in range(len(se_points)):
    msg += str(se_points[i][0]) + " " + str(se_points[i][1]) + "\n"
fp.write(msg)
fp.close()

msg = ""
fp = open("check_branch.txt","w")

for i in range(len(branch_points)):
    msg += str(branch_points[i][0]) + " " + str(branch_points[i][1]) + "\n"
fp.write(msg)
fp.close()"""

#form trajectories
print("searching paths")
fp = open("path.txt","w")
se_points_original = copy.deepcopy(se_points)
msg = ""

drawn_matrix = []
for i in range(n):
    for j in range(m):
        if j == 0:
            drawn_matrix.append([0])
        else:
            drawn_matrix[i].append(0)



loop = 0
path_number = 0

#for animation
x_line = []
y_line = []
# Kev: animation of path
figure, ax = plt.subplots()
# Setting limits for x and y axis
ax.set_xlim(0, m)
ax.set_ylim(0, n)
ax.set_aspect('equal')
# Since plotting a single graph
line, = ax.plot(0, 0)
dot, = ax.plot(0,0,'bo')


comb_paths = []
while ((len(se_points) != 0) & (loop < 300)):
    loop += 1
    #print("start")
    visited_matrix = []
    for i in range(n):
        for j in range(m):
            if j == 0:
                visited_matrix.append([0])
            else:
                visited_matrix[i].append(0)
    path = []
    path.append(se_points[0])
    c = 0
    next_point = se_points[0]
    visited_matrix[next_point[0]][next_point[1]] = 1
    drawn_matrix[next_point[0]][next_point[1]] = 1
    drawn_matrix[next_point[0]-1][next_point[1]-1] = 1
    drawn_matrix[next_point[0]-1][next_point[1]] = 1
    drawn_matrix[next_point[0]-1][next_point[1]+1] = 1
    drawn_matrix[next_point[0]+1][next_point[1]-1] = 1
    drawn_matrix[next_point[0]+1][next_point[1]] = 1
    drawn_matrix[next_point[0]+1][next_point[1]+1] = 1
    drawn_matrix[next_point[0]][next_point[1]-1] = 1
    drawn_matrix[next_point[0]][next_point[1]+1] = 1
    while c == 0:

        # c == 0 branch point
        # c == 1 start/end point
        # c == 2 neither
        next_point, c = search_neighbors(next_point,brmatrix,se_points, visited_matrix)
        #print(next_point)
        if c==2:
            break
        drawn_matrix[next_point[0]][next_point[1]] = 1
        drawn_matrix[next_point[0]-1][next_point[1]-1] = 1
        drawn_matrix[next_point[0]-1][next_point[1]] = 1
        drawn_matrix[next_point[0]-1][next_point[1]+1] = 1
        drawn_matrix[next_point[0]+1][next_point[1]-1] = 1
        drawn_matrix[next_point[0]+1][next_point[1]] = 1
        drawn_matrix[next_point[0]+1][next_point[1]+1] = 1
        drawn_matrix[next_point[0]][next_point[1]-1] = 1
        drawn_matrix[next_point[0]][next_point[1]+1] = 1
        branch_points = [subl for subl in branch_points if ((subl[0] != next_point[0])or(subl[1] != next_point[1]))]
        #brmatrix[next_point[0]][next_point[1]] = 0
        if visited_matrix[next_point[0]][next_point[1]] == 0: 
            path.append(next_point)
            visited_matrix[next_point[0]][next_point[1]] = 1
        else:
            break
    #print(path)
    #print("done")
    if c == 1:
        branch_points = [subl for subl in branch_points if ((subl[0] != next_point[0])or(subl[1] != next_point[1]))]
   

    if len(se_points) > 2:
        #print(next_point)
        se_points.pop(0)
        try:
            se_points = [subl for subl in se_points if ((subl[0] != next_point[0])or(subl[1] != next_point[1]))]
        except ValueError:
            None
    else:
        se_points.pop(0)
        #print(se_points)
    if len(path) > 1:
        print("success")
        path_number += 1
        #print(loop)
        comb_paths.extend(path)


        for i in range(len(path)):
            #print(path[i])
            if (i == (len(path)-1)):
                msg += str(path[i][0]) + " " + str(path[i][1]) + "\n"
            else:
                msg += str(path[i][0]) + " " + str(path[i][1]) + ","
    
    for i in range(len(branch_points)):
        if  drawn_matrix[branch_points[i][0]][branch_points[i][1]] == 0:
            #print("activated")
            a = copy.deepcopy([branch_points[i][0],branch_points[i][1]])
            se_points.append([branch_points[i][0],branch_points[i][1]])
            branch_points = [subl for subl in branch_points if ((subl[0] != a[0])or(subl[1] != a[1]))]
            #print(se_points[0])
            break
    
fp.write(msg)
fp.close()

#animation

animation = FuncAnimation(figure,
                          func=animate,
                          frames=np.arange(0, len(comb_paths), 3),
                          fargs=(comb_paths,),
                          interval=0)

plt.show()

#draw a image to check

drawn_matrix = np.array(drawn_matrix)
for i in range(n):
    for j in range(m):
        if drawn_matrix[i][j] == 1:
            drawn_matrix[i][j] = 255
        else:
            drawn_matrix[i][j] = 0

print(path_number)
plt.imshow(drawn_matrix, cmap='gray', vmin=0, vmax=255)
plt.show()
#u8 = cv.convertScaleAbs(drawn_matrix)
#cv.imshow("drawn",u8)
#cv.waitKey(0)
#cv.destroyAllWindows()










