from procgen import ProcgenGym3Env
import matplotlib.pyplot as plt
import numpy as np
import cv2

env = ProcgenGym3Env(num=1, env_name="maze", start_level=0, num_levels=1,use_backgrounds=False,use_monochrome_assets=True)
states = env.observe()
img = states[1]["rgb"][0]
R = []
G = []
B = []
coords = []
mouseSize = 0


for y,row in enumerate(img):
    r = []
    g = []
    b = []
    min=64
    max=0
    for x,rgb in enumerate(row):
        r.append(rgb[0])
        g.append(rgb[1])
        b.append(rgb[2])
 
        if rgb[0]== 127 and rgb[1]==255 and rgb[2]==127:
            mouseSize += 1

        if rgb[0]==0 and rgb[1]==0 and rgb[2]==0:
            if x < min:min = x
            if x > max:max = x
    if min!=46 and max!=0:
        coords.append([[min,y],[max,y]])

    R.append(r)
    G.append(g)
    B.append(b)
  

plt.imshow(img)
plt.savefig("maze4_2.png")
plt.cla()

img = np.dstack((R,G,B))
img[np.all(img == (0, 0, 0), axis=-1)] = (0,0,0)
img[np.all(img == (191, 127, 63), axis=-1)] = (255,255,255)
coords = np.array(coords)
minx,miny = np.min(coords[:,0]),np.min(coords[:,1])
maxx,maxy =np.max(coords[:,0]),np.max(coords[:,1])
cropped = img[minx:maxx,miny:maxy]

steps=[2,3]
i=0
prev=0
j=1
Xs = []
while i < cropped.shape[0]:
    i += steps[j%2]
    j+=1
    Xs.append(cropped[prev:i,:,:])
    prev = i
    #cv2.line(cropped,(i,0),(i,cropped.shape[0]),(255,0,0),1)
newImg= np.zeros((len(Xs),len(Xs),3),np.uint16)
matrix = []
for x,xs in enumerate(Xs):
    i=0
    j=1
    prev=0
    r = []
    while i < cropped.shape[1]:
        newImg[x][j-1][0] = xs[1][i+1][0]
        newImg[x][j-1][1] = xs[1][i+1][1]
        newImg[x][j-1][2] = xs[1][i+1][2]
        if(xs[1][i+1][0]==0 and xs[1][i+1][1]==0 and xs[1][i+1][2]==0):r.append('w')
        elif(xs[1][i+1][0]==255 and xs[1][i+1][1]==255 and xs[1][i+1][2]==255):r.append('-')
        elif(xs[1][i+1][0]==127 and xs[1][i+1][1]==255 and xs[1][i+1][2]==127):r.append('s')
        else:r.append('e')

        i += steps[j%2]
        if i>cropped.shape[1]:i-1
        j+=1
        prev = i
    matrix.append(r)   
    #cv2.line(cropped,(0,i),(cropped.shape[1],i),(255,0,0),1)
print(matrix)
plt.imshow(newImg)
plt.savefig("maze4.png")
plt.cla()
plt.imshow(cropped)
plt.savefig("maze4_1.png")
