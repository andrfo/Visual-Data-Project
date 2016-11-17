import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import collections
import math
import random


def greyscale(img):
    return ((img[:, :, 0]*0.2126 + img[:, :, 1]*0.7152 + img[:, :, 2]*0.0722))


image = misc.imread("./images/difficult01.png")
laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
pixelsInRegion = np.zeros_like(image)

def getNeighbours(point):
    r = []
    r.append([point[0] - 1, point[1]]) #over
    r.append([point[0] + 1, point[1]]) #under
    r.append([point[0], point[1] - 1]) #left
    r.append([point[0], point[1] + 1]) #right

    return r
def difference(p1, p2):
    diff = abs(int(p1[0]) - int(p2[0])) + abs(int(p1[1]) - int(p2[1])) + abs(int(p1[2]) - int(p2[2]))
    return diff

def getEdges(img, T):
    newImage = np.zeros_like(image)
    for x in range(len(img[0])):
        for y in range(len(img)):
            for p in getNeighbours([y, x]):
                try:
                    if difference(img[y][x], img[p[0]][p[1]]) < T:
                        newImage[y][x] = [255, 255, 255]
                except:
                    pass
    return newImage


#Expands a region from one seed using kind of DFS
def makeRegion(img, seed, T):
    region = [] #Candidates that have been appended to the retgion
    candidates = collections.deque()#Candidates to be considered
    candidates.append(seed)
    seedIntensity = img[seed[0], seed[1]]

    while(len(candidates) > 0):
        current = candidates.popleft()
        I = img[current[0], current[1]]
        if(difference(seedIntensity, I) > T):
            for p in getNeighbours(current):
                if p not in region and p not in candidates:
                    candidates.append(p)
            region.append(current)
    return region

#applies the grown regions on an empty image
def makeImage(regions):
    img = np.zeros_like(image)
    colors = []
    for r in regions:
        colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    for list in regions:
        color = colors.pop()
        for point in list:
            img[point[0], point[1]] = color
    return img

#makes regions from a set of seeds
def makeRegions(startSeed, T):
    regions = []
    regions.append(makeRegion(image, startSeed, T))


def makeSeeds(img, T):
    pass




print(len(image))
print(len(image[0]))


threshold = 20
#s = makeSeeds(image, threshold)
#print(len(s))

#RGImage = makeImage(makeRegions(s, threshold))




plt.subplot(121)
plt.imshow(image, cmap=plt.cm.gray)
plt.title("org image")

plt.subplot(122)
plt.imshow(getEdges(image, 20), cmap=plt.cm.gray)
plt.title("RG image")

plt.show()












