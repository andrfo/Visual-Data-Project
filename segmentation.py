import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import collections
import math
import random


image = misc.imread("./images/difficult01.png")
regionImage = np.zeros_like(image)

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
        if(difference(seedIntensity, I) < T):
            for p in getNeighbours(current):
                if p not in region and p not in candidates:
                    candidates.append(p)
            region.append(current)
    makeImage(region)

#applies the grown regions on an empty image
def makeImage(region):
    color = [random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)]
    for point in region:
        regionImage[point[0], point[1]] = color

#makes regions from a set of seeds
def makeRegions(startSeed, T):
    makeRegion(image, startSeed, T)
    complete = True
    counter = 0
    while(complete):
        try:
            for y in range(len(image)):
                for x in range(len(image[0])):
                    if (regionImage[y][x][0] == 0 and regionImage[y][x][1] == 0 and regionImage[y][x][1] == 0):
                        makeRegion(image, [x, y], T)
                        counter += 1
                        raise ValueError
            complete = False
        except ValueError as e:
            print(str(e))
            print(counter)
            pass






print(len(image))
print(len(image[0]))


threshold = 60
makeRegions([10, 10], threshold)




plt.subplot(121)
plt.imshow(image, cmap=plt.cm.gray)
plt.title("org image")

plt.subplot(122)
plt.imshow(regionImage, cmap=plt.cm.gray)
plt.title("RG image")

plt.show()












