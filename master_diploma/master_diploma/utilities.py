import cv2
import numpy as np
import os
import re
import yaml

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat


def testPointsFilter (GTF, points1, points2):
    points1 = [np.matrix (list(point) + [1]) for point in points1]
    points2 = [np.matrix (list(point) + [1]) for point in points2]

    goodPoints1 = []
    goodPoints2 = []

    for (point1, point2) in zip (points1, points2):
        error = abs ((point1 * GTF * point2.transpose()).max())
        if (error <= 1):
            goodPoints1.append ([point1[0,0],point1[0,1]])
            goodPoints2.append ([point2[0,0],point2[0,1]])

    return (goodPoints1, goodPoints2)


def readImagesFromFolder (imagesFolder, primarySide = 'left'):
    images = []
    files = sorted (os.listdir (imagesFolder))
    num = re.compile ('\.\d*\.')
    side = re.compile ('left|right')
    last = 0
    otherSide = 'right' if (primarySide == 'left') else 'left'

    for file in files:
        if (side.search (file).group(0) != primarySide):
            continue
        current = int (num.search (file).group(0) [1:-1])

        if (last != current - 1):
            images.append ([])
        last = current
        image1 = cv2.imread (imagesFolder + '/' + file, 0)

        file2 = side.split (file)
        file2 = file2[0] + otherSide + file2[1]
        image2 = cv2.imread (imagesFolder + '/' + file2, 0)

        images[-1].append ((image1, image2))
        pass
    return images