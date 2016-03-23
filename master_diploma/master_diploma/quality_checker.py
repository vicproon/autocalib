import numpy as np
import random as rand
import cv2
import math

def matrixNormTest (F1, F2, p = 2, q = 2):
    p, q = float(p), float(q)
    F1 = np.matrix (F1)
    F2 = np.matrix (F2)
    
    normalizationCoeff1 = 1 / F1[2,2]
    normalizationCoeff2 = 1 / F2[2,2]
        
    F1 = np.array ((normalizationCoeff1 * F1))
    F2 = np.array ((normalizationCoeff2 * F2))

    F1Norm = pow (sum ([pow (sum ([ pow (x,p) for x in F1[:,i]]), q/p) for i in range (len (F1))]), 1 / q)
    F2Norm = pow (sum ([pow (sum ([ pow (x,p) for x in F2[:,i]]), q/p) for i in range (len (F2))]), 1 / q)
    print F1Norm
    print F2Norm

    return abs (F1Norm - F2Norm)
    pass


def centerPointsTest (F1, F2):
    F1 = np.matrix (F1)
    F2 = np.matrix (F2)

    point1 = __findCenter (F1)
    point2 = __findCenter (F2)

    distancesToCenter = [np.linalg.norm (point1), np.linalg.norm (point2)]
    print distancesToCenter
    distancesBetween = np.linalg.norm (np.matrix (point1) - np.matrix(point2))

    result = distancesToCenter / np.linalg.norm (distancesToCenter)
    return abs (result[0] - result[1])
    pass


def epipolAverageDistance (F1, F2):
    testPoints = np.int32 ([(rand.randint (0, 1000), rand.randint (0, 1000)) for i in range (1000)]).reshape (-1,1,2)
    lines1 = cv2.computeCorrespondEpilines (testPoints, 1, F1).reshape (-1,3)
    lines2 = cv2.computeCorrespondEpilines (testPoints, 1, F2).reshape (-1,3)
        
    points1 = [-line[2] / line[1] for line in lines1]
    points2 = [-line[2] / line[1] for line in lines2]

    distances = [abs (point1 - point2) for (point1, point2) in zip (points1, points2)]
    average1 = np.mean (distances)
    average2 = np.mean ([abs (distance - average1) for distance in distances])
    return (average1, average2)
    pass

def __findCenter (F):
    line1 = F * (np.matrix([15,16,1])).transpose()
    line2 = F * (np.matrix([341,532,1])).transpose()

    a1,b1,c1 = line1.transpose().tolist()[0]
    a2,b2,c2 = line2.transpose().tolist()[0]
    x = - (c1*b2 - c2*b1) / (a1*b2 - a2*b1)
    y = - (a1*c2 - a2*c1) / (a1*b2 - a2*b1)
    return (x,y)
    pass