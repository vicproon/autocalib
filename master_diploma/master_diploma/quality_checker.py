import numpy as np
import random as rand
import cv2
import math
import logging as log

logger = log.getLogger ('quality_checker')
logger.setLevel (log.DEBUG)

class quality_checker (object):
    __TestData = []

    def __init__(self, GTF, imageShape):
        self.__GTF = GTF
        self.__imageShape = imageShape
        pass

    def computeTests (self, F, calcPoints1, calcPoints2, testPoints1, testPoints2, p = 2, q = 2):
        test = []
        test.append (self.__matrixNormTest (F, self.__GTF, p ,q))
        eADT = self.__epipolarsAverageDistanceTest (F, self.__GTF)
        test.append (eADT[0])
        test.append (eADT[1])
        test.append (self.__epipolPointsTest (F, self.__GTF, self.__imageShape))
        test.append (self.__averageErrorTest (self.__GTF, calcPoints1, calcPoints2))
        test.append (self.__averageErrorTest (F, testPoints1, testPoints2))
        self.__TestData.append (test)
        pass

    def printTests (self, F, calcPoints1, calcPoints2, testPoints1, testPoints2, p = 2, q = 2):
        print 'GTF: ' + str (self.__GTF)
        print 'Разница норм матриц: '.decode ('utf-8') + str (self.__matrixNormTest (F, self.__GTF, p ,q))
        printStr = 'Среднее расстояние между эпиполярными линиями Ground Truth и тестовой матрицами: {0:.3f}\n'
        printStr = printStr + 'Дисперсия расстояний между эпиполярными линиями Ground Truth и тестовой матрицами: {1:.3f}'
        print (printStr.format (*self.__epipolarsAverageDistanceTest (F, self.__GTF)).decode ('utf-8'))
        print 'Расстояние между точками пересечения эпиполярных линий Ground Truth и тестовой матрицами: '.decode ('utf-8') + str (self.__epipolPointsTest (F, self.__GTF, self.__imageShape))
        print 'Средняя ошибка соответствия ключевых точек, выбранных для вычислений: '.decode ('utf-8') + str (self.__averageErrorTest (self.__GTF, calcPoints1, calcPoints2))
        print 'Средняя ошибка вычисления x\'Fx для тестовых точек: '.decode ('utf-8') + str (self.__averageErrorTest (F, testPoints1, testPoints2))
        pass

    def __matrixNormTest (self, F1, F2, p = 2, q = 2):
        logger.info ('\nmatrixNormTest\n')
        logger.debug ('F1: ' + str (F1) + '\nF2: ' + str (F2) + '\np = ' + str (p) + '\nq = ' + str (q) + '\ntype of F1: ' + str (type (F1)) + '\ntype of F2: ' + str (type (F2)))

        p, q = float (p), float (q)
        F1 = np.matrix (F1)
        F2 = np.matrix (F2)

        logger.debug ('F1: ' + str (F1) + '\nF2: ' + str (F2))

        normalizationCoeff1 = 1 / np.linalg.norm (F1)
        normalizationCoeff2 = 1 / np.linalg.norm (F2)

        logger.debug ('normalizationCoeff1 = ' + str (normalizationCoeff1) + '\nnormalizationCoeff2 = ' + str (normalizationCoeff2))
     
        F1 = np.array ((normalizationCoeff1 * F1))
        F2 = np.array ((normalizationCoeff2 * F2))
        F = F1 - F2

        logger.debug ('F1: ' + str (F1) + '\nF2: ' + str (F2) + '\nF: ' + str (F))

        FNormLPQ = pow (sum ([pow (sum ([ pow (x,p) for x in F[:,i]]), q / p) for i in range (F.shape[1])]), 1 / q)
        FNorm2 = np.linalg.norm (F, 2)

        logger.info ('FNormLPQ = ' + str (FNormLPQ))
        logger.info ('FNorm2 = ' + str (FNorm2))

        FNorm = FNorm2
        #return abs (F1Norm - F2Norm)
        #return FNormLPQ
        return FNorm
        pass

    def __epipolPointsTest (self, F1, F2, imageShape):
        logger.info ('\nepipolPointsTest\n')
        logger.debug ('F1: ' + str (F1) + '\nF2: ' + '\ntype of F1: ' + str (type (F1)) + '\ntype of F2: ' + str (type (F2)) + '\nimageShape')
    
        F1 = np.matrix (F1)
        F2 = np.matrix (F2)
        imageShape = [x / 2 for x in imageShape]
    
        logger.debug ('F1: ' + str (F1) + '\nF2: ' + '\ntype of F1: ' + str (type (F1)) + '\ntype of F2: ' + str (type (F2)) + '\nimageShape')
    
        point1 = [a - b for a,b in zip (self.__findEpipol (F1), imageShape)]
        point2 = [a - b for a,b in zip (self.__findEpipol (F2), imageShape)]
    
        logger.debug ('findEpipol1: ' + str (self.__findEpipol (F1)) + '\nfindEpipol2: ' + str (self.__findEpipol (F2)))
        logger.debug ('point1: ' + str (point1) + '\npoint2: ' + str (point2))

        distancesToCenter = [np.linalg.norm (point1), np.linalg.norm (point2)]
        distancesBetween = np.linalg.norm (np.matrix (point1) - np.matrix (point2))
    
        logger.info ('distancesToCenter: ' + str (distancesToCenter))
        logger.info ('distanceBetween: ' + str (distancesBetween))

        result = distancesToCenter / np.linalg.norm (distancesToCenter)
        logger.debug ('result: ' + str (result))
        return abs (result[0] - result[1])
        pass


    def __epipolarsAverageDistanceTest (self, F1, F2):
        logger.info ('\nepipolarsAverageDistanceTest\n')
        logger.debug ('F1: ' + str (F1) + '\nF2: ' + '\ntype of F1: ' + str (type (F1)) + '\ntype of F2: ' + str (type (F2)))

        testPoints = np.int32 ([(rand.randint (0, 1000), rand.randint (0, 1000)) for i in range (1000)]).reshape (-1,1,2)
        lines1 = cv2.computeCorrespondEpilines (testPoints, 1, F1).reshape (-1,3)
        lines2 = cv2.computeCorrespondEpilines (testPoints, 1, F2).reshape (-1,3)
    
        logger.debug ('testPoints: ' + str (testPoints[:10]))
        logger.debug ('lines1:' + str (lines1[:10]) + '\nlines2:' + str (lines2[:10]))

        points1 = [-line[2] / line[1] for line in lines1]
        points2 = [-line[2] / line[1] for line in lines2]

        logger.debug ('points1: ' + str (points1[:10]))
        logger.debug ('points2: ' + str (points2[:10]))

        distances = [abs (point1 - point2) for (point1, point2) in zip (points1, points2)]
        logger.info ('distances: ' + str (distances[:10]))

        average1 = np.mean (distances)
        average2 = np.mean ([abs (distance - average1) for distance in distances])

        logger.debug ('average1: ' + str (average1))
        logger.debug ('average2: ' + str (average2))

        return (average1, average2)
        pass

    def __averageErrorTest (self, F, points1, points2):
        logger.info ('averageErrorTest\n')
        logger.debug ('F: ' + str (F) + '\npoints1: ' + str (points1[:10]) + '\npoints2: ' + str (points2[:10]))

        points1 = [np.matrix (list (point) + [1]) for point in points1]
        points2 = [np.matrix (list (point) + [1]) for point in points2]

        logger.debug ('points1: ' + str (points1[:10]) + '\npoints2: ' + str (points2[:10]))
        error = sum ([abs ((point1 * F * point2.transpose ()).max ()) for (point1,point2) in zip (points1, points2)]) / len (points1)
        logger.info ('error: ' + str (error))
        return error
        pass


    def __findEpipol (self, F):
        line1 = F * (np.matrix ([15,16,1])).transpose ()
        line2 = F * (np.matrix ([341,532,1])).transpose ()

        a1,b1,c1 = line1.transpose ().tolist ()[0]
        a2,b2,c2 = line2.transpose ().tolist ()[0]
        x = - (c1 * b2 - c2 * b1) / (a1 * b2 - a2 * b1)
        y = - (a1 * c2 - a2 * c1) / (a1 * b2 - a2 * b1)
        return (x,y)
        pass
