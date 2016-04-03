import cv2
import numpy as np
import scipy.sparse.linalg as linalg
import logging as log

class f_matrix_computer(object):
    """description of class"""
    def compute (self, keyPoints1, keyPoints2):
        KPChoose = self.__transformKeyPoints (keyPoints1, keyPoints2)
        KPChoose1, trMatrices = self.__normalization (KPChoose)

        #F = self.__matrixCalculation (KPChoose[:8], True)
        F = trMatrices[0].transpose() * self.__matrixCalculation (KPChoose1) * trMatrices[1]
        #F1 = self.__matrixCalculation (KPChoose)
        #print F
        
        return F.transpose()
        pass
    
    def __matrixCalculation (self, keyPoints, flag = False):
        A = None
        if (flag) :
            A = np.matrix ([ [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1] for ((x1, y1), (x2, y2)) in keyPoints], np.uint64)
        else:
            A = np.matrix ([ [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1] for ((x1, y1), (x2, y2)) in keyPoints], np.float64)
        At = A.transpose()
        #print At*A
        w,v  = np.linalg.eigh (At*A)
        print 'Собственные значения numpy: '.decode('utf-8') + str(w)
        #w,v  = linalg.eigsh (At*A.astype ('d'), which = 'SM')
        #print 'Собственные значения scipy: '.decode('utf-8') + str(w)
        #b,w,v  = cv2.eigen (At*A.astype ('d'))
        #print 'Собственные значения opencv: '.decode('utf-8') + str(w)
        eigenvectorNumber = np.argmin (np.abs (w))
        #eigenvectorNumber = np.argmin (w)
        v = v.transpose()
        print 'eigenvector: ' + str(v)
        F = v[eigenvectorNumber].tolist()[0]
        F = np.matrix ((F[:3], F[3:6], F[6:9]), np.float64)

        U,S,V = np.linalg.svd (F, True, True)
        print 'S: ' + str(S)
        S [ S == min(S)] = 0
        S = np.diag (S)
        F = U*S*V

        #print np.linalg.eigvals (F)
        return F
        pass

    def __transformKeyPoints (self, keyPoints1, keyPoints2):#, matches):
        #KPChoose = [(keyPoints1[x.queryIdx], keyPoints2[x.trainIdx]) for x in matches]
        #KPChoose = [ ((int (round (x.pt[0])), int (round (x.pt[1]))), (int (round (y.pt[0])), int (round (y.pt[1])))) for (x, y) in KPChoose]
        #KPChoose = [ x for x in set (y for y in KPChoose)]
        #print 'KPChoose: ' + str(KPChoose)
        KPChoose = zip (keyPoints1, keyPoints2)
        return KPChoose
        pass

    def __normalization (self, keyPoints):
        result = []
        trMatrices = [np.matrix (np.identity (3), np.int32) for x in range (len (keyPoints[0]))]
        for i, kPoints in zip (range (len (keyPoints[0])), zip(*keyPoints)):
            center = (round (sum (pt[0] for pt in kPoints) / len (kPoints)), round (sum (pt[1] for pt in kPoints) / len (kPoints)))
            current = [(x[0] - center[0], x[1] - center[1]) for x in kPoints]
            
            trMatrices[i].itemset ((0,2), -center[0])
            trMatrices[i].itemset ((1,2), -center[1])

            avr = (round (sum (abs (pt[0]) for pt in current) / len (current)), round (sum (abs (pt[1]) for pt in current) / len (current)))
            current = [(x[0] / avr[0], x[1] / avr[1]) for x in current]

            trMatrices[i] = np.matrix (np.diag ([1/avr[0], 1/avr[1], 1])) * trMatrices[i]
            result.append (current)
            #print current
            #print trMatrices[i]
            pass

        result = zip(*result)
        #print result
        return result, trMatrices
        pass

    pass


