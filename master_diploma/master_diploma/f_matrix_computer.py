import cv2
import numpy as np

class f_matrix_computer(object):
    """description of class"""
    def compute (self, keyPoints1, keyPoints2, mathces):
        KPChoose = self.__transformKeyPoints (keyPoints1, keyPoints2, mathces)
        KPChoose = self.__normalization (KPChoose)

        F = self.__matrixCalculation (KPChoose)
        #print F
        pass
    
    def __matrixCalculation (self, keyPoints):
        result = 0
        A = np.matrix ([ [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1] for ((x1, y1), (x2, y2)) in keyPoints], np.int64)
        print A
        At = A.transpose()
        #print A
        #print At*A
        w,v  = np.linalg.eig (At*A)
        eigenvectorNumber = w.tolist().index (min (w))
        v = v.transpose()
        #print v
        F = v[eigenvectorNumber].tolist()[0]
        F = np.matrix ((F[:3], F[3:6], F[6:9]))
        print F
        return result
        pass

    def __transformKeyPoints (self, keyPoints1, keyPoints2, mathces):
        KPChoose = [(keyPoints1[x.queryIdx], keyPoints2[x.trainIdx]) for x in mathces]
        KPChoose = [ ((int (round (x.pt[0])), int (round (x.pt[1]))), (int (round (y.pt[0])), int (round (y.pt[1])))) for (x, y) in KPChoose]
        KPChoose = [ x for x in set (y for y in KPChoose)]
        return KPChoose
        pass

    def __normalization (self, keyPoints):
        return keyPoints[:8]
        pass

    pass


