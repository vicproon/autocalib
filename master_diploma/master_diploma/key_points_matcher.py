import cv2
import logging as log

class key_points_matcher(object):
    """description of class"""

    __keyPoints = []

    def detectKeyPoints (self, image):
        detector = cv2.xfeatures2d.SIFT_create()
        keyPoints = detector.detect(image)
        return keyPoints
        pass

    def computeDescriptors (self, image, keyPoints):
        extractor = cv2.xfeatures2d.FREAK_create()
        keyPointsRes, descriptors = extractor.compute (image, keyPoints)
        return keyPointsRes, descriptors
        pass

    def pointsFilteringByDistance (self, keyPoints1, keyPoints2, matches, vicinityThreshold):
        matches = sorted (matches, key = lambda x:x.distance)[:500]
        
        result = []
        
        for current in matches:
            isGoodPoint = True
            for test in result:
                x1,y1 = keyPoints1 [current.queryIdx].pt
                xt1,yt1 = keyPoints1 [test.queryIdx].pt
                
                if (abs (x1 - xt1) < vicinityThreshold and abs (y1 - yt1) < vicinityThreshold):
                    isGoodPoint = False
                    break
                    pass
                pass
            if (isGoodPoint): result.append (current)
            pass
        return result
        pass

    def pointsFilteringByThreshold (self, matches, threshold):
        minDistance = (min (matches, key = lambda x:x.distance)).distance
        goodMatches = filter (lambda x: (x.distance <= (minDistance * threshold)), matches)
        return goodMatches
        pass

    def saveKeyPoints (self, points1, points2):
        self.__keyPoints = self.__keyPoints + [(point1, point2) for (point1, point2) in zip (points1, points2)]
        pass

    def loadKeyPoints (self):
        return zip (*self.__keyPoints)
        pass
    pass


