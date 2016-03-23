import cv2

class key_points_matcher(object):
    """description of class"""
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

    def pointsFiltering (self, keyPoints1, keyPoints2, matches, vicinityThreshold):
        matches = sorted (matches, key = lambda x:x.distance)[:500]
        print 'matches before: ' + str (len (matches))
        
        result = []
        
        for current in matches:
            isGoodPoint = True
            for test in result:
                x1,y1 = keyPoints1 [current.queryIdx].pt
                #x2,y2 = keyPoints2 [current.trainIdx].pt
                xt1,yt1 = keyPoints1 [test.queryIdx].pt
                #xt2,yt2 = keyPoints2 [test.trainIdx].pt
                
                if (abs (x1 - xt1) < vicinityThreshold and abs (y1 - yt1) < vicinityThreshold):# and abs (x2 - xt2) < vicinityThreshold and abs (y2 - yt2) < vicinityThreshold):
                    isGoodPoint = False
                    break
                    pass
                pass
            if (isGoodPoint):
                result.append (current)
            pass
        print 'matches after: ' + str (len (result))
        return result
        pass

    pass


