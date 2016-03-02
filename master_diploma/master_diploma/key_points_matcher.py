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

    pass


