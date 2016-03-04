import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from f_matrix_computer import f_matrix_computer
from key_points_matcher import key_points_matcher

def pointTracking (imagesFolder):
    images = list()
    resultImage = None
    for file in sorted(os.listdir (imagesFolder)):
        image = cv2.imread (imagesFolder + '/' + file, 0)
        images.append (image)
        pass

    images.reverse ()
    color = np.random.randint (0,255,(100,3))
    gfttParametrs = dict ( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

    prevKeyPoints = cv2.goodFeaturesToTrack (images[0], **gfttParametrs)
    prevImage = images[0]
    #keyPoints = prevKeyPoints
    keyPoints = 0

    resultImage = cv2.cvtColor (images[0], cv2.COLOR_GRAY2BGR)
    mask = np.zeros_like (resultImage)

    for image in images[1:]:
        keyPoints, status, err = cv2.calcOpticalFlowPyrLK (prevImage, image, prevKeyPoints, keyPoints)

        gnew = keyPoints [status==1]
        gold = prevKeyPoints [status==1]

        for i, (new,old) in enumerate (zip (gnew, gold)):
            a,b = old.ravel()
            c,d = new.ravel()
            mask = cv2.line (mask, (a,b), (c,d), color[i].tolist() , 2)
            pass
        cv2.imshow ('mask', mask)
        prevImage = image
        prevKeyPoints = keyPoints
        keyPoints = 0
        pass

    resultImage = cv2.add (resultImage, mask)

    cv2.imshow ('try', resultImage)
    pass

def debug8Points (image1, image2):
    image3 = 0
    image4 = image2
    kpm = key_points_matcher ()
    fmc = f_matrix_computer ()

    keyPoints1 = kpm.detectKeyPoints (image1)
    keyPoints2 = kpm.detectKeyPoints (image2)

    #keyPoints1 = cv2.goodFeaturesToTrack (image1, 1512, 0.01, 2, useHarrisDetector = True)
    #keyPoints2 = cv2.goodFeaturesToTrack (image2, 1512, 0.01, 2, useHarrisDetector = True)
    #keyPoints1 = cv2.KeyPoint_convert(keyPoints1)
    #keyPoints2 = cv2.KeyPoint_convert(keyPoints2)
    
    keyPoints1, descriptors1 = kpm.computeDescriptors (image1, keyPoints1)
    keyPoints2, descriptors2 = kpm.computeDescriptors (image2, keyPoints2)

    bf = cv2.BFMatcher (cv2.NORM_HAMMING)
    matches = bf.match (descriptors1, descriptors2)
    matches = sorted(matches, key = lambda x:x.distance)
    #print '\n'.join (str(x.queryIdx) + ' ' + str(x.trainIdx) for x in matches)

    image3 = cv2.drawMatches(image1, keyPoints1, image2, keyPoints2, matches[:20], image3)
    #image3 = cv2.drawKeypoints (image1, keyPoints1, image3)
    #image4 = cv2.drawKeypoints (image2, keyPoints1, image4)

    #plt.imshow(image3),plt.show()

    fmc.compute (keyPoints1, keyPoints2, matches[:20])
    
    pts1, pts2 = [], []
    for m in matches[:20]:
        pts2.append (keyPoints2 [m.trainIdx].pt)
        pts1.append (keyPoints1 [m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat (pts1, pts2, cv2.FM_LMEDS)
    print F

    #cv2.imshow ('image1', image3)
    #cv2.imshow ('image2', image4)

    pass

if __name__ == "__main__":
    image1 = cv2.imread ("20160203_155936_left.000100.bmp", 0)
    image2 = cv2.imread ("20160203_155936_right.000100.bmp", 0)

    #pointTracking ('forTracking')
    debug8Points (image1, image2)

    cv2.waitKey()
    cv2.destroyAllWindows()
pass