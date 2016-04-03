import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import logging as log
from datetime import datetime
import yaml

from f_matrix_computer import f_matrix_computer
from key_points_matcher import key_points_matcher
from quality_checker import quality_checker
import utilities as util


yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", util.opencv_matrix)
with open ("extrinsics.yml", "r") as file:
    GTF = yaml.load (file.read ())['F']
    pass

image1 = cv2.imread ("20160203_155936_left.000100.bmp", 0)
image2 = cv2.imread ("20160203_155936_right.000100.bmp", 0)
log.basicConfig (filename= datetime.now().strftime('log %H : %M %d.%m.%Y.log'), level=log.INFO)
kpm = key_points_matcher ()
fmc = f_matrix_computer ()
qch = quality_checker (GTF, image1.shape)

def pointTracking (imagesFolder):
    resultImage = None
    images = list (zip (*(util.readImagesFromFolder (imagesFolder)[0]))[0])

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

    height, width = image1.shape[0], image1.shape[1]
    vicinityThreshold = (height + width) / 200
    print 'threshold: ' + str (vicinityThreshold)
    matches = kpm.pointsFiltering (keyPoints1, keyPoints2, matches, vicinityThreshold)
    
    #print '\n'.join (str(x.queryIdx) + ' ' + str(x.trainIdx) for x in matches)

    #image3 = cv2.drawMatches(image1, keyPoints1, image2, keyPoints2, matches[:20], image3)
    #image3 = cv2.drawKeypoints (image1, keyPoints1, image3)
    #image4 = cv2.drawKeypoints (image2, keyPoints1, image4)

    #plt.imshow(image3),plt.show()

    #Точки для тестирования
    pts1, pts2 = [], []
    for m in matches:
        pts2.append (keyPoints2 [m.trainIdx].pt)
        pts1.append (keyPoints1 [m.queryIdx].pt)

    pts1, pts2 = util.testPointsFilter (GTF, pts1, pts2)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    #Точки для вычислений
    ptc1, ptc2 = [], []
    for m in matches[:20]:
        ptc2.append (keyPoints2 [m.trainIdx].pt)
        ptc1.append (keyPoints1 [m.queryIdx].pt)

    ptc1 = np.int32(ptc1)
    ptc2 = np.int32(ptc2)
    
    #F, mask = cv2.findFundamentalMat (ptc1, ptc2, cv2.FM_RANSAC)
    F = fmc.compute (ptc1, ptc2)
    print 'F: ' + str(F/F[2,2])
    
    #print line
    #print GTLine

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines (pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape (-1,3)
    img5,img6 = util.drawlines (image1, image2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines (pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape (-1,3)
    img3,img4 = util.drawlines (image2, image1, lines2, pts2, pts1)

    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()

    #cv2.imshow ('image1', image3)
    #cv2.imshow ('image2', image4)

    pass

def pointsCollectingTest (imagesFolder):
    images = list (zip (*util.readImagesFromFolder (imagesFolder))[0])

    for (image1, image2) in images:
        keyPoints1 = kpm.detectKeyPoints (image1)
        keyPoints2 = kpm.detectKeyPoints (image2)

        keyPoints1, descriptors1 = kpm.computeDescriptors (image1, keyPoints1)
        keyPoints2, descriptors2 = kpm.computeDescriptors (image2, keyPoints2)

        bf = cv2.BFMatcher (cv2.NORM_HAMMING)
        matches = bf.match (descriptors1, descriptors2)

        pts1, pts2 = [], []
        for m in matches:
            pts2.append (keyPoints2 [m.trainIdx].pt)
            pts1.append (keyPoints1 [m.queryIdx].pt)
        pts1, pts2 = util.testPointsFilter (GTF, pts1, pts2)

        height, width = image1.shape[0], image1.shape[1]
        vicinityThreshold = (height + width) / 200
        matches = kpm.pointsFilteringByDistance (keyPoints1, keyPoints2, matches, vicinityThreshold)
        matches = kpm.pointsFilteringByThreshold (matches, 3)

        ptc1, ptc2 = [], []
        for m in matches:
            ptc2.append (keyPoints2 [m.trainIdx].pt)
            ptc1.append (keyPoints1 [m.queryIdx].pt)

        kpm.saveKeyPoints (ptc1, ptc2)

        keyPoints1, keyPoints2 = kpm.loadKeyPoints()
        keyPoints1, keyPoints2 = np.int32(keyPoints1), np.int32(keyPoints2)

        F, mask = cv2.findFundamentalMat (keyPoints1, keyPoints2, cv2.FM_LMEDS)
        #F = fmc.compute (ptc1, ptc2)

        #qch.computeTests (F, keyPoints1, keyPoints2, pts1, pts2)
        qch.printTests (F, keyPoints1, keyPoints2, pts1, pts2)
        pass
    pass

if __name__ == "__main__":
    #image2 = cv2.imread ("20160203_155936_left.000100.bmp", 0)

    #pointTracking ('forTracking')
    #debug8Points (image1, image2)
    pointsCollectingTest ('forCollecting')

    cv2.waitKey()
    cv2.destroyAllWindows()
pass