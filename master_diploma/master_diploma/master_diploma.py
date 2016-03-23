import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from f_matrix_computer import f_matrix_computer
from key_points_matcher import key_points_matcher
import quality_checker as qch

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

import yaml
def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat
yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)

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
    for m in matches[:30]:
        pts2.append (keyPoints2 [m.trainIdx].pt)
        pts1.append (keyPoints1 [m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    #Точки для вычислений
    ptc1, ptc2 = [], []
    for m in matches[:20]:
        ptc2.append (keyPoints2 [m.trainIdx].pt)
        ptc1.append (keyPoints1 [m.queryIdx].pt)

    ptc1 = np.int32(ptc1)
    ptc2 = np.int32(ptc2)
    
    #F, mask = cv2.findFundamentalMat (ptc1, ptc2, cv2.FM_8POINT)
    F = fmc.compute (ptc1, ptc2)
    #F = fmc.compute (keyPoints1, keyPoints2, matches)
    #print 'F: ' + str(F)

    with open ("extrinsics.yml", "r") as file:
        GTF = yaml.load (file.read ())['F']
        pass

    print 'Разница норм матриц: '.decode ('utf-8') + str (qch.matrixNormTest (F, GTF))
    printStr = 'Среднее расстояние между эпиполярными линиями Ground Truth и тестовой матрицами: {0:.3f}\n'
    printStr = printStr + 'Среднее расстояние между эпиполярными линиями тестовой матрицы: {1:.3f}'
    print (printStr.format (*qch.epipolAverageDistance (F, GTF)).decode ('utf-8'))
    print 'Расстояние между точками пересечения эпиполярных линий Ground Truth и тестовой матрицами: '.decode ('utf-8') + str (qch.centerPointsTest (F, GTF))
    
    #print line
    #print GTLine

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines (pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape (-1,3)
    img5,img6 = drawlines (image1, image2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines (pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape (-1,3)
    img3,img4 = drawlines (image2, image1, lines2, pts2, pts1)

    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()

    #cv2.imshow ('image1', image3)
    #cv2.imshow ('image2', image4)

    pass

if __name__ == "__main__":
    image1 = cv2.imread ("20160203_155936_left.000100.bmp", 0)
    image2 = cv2.imread ("20160203_155936_right.000100.bmp", 0)
    #image2 = cv2.imread ("20160203_155936_left.000100.bmp", 0)

    #pointTracking ('forTracking')
    debug8Points (image1, image2)

    cv2.waitKey()
    cv2.destroyAllWindows()
pass