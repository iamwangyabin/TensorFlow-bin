import cv2  
import numpy as np  
#import pdb  
#pdb.set_trace()#turn on the pdb prompt  
  
#read image  
path = 'G:\\pytext\\reco\\datasets\\train\\0b7b02087bf40ad1aa2ba0685d2c11dfa8ecce6a.jpg'
img = cv2.imread(path,cv2.IMREAD_COLOR)  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
cv2.imshow('origin',img)
  
#SIFT  
detector = cv2.xfeatures2d.SIFT_create()
keypoints = detector.detect(gray,None)  
img = cv2.drawKeypoints(gray,keypoints,img)  
#img = cv2.drawKeypoints(gray,keypoints,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  
cv2.imshow('test',img);  
cv2.waitKey(0)  
cv2.destroyAllWindows()  




G:\\pytext\\reco\\datasets\\train\\a50f4bfbfbedab644e5be45efc36afc379311e06.jpg
G:\\pytext\\reco\\datasets\\train\\a8014c086e061d9502ad0f6d70f40ad163d9caff.jpg



0b7b02087bf40ad122bb285d5d2c11dfa9ecce35.jpg
#coding=utf-8  
import cv2  
import scipy as sp  
  
img1 = cv2.imread('G:\\pytext\\reco\\datasets\\train\\908fa0ec08fa513db1f2d64c366d55fbb3fbd9cc.jpg',0) # queryImage  
img2 = cv2.imread('G:\\pytext\\reco\\datasets\\train\\a8014c086e061d9502ad0f6d70f40ad163d9caff.jpg',0) # trainImage  
img2 = cv2.imread('G:\\pytext\\reco\\datasets\\train\\9358d109b3de9c82795438c76781800a19d84341.jpg',0)
# Initiate SIFT detector  
sift = cv2.xfeatures2d.SIFT_create()
  
# find the keypoints and descriptors with SIFT  
kp1, des1 = sift.detectAndCompute(img1,None)  
kp2, des2 = sift.detectAndCompute(img2,None)  
  
# FLANN parameters  
FLANN_INDEX_KDTREE = 0  
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)  
search_params = dict(checks=50)   # or pass empty dictionary  
flann = cv2.FlannBasedMatcher(index_params,search_params)  
matches = flann.knnMatch(des1,des2,k=2)  
  
print('matches...',len(matches)  )
# Apply ratio test  
good = []  
for m,n in matches:  
    if m.distance < 0.75*n.distance:  
        good.append(m)  
print('good',len(good))
# #####################################  
# visualization  
h1, w1 = img1.shape[:2]  
h2, w2 = img2.shape[:2]  
view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)  
view[:h1, :w1, 0] = img1  
view[:h2, w1:, 0] = img2  
view[:, :, 1] = view[:, :, 0]  
view[:, :, 2] = view[:, :, 0]  
  
for m in good:  
    # draw the keypoints  
    # print m.queryIdx, m.trainIdx, m.distance  
    color = tuple([sp.random.randint(0, 255) for _ in range(3)])  
    #print 'kp1,kp2',kp1,kp2  
    cv2.line(view, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])) , (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1])), color)  
  
cv2.imshow("view", view)  
cv2.waitKey(0) 




path=''
def load_data(label):
    train=open('G:\\pytext\\reco\\datasets\\train.txt')
    image_path=[]
    root='G:\\pytext\\reco\\datasets\\train\\'
    for line in train.readlines():
        k=line.strip().split()
        if int(k[1])==label:
            file_path=os.path.join(root,k[0])
            image_path.append(file_path)
    return image_path

label=2
image_path=load_data(label)
img1 = cv2.imread('G:\\pytext\\reco\\datasets\\train\\2934349b033b5bb5f4a305393fd3d539b700bc84.jpg',0)
score=0
for image in image_path:
    img2 = cv2.imread(image,0) # trainImage  
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT  
    kp1, des1 = sift.detectAndCompute(img1,None)  
    kp2, des2 = sift.detectAndCompute(img2,None)  
    # FLANN parameters 
    FLANN_INDEX_KDTREE = 0  
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)  
    search_params = dict(checks=50)   # or pass empty dictionary  
    flann = cv2.FlannBasedMatcher(index_params,search_params)  
    matches = flann.knnMatch(des1,des2,k=2)  
    good = []  
    for m,n in matches:  
        if m.distance < 0.75*n.distance:  
            good.append(m)  
    print('good',len(good))
    score+=len(good)
min=9999.0
for i in matches:
    q=i[0]
    w=i[1]
    print(str(q.distance)+'  '+str(w.distance))




# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)

plt.imshow(img3),plt.show()

