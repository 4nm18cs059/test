import cmath
import os
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
dir = 'act'

# data = []

# cate = ['Sitting', 'Walk']
# for ct in cate:
#     path = os.path.join(dir,ct)
#     lab = cate.index(ct)

#     for im in os.listdir(path):
#         impt = os.path.join(path,im)
#         fim= cv2.imread(impt,0)
#         try:
#           fim= cv2.resize(fim,(200,200))
#           image = np.array(fim).flatten()

#           data.append([image,lab])
#         except Exception as e:
#             pass

# print(len(data))


# pick_in = open('data1.pickle','wb')
# pickle.dump(data,pick_in)
# pick_in.close()

pick_in = open('data1.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()


random.shuffle(data)

fet = []
lab = []

for fets, labs in data:
    fet.append(fets)
    lab.append(labs)

xtrain, xtest, ytrain, ytest = train_test_split(fet, lab, test_size=0.70)


model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(xtrain, ytrain)

pick = open('model.sev', 'rb')
# pickle.dump(model,pick)
model = pickle.load(pick)
pick.close()


predict = model.predict(xtest)
accu = model.score(xtest, ytest)
cate = ['Sitting', 'Walk']

mypt = xtest[0].reshape(200, 200)
# plt.imshow(mypt)
# plt.show()
# Initializing the HOG person
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


#mypt = imutils.resize(mypt, width=min(500, mypt.shape[1]))


# Detecting all humans
(humans, _) = hog.detectMultiScale(
    mypt, winStride=(5, 5),   padding=(3, 3), scale=1.21)
if len(humans) >= 1:
    print('Human Detected ', len(humans))
else:
    print('Human Detected ')

print("Accuracy:", accu)

print("Prediction is:", cate[predict[0]])

# Drawing the rectangle regions
for (x, y, w, h) in humans:
    cv2.rectangle(mypt, (x, y), (x + w, y + h),  (0, 0, 255), 2)

# Displaying the output Image
# mypt= imutils.resize((400,400))
cv2.imshow("Image", mypt)
cv2.waitKey(0)
cv2.destroyAllWindows()
