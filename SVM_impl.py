from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits = datasets.load_digits()

#-----------------------------------------------------------------------
#                               Main code
#-----------------------------------------------------------------------


#images are flattened in order to change the data to (data,target)
n = len(digits.images)
data = digits.images.reshape((n,-1))
#print(data)

#using the default svm classifier
clf = svm.SVC(kernel='rbf',gamma='auto')

xTrain, xTest, yTrain, yTest = train_test_split(data, digits.target, test_size=0.3, shuffle=True)

clf.fit(xTrain,yTrain)
pred = clf.predict(xTest)

good = 0
for x in range(len(pred)):
    if pred[x] == yTest[x]:
        good+=1

success = (good/len(pred))*100
print("success rate = ", success, "%")

#-----------------------------------------------------------------------
#                  Testing/Displaying functions results
#-----------------------------------------------------------------------

'''
#data
print(numbers.images)
#classes
print(numbers.target)
#testing zip
print(list(zip(numbers.images, numbers.target)))

#Checking if shuffling is done correctly
plt.imshow(xTrain[0])
print(yTrain[0])
plt.show()

#testing out reshape
plt.imshow(xTrain[0].reshape((8,8)))
print(yTrain[0])
plt.show()

#testing out result
print("Predicted =", clf.predict(xTest)[0])
print("Target =", yTest[0])
plt.imshow(xTest[0].reshape((8,8)))
plt.show()

fig, axes = plt.subplots(1,5)

fig.suptitle('Sample of the training set')

for x,i in zip(axes[:],range(5)):
    x.set_axis_off()
    x.imshow(xTrain[i].reshape(8,8), cmap=plt.cm.gray_r)
    x.set_title('Target = %i' % yTrain[i])

#plt.show()
'''

resFig, resAxes = plt.subplots(1,5)

resFig.suptitle('Results')

for x,i in zip(resAxes[:],range(5)):
    x.set_axis_off()
    x.imshow(xTest[i].reshape(8,8), cmap=plt.cm.gray_r)
    x.set_title('Target = %i' % yTest[i] + '\nPrediction = %i' % pred[i])

plt.show()