# Andy Bodell
# Logistic Regression 

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix as cf

def sigmoid(x,w):
	return 1/(1 + np.exp(-np.dot(x,w)))


def model(x,y,w,m,alpha,iteration,ones):
	costValues = []
	for i in range(iteration):
		h = sigmoid(x,w)
		cost = -y * np.log(h) - (1-y) * np.log(1-h)
		j = (1/len(m)) * np.dot(ones,cost)
		costValues.append(j)
		gradient = alpha/(len(m)) * np.dot(np.subtract(h,y).T,x)
		w = w - gradient.T
		
	return w,h,costValues,j,cost, alpha
	
def main():

    # load the training data in
    filename = 'P3train.txt'
    # we need to add the new features to the file
    originalFile = np.loadtxt(filename, skiprows = 1)
    x1 = originalFile[:,0]
    x2 = originalFile[:,1]
    y1 = originalFile[:,2]
    originalFile = originalFile[:,2]
    originalFile = np.reshape(originalFile, (len(x1), 1))
    thePower = 2
    fout1 = open('bodell_andy_P3train.txt', 'w')
    fout1.write('85\t9\n')
    for k in range(len(x1)):
        for i in range(thePower + 1):
            for j in range(thePower + 1):
                answer = (np.power(x1[k], j)) * (np.power(x2[k], i))
                if (answer != 1):
                    fout1.write(str(answer) + '\t')
        fout1.write(str(y1[k]) + '\n')
    fout1.close()
    filename = input("Enter the name of the training file: ")
    # load the data, but skip the first row
    x = np.loadtxt(filename,skiprows = 1)
    # load the classification values into the y array
    y = x[:,-1]
    # change the y array to a m x 1 list
    y = np.reshape(y, (len(y), 1))
    # remove the classification values from the x list
    x = x[:,:-1]
    # create a list of ones so we can do calculations
    m = np.ones([len(x),1])
    # add the list of ones to the far left column of the x array
    x = np.concatenate((m,x),axis = 1)
    # create the list of the weights which will all start at 0
    w = np.zeros([len(x[0]), 1])
    # learning rate, alpha
    alpha = 0.1
    # number of iterations we will complete
    iteration = 100000
    ones = np.ones([1,x.shape[0]])

    # run the model and return the necessary values
    w,h,costValues,j,cost, alpha = model(x,y,w,m,alpha,iteration,ones)
    print(f"\nAlgorithm went through {iteration} iterations\n")
    print(f"Final value of J on the training set is {j[0][0]}\n")
    print(f"The final value for the learning rate is {alpha}\n")
    print(f"The weights of the training set are \n{w}")
    finalH = sigmoid(x, w)
    # reshape into iteration x 1 list
    costValues = np.reshape(costValues, (iteration, 1))
    plotIterations = list(range(1, len(costValues) + 1))

    x1_min, x1_max = x1.min() - .1, x1.max() +.1
    x2_min, x2_max = x2.min() -.1, x2.max() + .1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    Z1 = xx1
    Z2 = xx1**2
    Z3 = xx2
    Z4 = xx1 * xx2
    Z5 = (xx1**2) * xx2
    Z6 = xx2**2
    Z7 = xx1 * (xx2**2)
    Z8 = (xx1**2) * (xx2**2)
    Z = w[0] + Z1 * w[1] + Z2 * w[2] + Z3 * w[3] + Z4 * w[4] + Z5 * w[5] + Z6 * w[6] + Z7 * w[7] + Z8 * w[8]
    plt.contour(xx1, xx2, Z, [0], colors = 'red', linestyles = 'dotted')
    plt.scatter(x1, x2, c = y1, cmap = plt.cm.rainbow)
    plt.show()

    
    plt.title("Number of Iterations vs. J(w)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost Value J(w)")

    plt.plot(plotIterations, costValues, 'bo')
    plt.show()
    
# --------------------------------------------------------------------
# Now for testing set

    fileTest = 'P3test.txt'
    # load the testing file then add the new features
    originalTest = np.loadtxt(fileTest, skiprows = 1)
    testX1 = originalTest[:,0]
    testX2 = originalTest[:,1]
    testY1 = originalTest[:,2]
    originalTest = originalTest[:,2]
    originalTest = np.reshape(originalTest, (len(testX1), 1))
    fout2 = open('bodell_andy_P3test.txt', 'w')
    fout2.write('33\t9\n')
    newPower = 2
    for k in range(len(testX1)):
        for i in range(newPower + 1):
            for j in range(newPower + 1):
                testAnswer = (np.power(testX1[k], j)) * (np.power(testX2[k], i))
                if (testAnswer != 1):
                    fout2.write(str(testAnswer) + '\t')
        fout2.write(str(testY1[k]) + '\n')
    fout2.close()
    fileTest = input("Enter the name of the test file: ")
    testX = np.loadtxt(filename, skiprows = 1)
    testY = testX[:,-1]
    testY = np.reshape(testY, (len(testY), 1))
    testX = testX[:,:-1]
    testM = np.ones([len(testX),1])
    testX = np.concatenate([testM, testX], axis = 1)
    testH = sigmoid(testX, w)

    testOnes = np.ones([1, testX.shape[0]])

    testCost = -testY * np.log(testH) - (1 - testY) * np.log(1 - testH)
    testJ = (1/len(testM)) * np.dot(testOnes, testCost)
    yPrediction = np.round(testH)
    print(f"Final J value on the test set is {testJ}")
    print(cf(testY, yPrediction))

    tn, fp, fn, tp = cf(testY, yPrediction).ravel()
    print(f"TP is {tp}")
    print(f"TN is {tn}")
    print(f"FP is {fp}")
    print(f"FN is {fn}")

    matrix = cf(testY, yPrediction)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"Accuracy is {accuracy}")
    precision = tp/(tp + fp)
    print(f"Precision is {precision}")
    recall = tp/(tp + fn)
    print(f"Recall is {recall}")
    f1 = (2 * precision * recall) / (precision + recall)
    print(f"F1 score is {f1}")
    confusion = cf(testY, yPrediction)
    display = ConfusionMatrixDisplay(confusion_matrix = confusion)
    display.plot()
    plt.show()


if __name__ == "__main__":
    main()