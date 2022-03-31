import matplotlib.pyplot as plt
import numpy as np
import sklearn.svm as sc
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, confusion_matrix


def main():
    filename = input("Enter the training set file: ")
    dataTrain = np.loadtxt(filename, skiprows = 1)
    # split the data into x and y values
    yValues = dataTrain[:,2]
    xValues1 = dataTrain[:,0]
    xValues = dataTrain[:,:-1]
    xValues2 = dataTrain[:,1]
    # run the model on the data
    clf = sc.SVC(kernel = 'rbf')
    clf.fit(xValues, yValues)
    ax = plt.gca()
    xValues1 = np.reshape(xValues1, (len(xValues1), 1))
    xValues2 = np.reshape(xValues2, (len(xValues2), 1))
    xx = np.linspace(xValues1.min() - .1, xValues1.max() + .1)
    yy = np.linspace(xValues2.min() - .1, xValues2.max() + .1)
    meshY, meshX = np.meshgrid(yy, xx)
    xy = np.vstack([meshX.ravel(), meshY.ravel()]).T
    # plot the original data
    plt.scatter(xValues1, xValues2, c = yValues, cmap = plt.cm.rainbow)
    # plot the support vectors 
    Z = clf.decision_function(xy).reshape(meshX.shape)
    plt.contour(meshX, meshY, Z, colors = 'k', levels = [-1, 0, 1], alpha = .5, linestyles = ["--", "-", "--"])
    plt.show()
    # run the algorithm on the test set
    testFile = input("Enter test file name: ")
    dataTest = np.loadtxt(testFile, skiprows=  1)
    yTest = dataTest[:,2]
    xValues = dataTest[:,:-1]
    # print the confusion matrix
    rbf_predictions = clf.predict(xValues)
    confusion = confusion_matrix(yTest, rbf_predictions, labels = clf.classes_)
    tp, tn, fp, fn = confusion_matrix(yTest, rbf_predictions).ravel()
    print(f"TP is {tp}")
    print(f"TN is {tn}")
    print(f"FP is {fp}")
    print(f"FN is {fn}")
    accuracy = accuracy_score(yTest, rbf_predictions)
    print(f"Accuracy is {accuracy}")
    precision = tp/(tp + fp)
    print(f"Precision is {precision}")
    recall = tp/(tp + fn)
    print(f"Recall is {recall}")
    f1 = (2 * precision * recall) / (precision + recall)
    print(f"F1 score is {f1}")
    display = ConfusionMatrixDisplay(confusion_matrix = confusion, display_labels = clf.classes_)
    display.plot()
    plt.show()




if __name__ == "__main__":
	main()
