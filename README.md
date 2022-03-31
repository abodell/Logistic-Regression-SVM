# Logistic-Regression-SVM
Created two machine-learning models to predict if capacitors would pass a quality control test.  
The first model is created using Logistic Regression which was built from scratch using NumPy.  
First off the new features are added to the data since the data is not linear.  If it was linear we could just use 
two features, but since it was not new features such as x1 ^ 2 * x2 and x1 * x2 ^ 2, etc. were added.  This will make the model
much more accurate.  After transforming the data, it is passed through the model in order to recieve weights.  These weights will
be the basis on how to graph the decision boundary around the data.  I will attach images of the decision boundary for both models.  


SVM was much of the same except this time I used sklearn packages in order to formulate the model.  I still added new features since
I was working with the same set of non-linear data.  This time though the sklearn packages handled the bulk of the calculations for me
and I mainly just had to focus on drawing the decision boundary on the data set.  
