#Prediction of housing prices (**California**)

###Using the native dataset from Colab

This prediction machine learning model is required to supply feeding information for the next ML model part of a pipeline in development.

We can use ***supervised*** learning, because we have labeled training data. It is a ***regression*** task, because we will be using it to predict a single parameter, as opposed to classification task, where we would require to fit the predicted outputs into different classes (e.g. cheap, medium, expensive), so we have more than one type of output and it is qualitative, not quantative. Moreover, it will be a ***multiple*** regression task, because we will be using multiple variables for the prediction (as an imput information). Also, it will be ***univariative*** in contrast to multivariative, because we will predict only a single price for the district, and not many. Finally, the data is comparatively small and will not change rapidly so it wil be a ***plain batch*** learning model.

The performance will be measured by the means of the RMSE (or the root mean square error) measurement, which is a measurement of the errors, the model makes.

$$\displaystyle{\displaylines{RMSE (X,h) = \sqrt[]{\frac{1}{m}\sum_{i = 1}^{m}(hx^{i}-y^{i})^{2}}}}$$

m - number of instances (districts)

xi - vector of parameters (input variables without label)

yi - the label (output parameter)

h -  hypothesis (prediction function)

$$hx^{i}=\hat{y}^{i}$$

#Exercises
Using this chapter’s housing dataset:
1. Try a Support Vector Machine regressor (sklearn.svm.SVR), with various hyper‐
parameters such as kernel="linear" (with various values for the C hyperpara‐
meter) or kernel="rbf" (with various values for the C and gamma
hyperparameters). Don’t worry about what these hyperparameters mean for now.
How does the best SVR predictor perform?
2. Try replacing GridSearchCV with RandomizedSearchCV.
3. Try adding a transformer in the preparation pipeline to select only the most
important attributes.
4. Try creating a single pipeline that does the full data preparation plus the final
prediction.
5. Automatically explore some preparation options using GridSearchCV.
