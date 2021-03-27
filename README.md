# 66DaysOfData R2
66DaysOfData Challenge
----------------------
Blog  
https://blog.naver.com/pbh6020
----------------------

This is a very simple challenge. There are only two requirements:

1. Learn data science for a minimum of 5 minutes per day for 66 days straight (consistency)
2. Share what you will be working on each day on the platform of your choice using #66DaysOfData (accountability)


__Day 1__

  ㆍThe difference between Linear Regression and Logistic Regression lies in what happens after the above linear function has been calculated. In Linear Regression, we are done and the Linear function is going to be the output. In Logistic Regression, we take the result of the Linear function and apply the specific non-linear function called the sigmoid function to it.  
  ㆍThe sigmoid function converts any number to value between 0 and 1. You can then use this to obtain a probability value.

__Day 2__

  ㆍIt is important to realize that the most important factor in determining whether an application will be successful is often not which machine learning method you choose to use, but how you use it. The amount of the training data, the way the data is preprocessed and represented, the way the results are interpreted and applied. However, at some point the methods themselves may become the limiting factor. One such limit is the linearity of the model.  
  ㆍI have taught since last Thursday in Jincheon from elementary to high schools. I could feel that children are pretty exposed to AI but they barely aware they live with AI. In addition, students are very interested in learning and understanding AI than I thought they are. I think it's a good sign.


__Day 3__

ㆍIn neural networks, we use slightly different terminology. Instead of coefficients we say weights.  
Furthermore, the non-linear part of the model, which in the case of logistic regression was the sigmoid function, is called the activation function. Neural networks are composed of 3 parts called Input layers, hidden layers, and output layers. As I mentioned above, we call the function applied in the latter stage the activation function. This crucial concept, non-linear activation function, contributes to making neural networks different from just big fat linear regression model.   
ㆍ In hidden layer we also have a bias node which is not connected to the input nodes. The purpose of the bias node is functionally the same with the intercept term is a linear regression. It can shift the input coming from a layer to another layer by some constant value.  
ㆍPictures of elementary school classes.
![수업](https://user-images.githubusercontent.com/68415173/112296907-9c6ac680-8cd8-11eb-9ee5-0d678c9b7814.jpg)
![수업1](https://user-images.githubusercontent.com/68415173/112296917-9d9bf380-8cd8-11eb-9555-f00587700ee3.jpg)

__Day 4__

ㆍI read the overall article about deep learning. As I read about deep learning, the more I wondered what is meta-cognition. So I found and studied lectures on meta-cognition and 'Karl Dunker's problem' who asked questions about human scarcity. I used these new knowledge to High school who were not interested in AI and it was very effective.  
ㆍI did a class satisfaction survey in today's class (elementary and high school), and I was very proud that the results came out better than I expected. It's a shame that we can't upload pictures together because we can't share the results of the survey.

__Day 5__

ㆍ In machine learning models, criteria are needed to determine how similar the actual and predicted values are, which is called the loss function.  
ㆍ Models can be broadly divided into regression and classification. Loss functions can also be divided accordingly. There are typical loss functions 'MAE', 'MSE', 'RMSE' for regression. On the other hand, we usually use cross-entropy for classification.  
MAE(Mean Absolute Error): It is easy to understand the degree to which the entire data is learned. However, Because absolute values are taken, it is difficult to determine how errors occur and whether they are negative or positive. Also, it is difficult to collect optimal values because the distance traveled is constant even when it is close to the optimal value.  
MSE(Mean Squared Error): It is the most commonly used loss function. Because of square, values smaller than 1 are becoming smaller and more are becoming larger. This is a disadvantage because it can cause distortion of the value. However, MSE is characterized by calculating the correct answer rate for other wrong answers in addition to the error of the correct answer rate for the actual value. In addition, unlike MAE, it is easy to converge to the optimal value because the distance traveled changes differently as you approach the optimal value.


__Day 6__

ㆍKey words of SVM(Support Vector Machine)  
- Hyperplane: infinite number of hyperplanes exists in SVM. we need to find the best division of two classes. Margin must be large to find the most effective boundary.
- Margin: The distance between the data point and the hyperplane. To be exact, the closest data of each class to the border.  
- Kernal Trick: A kernel is a function that outputs real number by receiving x,y factors such as mistakes, functions, and vectors. The kernel trick is to use these kernel functions to view the data at a higher dimension.
