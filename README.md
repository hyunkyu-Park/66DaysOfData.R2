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

__Day 7__

ㆍIn general, the first layer of the neural network that is directly processing the pixel-level data has to have at least as many parameters as there are pixels. Otherwise, some of the pixels aren't connect to anything and they will be ignored. The key insight in CNNs is the realization that even if there are millions of parameters related to the pixels of the input image, there doesn't necessarily have to be millions of different parameters. A CNN is composed of a collection or "bank" of so- called filters that are rectangular image patches much smaller than the original image. These filters are matched to different places on the image in a process that corresponds to the mathematical operation of convolution, hence the name CNN. In terms of the neural network architecture, each filter corresponds to a set of parameters that are used to multiply pixel values inside the rectangular patch. The same set of parameters is shared by different neurons that process inputs within different patches. The network can contain multiple convolutional layers: in the first layer, the patches contain pixels in the original image, and in the further layers, the patches contain outputs from the previous layer which is structured in a rectangular form just like the input image.  

Video link : https://www.youtube.com/watch?v=DEcvz_7a2Fs&list=PL_Nji0JOuXg2udXfS6nhK3CkIYLDtHNLp&index=15
![image](https://user-images.githubusercontent.com/68415173/112751638-f71e5e00-9009-11eb-9444-e108b9a66a9b.png)

__Day 8__

ㆍ As a matter of fact, I barely understand definition of the SVM yesterday. Now, however, I would like to say I almost understand the concept of the SVM. SVM is one of the traditional binary classification methods and is a classification technique for finding hyper-planes that can divide space into (N-1) dimensions. For example, A line can classify things in 2 dimensional space and A plane can classify things in 3 dimensional space. In here, the optimal boundary can be found through margins of SVM.  
![image](https://user-images.githubusercontent.com/68415173/112809967-8b91ca80-90b5-11eb-8894-b5bfe56297f8.png)
https://blog.naver.com/winddori2002/221662413641
Margin means the distances between each class or data located at the end of each class.  
Support vector is the closest data from margin. In the figure above, the two data above the dotted line are Support vector. The reason why this is called a support vector is that it supports the hyperplane function because the position of the boundary (ultra plane) depends on the location of these data.

__Day 9__

ㆍI solved the algorithm problem after a long time. It was about Brute Force and felt difficult. It's because I do not solved algorithm problems periodically.  
ㆍI looked at GAN, CNN, and RNN and thought of ideas to do projects using deep learning. I would like to plan a project for the public good.


__Day 10__

ㆍAccording to Yann LeCun, founder of the CNN deep learning algorithm, Gan(Generative Adversarial Network) is the most interesting idea in the last ten years in machine learning.  
![image](https://user-images.githubusercontent.com/68415173/113148768-37314b00-926d-11eb-81d8-888ba94d4e7a.png)
In GAN, generator and discriminator are the main factors. The generator learns to deceive the discriminator, and vice versa, the discriminator continues to learn not to be deceived by the generator. At this point, We don't train two things separately, we train one by one in rotation.Through gan, black-and-white images can be changed to color photographs or night images to daytime images. In other words, you can look at it from the perspective of creating something that seems to be real.  
ㆍCycle Gan is used when we do not have pair data such as dealing with Van Gogh's painting.  
ㆍStack Gan is used to create picture based on text.

__Day 11__

ㆍWhile searching for sources for GAN, I learned about Professor Yann LeCun's NYU-deep learning lecture and I was going to study. However, I felt that it was still a little too much to follow the lecture, so I was able to find a lecture with a relatively low entry barrier called Deep Learning for All and started to learn.  
ㆍInterestingly, the first lecture was not about machine learning or deep learning, but about Docker settings. I didn't download it because I have not used Docker and I thought it could be replaced by a Jupiter laptop or Google COLAB.

__Day 12__

ㆍAs a matter of fact, I literally did not have time to study..However, I watched second lecture of Deep Learning for All for 5 minutes which is the minimum time to keep promise of 66DaysOfData.  
ㆍMachine learning is kind of SW. It is used to problems that human cannot manage all the options and number of cases.  
-Spam filter: many rules  
-Automatic driving: too many rules  
ㆍMachine learning: 'Field of Study that gives computers the ability to learn without being explicitly programmed' Arthur Samuel(1959)

__Day 13__

##ㆍSupervised Learning: Most common problem type in ML  
- Learning with labeled samples(ex: Image labeling, Email spam filter)  
- Type of supervised learning  
a) Predicting final exam score based on time spent - regression  
b) Pass/Non-pass based on time spent - binary classification  
c) Letter grade(A, B, C, E and F) based on time spent - multi-label classification  
  
##ㆍUnsupervised Learning  
- Google news grouping  
- Word Clustering  
