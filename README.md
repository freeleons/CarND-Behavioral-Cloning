# CarND-Behavioral-Cloning Project: Use Deep Learning to Clone Driving Behavior
At first, I really do not have any clues about what's the deep neural network architecture look alike to handle such a problem. It is a regression problem and not a classification problem. So I asked my mentor about the architecture of the neural network and my mentor recommended me a paper called ["End to End Learning for Self-Driving Cars"](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) 

So I used their architecture to build the network. Here is what is look alike and this picture is from the paper, End to End Learning for Self-Driving Cars:
![alt text](Architecture.png "Architecture")

|Layer (type) |                    Output Shape    |      Param# 	|   Connected to|
|-------------|------------------|------------------------------|---------------|
|cropping2d_1 (Cropping2D)       | (None, 80, 320, 3) |   0     |      cropping2d_input_1[0][0]
|lambda_1 (Lambda)             |(None, 40, 160, 3)   | 0     |      cropping2d_1[0][0]
|batchnormalization_1 (BatchNorma |(None, 40, 160, 3)  |  160    |     lambda_1[0][0]
|convolution2d_1 (Convolution2D)  |(None, 40, 160, 24) |  1824   |     batchnormalization_1[0][0]
|maxpooling2d_1 (MaxPooling2D)    |(None, 20, 80, 24)  |  0      |     convolution2d_1[0][0]
|spatialdropout2d_1 (SpatialDropo |(None, 20, 80, 24)  |  0      |     maxpooling2d_1[0][0]
|convolution2d_2 (Convolution2D)  |(None, 20, 80, 36)  |  21636  |     spatialdropout2d_1[0][0]
|maxpooling2d_2 (MaxPooling2D)    |(None, 10, 40, 36)  |  0      |     convolution2d_2[0][0]
|spatialdropout2d_2 (SpatialDropo |(None, 10, 40, 36)  |  0      |     maxpooling2d_2[0][0]
|convolution2d_3 (Convolution2D)  |(None, 10, 40, 48)  |  43248  |     spatialdropout2d_2[0][0]
|maxpooling2d_3 (MaxPooling2D)    |(None, 5, 20, 48)   |  0      |     convolution2d_3[0][0]
|spatialdropout2d_3 (SpatialDropo |(None, 5, 20, 48)   |  0      |     maxpooling2d_3[0][0]
|convolution2d_4 (Convolution2D)  |(None, 5, 20, 64)   |  27712  |     spatialdropout2d_3[0][0]
|maxpooling2d_4 (MaxPooling2D)    |(None, 3, 10, 64)   |  0      |     convolution2d_4[0][0]
|spatialdropout2d_4 (SpatialDropo |(None, 3, 10, 64)   |  0      |     maxpooling2d_4[0][0]
|convolution2d_5 (Convolution2D)  |(None, 3, 10, 64)   |  36928  |     spatialdropout2d_4[0][0]
|maxpooling2d_5 (MaxPooling2D)    |(None, 2, 5, 64)    |  0      |     convolution2d_5[0][0]
|spatialdropout2d_5 (SpatialDropo |(None, 2, 5, 64)    |  0      |     maxpooling2d_5[0][0]
|flatten_1 (Flatten)              |(None, 640)         |  0      |     spatialdropout2d_5[0][0]
|dense_1 (Dense)                  |(None, 100)         |  64100  |     flatten_1[0][0]
|dense_2 (Dense)                  |(None, 50)          |  5050   |     dense_1[0][0]
|dense_3 (Dense)                  |(None, 10)          |  510    |     dense_2[0][0]
|dense_4 (Dense)                  |(None, 1)           |  11     |     dense_3[0][0]
Total params: 201,179
Trainable params: 201,099
Non-trainable params: 80
____________________________________________________________________________________________________

I have used the [Sample Training Data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) from Udacity to train the model.

I have tried lots of learning rates like 0.1, 0.01, 0.001, 0.0001, 0.00001 and so on. I found the learning rate 0.0003 is the best for the model because the driving behavior is the best I have seen. However, when I tried to run the model on the simulator, it cannot last for a lap. The car will run out of the road eventually. 

###Please discuss how did you decide the number and type of layers.
I can not decide how many number and types of layers, so I just followed the paper called, End to End Learning for Self-Driving Cars. I used their achitecture.

###Please discuss how would you evaluate the model.
I evaluate the model by using the loss durning the training of the model and I evaluate it by use the automous driving mode in the simulator to test drive the car.

###Please discuss why this model is suitable for this question.
I learned it from the paper, End to End Learning for Self-Driving Cars, that this model is suitable for this question.

###Please discuss what problems you have met.
After I trained the model and get model.json and model.h5, I use them to drive the car in the simulator, the car will always drive off the road. I have tried so many times with different learning rates. I still can not keep the car on the road for more than a lap. The best case is the car keep on the road for one lap.

###Please discuss what strategy have you been adopted.
I have no strategy for this project. I felt the learning curve for this project is too big.


I have tried so many times to adjust the learning rate. I can not make meet the specifications that "No tire may leave the drivable portion of the track surface", Could you give me some suggestions to guide me to meet this specification?










