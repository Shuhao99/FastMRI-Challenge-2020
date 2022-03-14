# 2020 FaceBook FastMRI Challenge

---


*   Woojin Chung
*   Shuhao Zhang
*   Monika Manuela Hengki
*   Martijn Eugenius Nicolaas Frederik Louis Schendstok
*   Hyun Lee
*   Huu Thanh Tra Huynh

---


Reconstructed data: https://drive.google.com/open?id=1DQXOuyrwtqUnykioa_-Sd1RB6gQ7WFfq 


---




# 1. Introduction

The task is to reconstruct images using magnetic resonance imaging (MRI), which is a very powerful diagnostic tool for a wide range of disorders, including neurological, musculoskeletal, and oncological diseases. However, the long acquisition time in MRI, which can easily exceed half an hour, leads to low patient throughput, problems with patient comfort and compliance, artifacts from patient motion, and high exam costs.

To reduce the scanning time, undersampling strategies are often used. For this project, the undersampling strategy used is the Cartesian undersampling trajectory and applying random mask functions on our test data for an acquisition speed 4 times faster and for 8 times faster. 

The data acquired from an MRI scanner is a complex-valued k-space data, where k-space represents the spatial frequency information (Fourier coefficients) of an object imaged. To reconstruct that image, one simply performs the inverse discrete Fourier transform (DFT) to the k-space data. The dataset includes 71 k-space volumes, each volume containing around 35 slices, every slice being an image. This gives us a total of about 2,485 images.

The task at hand is to create and train a neural network that reconstructs images of the same quality of the original data from the undersampled data. We define the ground
truth image, which is the absolute value of the complex-valued fully sampled image, calculated (via inverse DFT) from the complex-valued fully sampled k-space. The similarity of the reconstructed image to the ground truth image will be measured using the Compute Structural Similarity Index Metric (SSIM). 


# 2. Design

The design of our final model is UNet, a fully convolutional neural network for biomedical image segmentation first designed by Olaf Ronneberger et al (Ronneberger, Fischer and Brox, 2015). The traditional convolutional neural networks have been widely used in classification problems where the number of possible outputs is far less than the number of inputs. However, the output of our task is the same size as the input. Therefore, we chose UNet, which have been found to be very effective for tasks where the output is of similar size as the input.

![alt text](https://drive.google.com/uc?id=1IlxlWavUIRssRUfCcDpBbZCQVsIMQz1G)

The model is composed of two main parts: Downsampling and upsampling layers. The downsampling section, which is the left-hand part of the UNet, follows the same method as the traditional convolutional neural networks. It applies two 3 x 3 convolutional filters, applies ReLu activation function, and applies 2 x 2 max pool. This process is repeated multiple times. 

After applying the downsampling, the model applies upsampling to increase the size of the output. The 2 x 2 filters are applied for transposed convolution to increase the size of the output. Then, 3 x 3 convolution filter are applied combined with ReLu. This process is repeated the same number of times as the number of downsampling process. This makes the network layout to resemble a U shape with downsampling process forms the left hand side of the U and the upsampling process forms the right hand part of the U.

Each upsample in the upsampling part of the network (right hand part of the U) needs to add pixels around the existing pixels and also in-between the existing pixels to eventually reach the desired resolution. This could be improved with some simple initialisation of the new pixels by using the weighted average of the pixels (using bi-linear interpolation).

We’ve followed the basic design of the above figure but with different input and output sizes. Also, we added padding in the 3 x 3 convolutional layer in the downsampling part to keep the same dimensions while the figure above did not. Also, to experiment with this model, we did hyperparameter tuning with  different loss functions, optimizers, different hyperparameters, which will be discussed in the experiments section.








# 3. Implementations

First we change the provided data_load_path function to divide the train data into train (80%) and test (20%).
Our final model is Unet model, which is usually used in biomedical field. We built a model based on fastMRI project of Zbontar, Jure et al. (2018). In the model we define a Convblock class for the convolution sequential layer. We insert the convolution layers to generate up-sampling layers and down-sampling layers in a for-loop.

For the up-sampling, we increase the channel number and keep the size of every output as same as input. After every up-sampling layer, we store the output in a stack, prepare for the copy  and crop process in Unet. It is a process of encoding. And then we write a down-sampling process to do the decoding. We do the max pool every time after an encoding layer, and we do the interpolate every time after decoding layer.

For the training, we write the generic code for loss backpropagation and optimiser. And return the loss value after every optimise.
In the train function. For every epoch, first we train the model with train data, and save the loss value in a list every iteration. Then we test the model with val data, and save the loss in a list every iteration. Because we only have a small quantity of train data, and we don’t have gt for the real test data. So we just use the ‘val data’ same as the test data. Finally we put the ssim value between the gt and the output we predict from val data into a list, and calculate the mean value of this list. We use this mean value to assess our model.

After we trained the model we save the parameters and use it for reconstruction.
We load in the h5 test files, containing the under sampled image volumes for both the acceleration of 4 and 8. We iterate over every slice per file per acceleration and reconstruct the slice image. These reconstructions are saved with the same structure as the original h5-file with the original name.


# 4. Experiments

To start off our experiment, we implemented our own convolutional model. The model had twelve layers that applied the convolutional neural network, RELU activation function, and batch normalization in each layer. Also, each layer had a kernel size of five and padding of two to conserve the image dimensions.

However, when we implemented this model on the MRI dataset, the SSIM actually went down then the undersampled inputs. So, our own model did not work. We think that one of the reasons for this underperformance is that we have not downsized the sample and kept the same image dimensions.

So after we have seen that our first model did not work, we implemented a new model that was already proven by researchers: the UNet model. When we implemented this model, we saw an improvement in the SSIM, so we’ve decided to keep this model.

## 4.1 Hyperparameter Tuning

With the new working model, we experimented with the hyperparameters. The following are the hyperparameters that we tuned: number of pooling layers, learning rate, optimizer, number of epoches, drop out rate, kernel size, and batch size. 

### Number of pooling layers
We would like to understand the influences of the number of pooling layers on the final result.The more pooling layers we have, the more layers we have in our model. We keep the same parameter values described in the previous code. We experiment then 4 numbers of poolings layers.


**5 pooling layers**: It takes 45 minutes to finish the training.
Final average SSIM: 0.61123

![5 pooling layers](https://drive.google.com/uc?id=1aTrOP7gbVAv286pPJdvZoI5-WSX02dtg)

**4 pooling layers**: It takes 28 minutes to finish the training.
Final average SSIM: 0.60325

![4 pooling layers](https://drive.google.com/uc?id=1YcxYPJvS3ke_Di5bSznGE7Fc4L5Vjxkn)

**3 pooling layers**: It takes 25 minutes to train the model.
Final average SSIM: 0.60158

![3 pooling layers](https://drive.google.com/uc?id=1Vf46IAMKUk3UeOo_mgek2NHEhyF-jOHp)

**2 pooling layers**: It takes 20 minutes to train the model.
Final average SSIM: 0.58736

![2 pooling layers](https://drive.google.com/uc?id=1ilGdBucL5NjiVZxrT4JVVIuPPgkwxeMo)

We can see that the more layers added, the longer the total time that the model takes to train and the better result we got. Even though it takes a longer time to train the model with 5 pooling layers, we believe that 5 pooling layers will give us a better result.

![Pooling layers](https://drive.google.com/uc?id=13a7J6POO5XTB5y7emyzcWKxH4YjRG9mp)

### Learning rate

Learning rate tells the stochastic gradient descent optimizer how far to move the weights in the direction opposite of the gradient for a mini-batch. If the learning rate is set to a low value, the training is more reliable, but optimization will take a long time because the gradient descent steps are tiny. On the other hand, if the learning rate is high, the training time is shorter, but it may not converge, or even diverge. The changes in weights can be too big that the optimizer overshoots the minimum and worsens the loss.

We experimented with 4 different learning rates, such as 1e-2, 1e-3, 1e-4, and 1e-5. Here are the SSIM results with respect to different learning rates:
- 1e-5

![1e-5 loss](https://drive.google.com/uc?id=1h1jwJDzSEOsH6VRNgSElUhLmvhXZYvUH)

Final average SSIM: 0.58334

- 1e-4

![1e-4 loss](https://drive.google.com/uc?id=1MZy_loCNAqRRhP017bTVVWnAQ9vftw0E)

Final average SSIM: 0.60684

- 1e-3 (0.001)

![1e-3 loss](https://drive.google.com/uc?id=1oBmilOBC-FZ61BrkjAOeTSQE-Ul-cvYg)

Final average SSIM: 0.60325

- 1e-2 (0.01)

![1e-2 loss](https://drive.google.com/uc?id=1-MqfETVVj2lBD0GOk-p1hA_dmE2XQf-3)

Final average SSIM: 0.58517

![SSIM vs lr](https://drive.google.com/uc?id=1mPQM7a01fb0DfJetp0GWsU3-Z17jTYtX)

![SSIM increase vs lr](https://drive.google.com/uc?id=1SAV-AhxdVCmPmj6zZ0ZiXTgZ-rzKQKx1)

As such, for 30 epochs, our learning rate should ideally be 1e-4 as it produces the highest final average SSIM.


### Optimiser

We also experimented with our choice of optimiser. Optimisers update the weight parameters to minimize the loss function. We tried 3 different optimisers, such as Adagrad, RMSProp, and Adam.

AdaGrad is a modified stochastic gradient descent algorithm that adapts its learning rate per parameter (Duchi, Hazan and Singer, 2011). The algorithm increases the learning rate for sparser parameters and decreases the learning rate for parameters that are less sparse. The algorithm is suitable for sparse data in large scale neural networks.

![Adagrad loss](https://drive.google.com/uc?id=1oBaIQNegZSHlTjmdZnGc60X4OrIKZdls)

Final average SSIM: 0.48103

RMSProp is another method in which the learning rate is adapted for each of the parameters (Tieleman and Hinton, 2012). The idea is to divide the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight.

![RMSProp loss](https://drive.google.com/uc?id=1XPmk4LNmwA3n3Av6fk8zw92vukjDLsmg)

Final average SSIM: 0.60154

Adam can be considered as a combination of RMSprop and Stochastic Gradient Descent with momentum (Kingma and Ba, 2015). It uses the squared gradients to scale the learning rate like RMSprop and it takes advantage of momentum by using moving average of the gradient instead of gradient itself like SGD with momentum.

![1e-3 loss](https://drive.google.com/uc?id=1oBmilOBC-FZ61BrkjAOeTSQE-Ul-cvYg)

Final average SSIM: 0.60325

![SSIM vs Optimiser](https://drive.google.com/uc?id=1Mryjl3K4GS8018-HSqkVJIBeLBerdkjF)

![SSIM increase vs optimiser](https://drive.google.com/uc?id=1LIhBKoR4eJmutLT9rrqYGxKPs70_o4GH)

The model trained using Adam as an optimiser performs the best, hence we will use Adam for the optimiser.

### Loss function
**Mean Absolute Error - L1 Loss**

Mean absolute error is measured as the average of sum of absolute differences between predictions and actual observations. L1 loss is robust to outliers since it does not make use of square.

![Mean Absolute Error](https://drive.google.com/uc?id=1NwOXov_bT3O8WAxqHhSZ2AY7x4H1KvQs)

Final average SSIM: 0.60325

**Mean square error (MSE)**

Mean square error is measured as the average of squared difference between predictions and actual observations. However, due to squaring, predictions which are far away from actual values are penalized heavily in comparison to less deviated predictions. 

![Mean square error](https://drive.google.com/uc?id=1k0_diCFN3d8wWCfgdBg7RmrNmhZRLf2f)

Final average SSIM:0.60100

The L1 loss seems to be a better choice for this problem.

### Epoch
**epoch 20**: average ssim: 0.605277

![epoch 20 A](https://drive.google.com/uc?id=1g80jr3MbvTTc7tEpKhoUwxIepMeH-p2Q)
![epoch 20 B](https://drive.google.com/uc?id=15KkSqE-cVp2ASdz_2N_oICFoK1tImq1L)

**epoch 30**: average ssim: 0.607800

![epoch 30 A](https://drive.google.com/uc?id=1Euz0-ipl7oj9ig-CkD9Wh9_Z0ZtUyon4)
![epoch 30 B](https://drive.google.com/uc?id=1jaJEd91svK1Kp63LWhcUR2VWI-2cWDtS)

**epoch 40**: average ssim: 0.607928

![epoch 40 A](https://drive.google.com/uc?id=1Rw3ta0itk4AkFhlUQrTjPrHiUCZD3oHk)
![epoch 40 B](https://drive.google.com/uc?id=1QCkQvrELjlUQlEzDz7uMvDOFzscfws29)

### Drop out rate

The dropout is a regularization technique which deactivates a certain percentage of randomly chosen neurons while training. By deactivating the neurons, this technique prevents the model from overfitting the data. Then, while predicting, all of the neurons in the network is used.
The control of our experiment was 0.0 drop-out rate, which had an average SSIM value of 0.60325 for final average SSIM for validation set. We've exerimented with three other drop out rate, which were 0.1, 0.2, and 0.3.

- 0.1 drop out rate

![alt text](https://drive.google.com/uc?id=1WM_uMNn1txU9IfXkX8nfagxyYsd8O_GR)

Final Average SSIM: 0.560

- 0.2 drop out rate

![alt text](https://drive.google.com/uc?id=1IpHfiNTONJXERxuS9QhAcSEcNIc6BRZI)

Final Average SSIM: 0.500

- 0.3 drop out rate

![alt text](https://drive.google.com/uc?id=1SBH3FRICVKHSTfQMEIqeFrnPq3bKu4Ir)

Final Average SSIM: 0.520


![alt text](https://drive.google.com/uc?id=12XIdOfTz3U9Mi0a9QqrCzQFry7dtKzVB)


We can see that the increased drop out rate affects the SSIM value negatively. The SSIM value for dropout rate of 0.3 was 0.520, which is approximately 0.08 lower than our control. So, increasing dropout rate corresponded to a decrease in SSIM values. 

I think that our model is not complex enough to overfit our data, and we also have many training examples to prevent overfitting. Because of this, the dropout rate is not actually beneficial for SSIM values.

### Kernel Size

The kernel is the filter for the convolutional layer, which is useful to get the spatial information. We chose filter of 3x3 as our control. We tried two other sizes, which were 5x5 filter and 7x7 filter.

- 5x5 filter


![alt text](https://drive.google.com/uc?id=136lRNgltnnJNrxYhhT7vye_ZF10etOMB)

Final Average SSIM: 0.606

The SSIM for kernel size of 5 was similar to the SSIM of kernel size of 3. The SSIM for both were very close to 0.600. We tried to run the kernel size of 7, but we ran out of GPU and was extremely slow.

![alt text](https://drive.google.com/uc?id=14GOuZQq1d2s1ihK-wefyXj5b133sOIed)

### Batch-size
batch-size = 1

average ssim = 0.5399118

![batch-size 1A](https://drive.google.com/uc?id=1XsLIs-U4PYPkYXiBR_R4nyWVnD3y1Wae)
![batch-size 1B](https://drive.google.com/uc?id=1NeBfBLWpquyD9jyLfnqvYeWAD-FSiyY5)

batch-size = 5

average ssim = 0.60780

![batch-size 5A](https://drive.google.com/uc?id=1I7jRMic1IqB6v2U-cOhzLrPbr-TUpATz)
![batch-size 5B](https://drive.google.com/uc?id=1hQkcrGi1IfkSe9FnxobVoKg3XUu67bTN)

If increase batch-size over 5, GPU ran out.



# 5. Conclusion

We experimented to find out which value shows the best performance in each hyperparameter and which hyperparameter affect result a lot. In terms of the number of pooling layers, 5 pooling layers showed the best result (average SSIM: 0.0617) among 2, 3, 4, 5 pooling layers. However, it takes more time as we use more layers in our algorithm. It takes 20 minutes for 2 layers and as we increment the number of layers it takes 5 more minutes.

Four different learning rates (1e-2, 1e-3, 1e-4, 1e-5) were tested with 30 epochs, and 1e-4 showed the best performance (average SSIM: 0.6068). But the difference of SSIM score between 1e-4 and other learning rates was only 0.003 ~ 0.023 (0.3% ~ 2.3%). 

Three optimizers (Adagrad, RMSProp, Adam) were used and Adam optimizer produced the highest final average SSIM score (0.60325). SSIM score of Adagrad was lower than others (0.48103) but RMSProp was almost similar to Adam (0.60154).

We also experimented to see if the number of epochs influences the average SSIM and 40 epochs showed the highest mark (0.607928). However, for epochs, 20, 30, 40 epochs produced almost same average SSIM mark (20: 0.60587, 30: 0.60780) and because higher number of epochs normally takes longer time and produces better result, it is hard to say that 40 epochs works better than 20 and 30 and therefore we could say the number of epochs does not affect the result. 

Three dropout rates were used (0.1, 0.2, 0.3) and compare to the control of our experiment (0.0) SSIM value was reduced as the dropout rate goes up. (0.60325 -> 0.560, 0.500, 0.520). 

The experiment also carried out with two different batch sizes (1, 5), and the SSIM value of batch size 5 (0.60780) was higher than batch size 1 (0.53991). 

Concerning kernel size, no difference was found between kernel size 5 and kernel size 3. Finally, we tried to change the 'interpolate' mode of our code but because of the limitation of GPU capacity, this couldn’t be carried out. 

In short, we carried out our experiment by changing several hyperparameters and figured out that only the number of pooling layers, learning rate, type of optimizer, batch size, and dropout rate influence the SSIM value. 

The optimal hyperparameters were 5 pooling layers, learning rate of 1e-4, Adam optimizer, dropout rate of 0.0, and batch size of 5. Two hyperparameters, the number of epochs and kernel size, affected the SSIM value minimally. However, it should be noted that if the number of epoch is very low, like 1 or 2, it will affect the SSIM value because the model is not fully trained yet.


# 6. Description of Contribution

Woojin wrote the conclusion, carried out part of experiment (interpolate mode)

Shuhao Zhang focused on optimising the model, added the ssim part in the training process. Optimised the data loader to allow batch-sizes over one. Optimised the data visualisation. Gathered papers of MRI reconstruction. 

Monika Manuela Hengki focused on tuning some of the hyperparameters of the UNet model. She researched into how varying learning rates would affect the training, and carried out experiments with various learning rates. She also researched into various optimizer algorithms and tested out each of them to see which one is the most suitable for training the model.

Martijn Schendstok mainly focused on the programming outside of the model itself. He created the functions to save and load a (partly) trained model, code to continue training a model if it had to be interrupted, and the code which reconstructs the test data and saves the reconstructions as h5-files in the required manner. Besides these specific cases he worked on code readability and functionality in general.

Hyun Lee built the initial model, which performed very poorly compared to the UNet. Then, he tested the affect of various dropout rate and kernel sizes. In addition, he wrote parts of design and experiment section as well as laid this Jupyter Notebook.


Tra prepared the outline of the code to easily test different models. She also tests many models before using the Unet model to solve this problem. She also works on finding the influences of the number of pooling layers and the loss functions used to our final reconstruction to find the best value for our model.

# 7. References
Duchi, J., Hazan, E. and Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. Journal of Machine Learning Research, 12, pp.2121-2159.

Kingma, D. and Ba, J. (2015). Adam: A Method for Stochastic Optimization. In: 3rd International Conference for Learning Representations.

Ronneberger, O., Fischer, P. and Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In: MICCAI.

Tieleman, T. and Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude.

P.W.D. Charles, Project Title, (2013), GitHub repository, https://github.com/charlespwd/project-title

Zbontar, Jure et al. (2018). “fastMRI: An Open Dataset and Benchmarks for
Accelerated MRI”. In: arXiv: 1811.08839. GitHub: https://github.com/facebookresearch/fastMRI 
