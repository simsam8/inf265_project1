---
title: 'Project 1: Backpropagation and Gradient Descent'
author:
- Simon Vedaa
- Sebastian Røkholt
numbersections: true
---

# Approach and Design choices

## Backpropagation
<!-- Todo -->

## Gradient descent

### Setup

- Set seed with value: 256
- Check if cuda is available
- Set default datatype for pytorch as double
- The MyMLP model architecture is created according to the task description.


### Loading data

The load_CIFAR2 function loads CIFAR10 data with only the labels airplane and bird.
The default train/validation split is 90/10, and is what we have used when training.

### Preprocessing

The only preprocessing step used was normalization.


### Training

Both training functions follows the same standard procedure for fitting a model.
For a given amount of epochs; forward pass through the network to calculate loss,
then compute the gradients with backpropagation, then update the parameters with 
the learning learning rate.

The functions differs in the parameter update step. The train function uses pytorch's 
SGD optimizer while the train_manual_update manually implements SGD with optional momentum and
weight decay.


### Model comparison

To compare models with different hyperparameters and ensure both training functions 
computes the same loss, we created the function aptly named "compare_models".

It defines the different hyperparameter combinations, and trains the models with both 
training functions. Each model is reseeded before training to ensure the same results for 
both training functions. The function returns the trained models, along with their validation
accuracies and hyperparameters.


### Model Evaluation

<!-- Todo -->
<!-- Check for class imbalance -->

Accuracy is the chosen performance measure. The model with the highest validation accuracy is selected,
and then evaluated on the test set.



# Q&A

a) Which PyTorch method(s) correspond to the tasks described in section 2?
    - In general the backward() method in a pytorch Tensor is responsible for calculating the gradients. 
    More specifically, the backward() is usually called on the loss object.
    This the backward() method corresponds to the tasks described in section 2.

b) Cite a method used to check whether the computed gradient of a function seems correct.
    Briefly explain how you would use this method to check your computed gradients in
    section 2.
    - The shape of the computed gradients should match the shape of respective parameter.
        E.x. if the shape of w1 is [4,2], then the shape of dl_w1 should be [4, 2]
    - In section 2, we would make sure the shape of each Tensor matches.

c) Which PyTorch method(s) correspond to the tasks described in section 3, question 4.?
    - The step method in the optimizer object is equivalent with the manual parameter update.

d) Briefly explain the purpose of adding momentum to the gradient descent algorithm.
    - The purpose of adding momentum to the gradient descent algorithm, is so that the optimizer
        "remembers" the direction it was previously moving. Benefits include faster convergence,
        less oscillations due to noise, or complex landscapes, escape local minima, and improve 
        exploration.

e) Briefly explain the purpose of adding regularization to the gradient descent algorithm.
    - The purpose of adding regularization is to penalize the algorithm for overfitting and 
    improve generalization performance.

f) Report the different parameters used in section 3, question 8., the selected parameters in
    question 9. as well as the evaluation of your selected model.
    <!-- Todo -->

g) Comment your results. In case you do not get expected results, try to give potential
    reasons that would explain why your code does not work and/or your results differ. 
    <!-- Todo -->
