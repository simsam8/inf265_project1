---
title: 'Project 1: Backpropagation and Gradient Descent'
author:
- Simon Vedaa
- Sebastian Røkholt
numbersections: true
---

# Approach and Design choices

## Backpropagation
The function ``backpropagation(model, y_true, y_pred)`` is a fully vectorized implementation of the backpropogation algorithm that computes:

- $\frac{\partial L}{\partial w^{[l]}_{i,j}}$ and stores them in the model class' weight attribute ``model.dL_dw[l][i,j]`` for $l \in [1 .. L]$ 
- $\frac{\partial L}{\partial b^{[l]}_{j}}$ and stores them in the model class' weight attribute ``model.dL_db[l][j]`` for $l \in [1 .. L]$ 

The function assumes that ``model`` is an instance of the ``MyNet`` class.

The aim of this implementation is to be functionally equivalent to Pytorch's `loss.backward()` method. It accepts Pytorch Tensors as inputs for `y_true`, `y_pred` and the model's parameters. The backpropagation algorithm is then run directly on Pytorch Tensors without conversion to Numpy arrays. The implementation utilizes the Pytorch matrix multiplication method `.mm()` to calculate $\frac{\partial L}{\partial a}$ and $\frac{\partial L}{\partial w}$, which is very computationally efficient because it takes full advantage of PyTorch's optimized backend operations. 

In accordance with the backpropagation algorithm, we iterate over the layers in reverse order such that the gradients are calculated starting from the output layer and propagated backward through the network. The layer-by-layer approach, coupled with the calculation of the derivative of the activation functions (`model.df[l](model.z[l])`), ensures that the implementation can handle networks with multiple layers and different activation functions.

The optional `verbose` parameter allows for printing debugging information related to the dimensionality of derivatives and matrix multiplications, which helped us troubleshoot the code during the development process and understand how the gradients flow through the network. 

The use of the `torch.no_grad()` context ensures that the gradients are not automatically computed by PyTorch for operations within its scope. This is important to avoid an accumulation of gradient computations that are not necessary for the forward pass or the gradient update step itself.

However, though it might seem like our implementation of backpropagation converges, there might be a logical error that prevents us from reaching an actual minima. Without comparing the calculated gradient to something we know is correct, we don't actually know whether the implementation is converging to an actual minimum. Therefore, we have compared the calculated gradients with the output from Pytorch's `loss.backward()`. Alternatively, we could have used the definition of the derivative to check whether the calculated gradients seem correct. This method is called **gradient checking**. The definition of the derivative is: 

$f'(x) = \frac{f(x + \epsilon) - f(x)}{\epsilon}$, where $\epsilon$ is some small number. 

In the case of backpropagation, the definition could be applied like this: 

$\frac{\partial L}{\partial \theta} = \frac{f(\theta + \epsilon) - f(\theta)}{\epsilon}$, where $\theta$ are the weights and biases for a layer, and $f(\theta)$ is the activation for the same layer. 

So we could, for a subset of the data or a pre-specified layer, compare the results from:
1) Running the backpropagation implementation for parameters $\theta$ and $\theta$ + $\epsilon$
2) Using the definition of the derivative: Set $\epsilon$ to a small number and calculate the result using the formula.

The difference between the two calculations should be less than $\epsilon^2$. Performing these two calculations is very computationally expensive, but it could be useful to debug the training algorithm. At every layer, we could include a gradient checking option that runs the additional computations, does the comparision, and raises an error when the calculated gradient is incorrect. 

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
