---
title: 'Project 1: Backpropagation and Gradient Descent'
author:
- Simon Vedaa
- Sebastian Røkholt
numbersections: true
---

# Approach and Design choices

# Q&A

a) Which PyTorch method(s) correspond to the tasks described in section 2?
    - The backward() method in the loss object correspond to the tasks described in section 2.

b) Cite a method used to check whether the computed gradient of a function seems correct.
    Briefly explain how you would use this method to check your computed gradients in
    section 2.

c) Which PyTorch method(s) correspond to the tasks described in section 3, question 4.?
    - The step method in the optimizer is equivalent with the manual parameter update.

d) Briefly explain the purpose of adding momentum to the gradient descent algorithm.
    - The purpose of adding momentum to the gradient descent algorithm, is so that the optimizer
        "remembers" the direction it was previously moving. Benefits include faster convergence,
        less oscillations due to noise, or complex landscapes, escape local minima, and improve 
        exploration.

e) Briefly explain the purpose of adding regularization to the gradient descent algorithm.
    - The purpose of adding regularization is to penalize the algorithm for overfitting.

f) Report the different parameters used in section 3, question 8., the selected parameters in
    question 9. as well as the evaluation of your selected model.

g) Comment your results. In case you do not get expected results, try to give potential
    reasons that would explain why your code does not work and/or your results differ. 
