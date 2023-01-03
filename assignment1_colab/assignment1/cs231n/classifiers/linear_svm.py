from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # dW = np.zeros(W.shape)  # initialize the gradient as zero

    # # compute the loss and the gradient
    # num_classes = W.shape[1]
    # num_train = X.shape[0]
    # loss = 0.0
    # for i in range(num_train):
    #     scores = X[i].dot(W)
    #     correct_class_score = scores[y[i]]
    #     for j in range(num_classes):
    #         if j == y[i]:
    #             continue
    #         margin = scores[j] - correct_class_score + 1  # note delta = 1
    #         if margin > 0:
    #             loss += margin

    # # Right now the loss is a sum over all training examples, but we want it
    # # to be an average instead so we divide by num_train.
    # loss /= num_train

    # # Add regularization to the loss.
    # loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW = np.zeros(W.shape)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        wrong_class_count = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                wrong_class_count += 1
                # for l in range(W.shape[0]):
                #     dW[l][j] += X[i][l]
                #     dW[l][y[i]] -= X[i][l]
        dW[:, y[i]] -= wrong_class_count * X[i]

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += reg * np.sum(2 * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    delta = 1

    scores = X.dot(W)
    num_train_interval = np.arange(num_train)
    correct_class_score = scores[num_train_interval, y]
    correct_class_score = np.reshape(correct_class_score, (num_train, -1))
    margins = np.maximum(0, scores - correct_class_score + delta)

    margins[num_train_interval, y] = 0

    loss += margins[margins > 0].sum()
    loss /= num_train
    loss += reg * np.sum(W*W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    wrong_classes = np.zeros(margins.shape)
    wrong_classes[margins > 0] = 1
    dW += X.T.dot(wrong_classes)

    wrong_classes_sum = -np.sum(wrong_classes, axis=1)
    wrong_classes = np.zeros(margins.shape)
    wrong_classes[num_train_interval, y] = wrong_classes_sum
    dW += X.T.dot(wrong_classes)

    dW /= num_train
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
