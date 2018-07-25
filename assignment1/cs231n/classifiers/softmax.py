import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
    scores = X[i].dot(W)
    # print(scores.shape)
    # scores -= np.max(scores)
    correct_class_score = scores[y[i]]
    exp_sum = np.sum(np.exp(scores))
    loss += np.log(exp_sum) - correct_class_score
    dW[:, y[i]] -= X[i]
    for j in range(num_class):
        dW[:, j] += (np.exp(scores[j]) / exp_sum) * X[i]
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[0]
  
  scores = X.dot(W)
  exp_scores = np.reshape(np.exp(scores), (num_train, -1))

  sum_exp_scores = np.reshape(np.sum(exp_scores, axis=1), (num_train, -1))
  correct_class_score = scores[np.arange(num_train), y]
  loss = np.sum(np.sum(np.log(sum_exp_scores)) - np.sum(correct_class_score))
  
 
  score_prob = exp_scores / sum_exp_scores
  y_real_class = np.zeros_like(score_prob)
  y_real_class[range(num_train), y] = 1.0
  dW += np.dot(X.T, score_prob - y_real_class) / num_train
  
  
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

