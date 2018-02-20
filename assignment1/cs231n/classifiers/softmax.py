import numpy as np
from random import shuffle
from past.builtins import xrange

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

  n_examples = X.shape[0]
  n_classes = W.shape[1]
  
  # looping over training data
  for i in range(n_examples):
      # computing scores for single example
      s = X[i] @ W
      # Dividing large numbers can be numerically unstable, so it is important to use a normalization trick.
      # http://cs231n.github.io/linear-classify/#softmax
      s -= np.max(s)
      # Exp of scores vector
      ex_s = np.exp(s)
      # Exp of correct score
      cor_s = np.exp(s[y[i]])
      loss += -np.log(cor_s/np.sum(ex_s))
      
      # Computing gradient
      grad = X[i] / np.sum(ex_s)
      for j in range(n_classes):
        dW[:, j] += grad * (np.exp(s[j]))
      dW[:, y[i]] -= X[i]

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss /= n_examples
  loss += reg * np.sum(W ** 2)
  dW /= n_examples
  dW += reg * 2 * W

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
  
  n_examples = X.shape[0]
  n_classes = W.shape[1]
  
  # computing scores
  S = X @ W
  # Dividing large numbers can be numerically unstable, so it is important to use a normalization trick.
  # http://cs231n.github.io/linear-classify/#softmax
  S = S - S.max(axis=1, keepdims=True)
  # Exp of scores vector
  EX_S = np.exp(S)
  # Sum of exp in each row
  EX_SUM = np.sum(EX_S, axis=1)
  # Exp for true classes
  EX_TRUE = EX_S[np.arange(n_examples), y]
  # Loss
  loss = (-np.log(EX_TRUE / EX_SUM)).sum() / n_examples + reg * np.sum(W ** 2)
  
  # Computing gradient
  grad = (X.T / EX_SUM)
  dW += (grad @ EX_S)
  for i in range(n_classes):
    dW[:, i] -= X[np.argwhere(y == i)].sum(axis=0).squeeze()

  dW /= n_examples
  dW += reg * 2 * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

