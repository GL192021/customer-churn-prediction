import pandas as pd
import numpy as np


def sigmoid_from_scratch(z):
    return np.array(1/(1+np.exp(-z)))


class LogReg_Linear_from_scratch():
    '''
    Basic dimensions:           X          (N, d)
                                A          (d,)
                                B          scalar

                                logits     (N,)
                                p_hat      (N,)
                                Y          (N,)

                                grad_A     (d,)
                                grad_B     scalar
    '''
    def __init__(self):
        self.A = None
        self.B = None
        self.losses = []


    def forward(self, x):
        logits = np.dot(self.A, x.T) + self.B
        return logits


    def get_prob_class_1_from_logits_from_scratch(self, z):
        '''
        :param z: np.array of shape (batch,), logits of batch
        :return: np.array of shape (batch,),
                 which represents P(class 1)
        '''
        z = np.array(z)

        p_hat = sigmoid_from_scratch(z)

        return p_hat

    def get_preds_from_logits_from_scratch(self, logits, t=0.5):
        '''
        :param probs: probability vector
        :return: prediction
        '''
        logits = np.array(logits)

        probs_class_1 = self.get_prob_class_1_from_logits_from_scratch(logits)

        y_hat = (probs_class_1 >= t).astype(int)

        return y_hat

    def get_preds_from_probs_from_scratch(self, probs_class_1, t=0.5):
        '''
        :param probs: probability vector
        :return: prediction
        '''
        probs_class_1 = np.array(probs_class_1)

        y_hat = (probs_class_1 >= t).astype(int)

        return y_hat


    def BCE_loss_logits_from_sratch(self, y, logits):
        y = np.array(y)
        probs_class_1 = self.get_prob_class_1_from_logits_from_scratch(logits)

        eps = 1e-12

        probs_class_1 = np.clip(probs_class_1, eps, 1-eps)

        batch = len(y)

        loss = - np.sum((y*np.log(probs_class_1) + (1-y)*np.log(1-probs_class_1))) / batch

        return loss


    def grad_of_loss_function_logits_from_scratch(self, x, y, logits):
        x = np.array(x)
        y = np.array(y)
        logits = np.array(logits)

        p_hats = self.get_prob_class_1_from_logits_from_scratch(logits)

        batch = len(y)

        grad_a = -(x.T @ (y - p_hats) / batch)
        grad_b = -(np.sum(y - p_hats) / batch)

        return grad_a, grad_b

    def grad_of_loss_function_probs_from_scratch(self, x, y, probs_class_1):
        x = np.array(x)
        y = np.array(y)
        p_hats = np.array(probs_class_1)

        batch = len(y)

        grad_a = -(x.T @ (y - p_hats) / batch)
        grad_b = -(np.sum(y - p_hats) / batch)

        return grad_a, grad_b


    def step(self, grad_A, grad_B, learning_rate):
        self.A -= learning_rate * grad_A
        self.B -= -learning_rate * grad_B

    def fit(self, X, Y, epochs, lr):
        X_dim = X.shape[1]
        self.A = np.zeros(X_dim)
        self.B = 0.0

        for epoch in range(epochs):
            logits = self.forward(X)

            epoch_loss = self.BCE_loss_logits_from_sratch(Y, logits)

            grad_a, grad_b = self.grad_of_loss_function_logits_from_scratch(X, Y, logits)

            self.step(grad_A=grad_a, grad_B=grad_b, learning_rate=lr)

            self.losses.append(epoch_loss)






