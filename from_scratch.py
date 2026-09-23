import pandas as pd
import numpy as np

## Helpers
def stratify_from_scratch():
    pass

def impute_from_scratch():
    pass

def standardize_from_scratch():
    pass

def sigmoid_from_scratch(z):
    return np.array(1/(1+np.exp(-z)))

# def gradient_wrt_weights_of_linear(x):
#     '''
#     :param x: R^{nx1} array
#     :return: gradient of function (a,b) mapsto ax+b,  wehre  a in R^{1xn}  and  b in R
#     '''
#     grad_a = x
#     grad_b = 1
#     return grad_a, grad_b
#
# def grad_sigmoid_from_scratch(z):
#     return np.exp(z)/(1+np.exp(z))**2

# x = np.array([[1,1,1], [2,2,2]])
# print(len(x))
# print(np.sum(x))
# print(np.shape(x))
# print(type(np.shape(x)))
# print(np.shape(x)[0])

def train_test_split_from_scratch():
    pass




# class LogReg_Linear_from_scratch():
#     def __init__(self, X_dim, Y_dim=1, random_state = 42):
#         self.X_dim = X_dim
#         self.Y_dim = Y_dim
#         # self.rng = np.random.default_rng(random_state)
#         # self.A = rng.random(X_dim)
#         # self.B = rng.random(Y_dim)
#         ### Logistic regression does not require random initialization. Neural networks generally do because random initialization breaks symmetry between hidden units.
#         self.A = np.zeros(X_dim)
#         self.B = 0.0
#
#     def forward(self, x):
#         logits = self.A.T @ x + self.B
#         return logits
#
#
#     def _get_probabs_from_logits_from_scratch(self, z):
#         '''
#         :param z: np.array of shape (batch,), logits of batch
#         :return: np.array of shape (batch, 2),
#                  where each row is [P(class 0), P(class 1)]
#         '''
#
#         # z = np.array([z])
#         # print(z)
#         # print(np.shape(z))
#         # print(type(np.shape(z)))
#         # print(np.shape(z)[0])
#         #
#         # p_hat = sigmoid_from_scratch(z)
#         # print(p_hat)
#         #
#         # if np.shape(z)[0] == 0:
#         #     return np.array([1-p_hat, p_hat])
#         #
#         # return np.column_stack((1-p_hat, p_hat))
#         z = np.array(z)
#
#         p_hat = sigmoid_from_scratch(z)
#
#         if np.ndim(z) == 0:
#             return np.array([1-p_hat, p_hat])
#
#         return np.column_stack((1-p_hat, p_hat))
#
#     def _get_preds_from_logits_from_scratch(self, logits, t=0.5):
#         '''
#         :param probs: probability vector
#         :return: prediction
#         '''
#         logits = np.array(logits)
#
#         if np.ndim(logits) == 0:
#             probs_class_1 = self._get_probabs_from_logits_from_scratch(logits)[1]
#         else:
#             probs_class_1 = self._get_probabs_from_logits_from_scratch(logits)[:, 1]
#
#         y_hat = (probs_class_1 >= t).astype(int)
#
#         return y_hat
#
#     def _get_preds_from_probs_from_scratch(self, probs, t=0.5):
#         '''
#         :param probs: probability vector
#         :return: prediction
#         '''
#         probs = np.array(probs)
#
#         if np.ndim(probs) == 1:
#             probs_class_1 = probs[1]
#         else:
#             probs_class_1 = probs[:, 1]
#
#         y_hat = (probs_class_1 >= t).astype(int)
#
#         return y_hat
#
#
#
# def BCE_loss_logits_from_sratch(model, y, logits):
#     y = np.array(y)
#     probs = model._get_probabs_from_logits_from_scratch(logits)
#
#     eps = 1e-12
#
#     if np.ndim(probs) == 1:
#         p_hat = probs[1]
#         p_hat = np.clip(p_hat, eps, 1-eps)
#         loss = -(y*np.log(p_hat) + (1-y)*np.log(1-p_hat))
#     else:
#         batch = len(y)
#         p_hat = probs[:, 1]
#         p_hat = np.clip(p_hat, eps, 1-eps)
#         loss = - np.sum((y*np.log(p_hat) + (1-y)*np.log(1-p_hat))) / batch
#
#     return loss
#
#
# def grad_of_loss_function_logits_from_scratch(model, x, y, logits):
#     x = np.array(x)
#     y = np.array(y)
#     logits = np.array(logits)
#
#     probs = model._get_probabs_from_logits_from_scratch(logits)
#     p_hat = probs[:, 1]
#
#     batch = len(y)
#
#     grad_a = x @ (p_hat - y) / batch
#     grad_b = np.sum(y-p_hat) / batch
#
#     return grad_a, grad_b
#
# def grad_of_loss_function_probs_from_scratch(model, x, y, probs):
#     x = np.array(x)
#     y = np.array(y)
#     probs = np.array(probs)
#
#     p_hat = probs[:, 1]
#
#     batch = len(y)
#
#     grad_a = x @ (p_hat - y) / batch
#     grad_b = np.sum(y - p_hat) / batch
#
#     return grad_a, grad_b
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






