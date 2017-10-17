import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

matplotlib.rcParams.update({'font.size': 15})

def load_data(idx='A', type='train'):
    with open('classification_data_HWK1/classification{}.{}'.format(idx, type)) as f:
        x = []
        y = []
        for line in f.readlines():
            x_i = [float(line.split('\t')[0]), float(line.split('\t')[1])]
            y_i = int(float(line.split('\t')[2].strip('\n')))
            x.append(x_i)
            y.append(y_i)
    return np.asarray(x), np.asarray(y)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def plot_curve(X, Y, idx, models, name):
    plt.scatter(X[Y > 0.5, 0], X[Y > 0.5, 1], marker='^', c='red', label='1')
    plt.scatter(X[Y < 0.5, 0], X[Y < 0.5, 1], marker='^', c='blue', label='0')
    plt.title('Training set {}'.format(idx))
    legend = []
    for model in models:
        model.plot_curve(legend)
    first_legend = plt.legend(handles=legend, loc=1)
    plt.gca().add_artist(first_legend)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend(loc=3)
    plt.savefig('output/dataset_{}_{}.png'.format(idx, name))
    plt.close()


def confusion_matrix(Y, Y_hat, threshold=0.5):
    tp = np.sum((Y > 0.5) * (Y_hat > threshold))
    fn = np.sum((Y > 0.5) * (Y_hat <= threshold))
    tn = np.sum((Y < 0.5) * (Y_hat <= threshold))
    fp = np.sum((Y < 0.5) * (Y_hat > threshold))
    print '{:15} {:5} {:5}'.format('True/Predicted',1, 0)
    print '{:15} {:5} {:5}'.format('1', tp, fn)
    print '{:15} {:5} {:5}'.format('0', fp, tn)
    print 'Error rate: {:.4f}\n'.format(float(fp+fn)/Y.shape[0])


class LinearRegression:
    def __init__(self):
        self.w_hat = None
        self.sigma = None

    def __str__(self):
        w_hat = 'w_hat: ' + repr(self.w_hat)
        sigma = 'sigma: ' + repr(self.sigma)
        return '\n'.join([w_hat, sigma])

    def train(self, X, Y):
        A = np.hstack([X, np.expand_dims(np.ones(X.shape[0]), axis=1)])
        A_T = np.transpose(A)
        self.w_hat = np.dot(np.dot(np.linalg.inv(np.dot(A_T, A)), A_T), Y)
        self.sigma = np.mean(np.square(Y - A.dot(self.w_hat)))

    def plot_curve(self, legend, start=-4., end=4.):
        x_1 = np.arange(start, end, 0.2)
        x_2 = - (self.w_hat[0] / self.w_hat[1]) * x_1 + (0.5 - self.w_hat[2]) / self.w_hat[1]

        plt.plot(x_1, x_2, 'orange')
        legend.append(mlines.Line2D([], [], color='orange', label='Linear Regression'))

    def inference(self, X):
        A = np.hstack([X, np.expand_dims(np.ones(X.shape[0]), axis=1)])
        return A.dot(self.w_hat)


class LogisticRegression:
    def __init__(self):
        self.w_hat = None

    def __str__(self):
        return 'w_hat: ' + repr(self.w_hat)

    def irls_step(self, X, Y):
        X_T = np.transpose(X)
        eta = sigmoid(np.dot(X, self.w_hat))
        D = np.diag(eta * (1 - eta))
        P = np.linalg.inv(np.dot(X_T, np.dot(D, X)))

        self.w_hat += np.dot(P, np.dot(X_T, Y - eta))

    def train(self, X, Y):
        A = np.hstack([X, np.expand_dims(np.ones(X.shape[0]), axis=1)])
        self.w_hat = np.asarray([0., 0., 0.])
        for i in range(5):
            self.irls_step(A, Y)

    def plot_curve(self, legend, start=-4., end=4.):
        x_1 = np.arange(start, end, 0.2)
        x_2 = - (self.w_hat[0] / self.w_hat[1]) * x_1 - (self.w_hat[2] / self.w_hat[1])

        plt.plot(x_1, x_2, 'purple')
        legend.append(mlines.Line2D([], [], color='purple', label='Logistic Regression'))

    def inference(self, X):
        A = np.hstack([X, np.expand_dims(np.ones(X.shape[0]), axis=1)])
        return sigmoid(np.dot(A, self.w_hat))


class LDA:
    def __init__(self):
        self.mu_1 = None
        self.mu_0 = None
        self.sigma = None
        self.sigma_inv = None
        self.pi = None

    def __str__(self):
        pi = 'pi: ' + repr(self.pi)
        mu_0 = 'mu_0: ' + repr(self.mu_0)
        mu_1 = 'mu_1: ' + repr(self.mu_1)
        sigma = 'sigma: ' + repr(self.sigma)
        return '\n'.join([pi, mu_0, mu_1, sigma])

    def train(self, X, Y):
        self.mu_1 = np.mean(X[Y>0.5,:], axis=0)
        self.mu_0 = np.mean(X[Y<0.5,:], axis=0)
        self.pi = np.sum(Y > 0.5) / float(Y.shape[0])

        X_0 = X[Y<0.5,:]
        S_0 = (X_0 - self.mu_0).transpose().dot(X_0 - self.mu_0) / float(X_0.shape[0])
        X_1 = X[Y>0.5,:]
        S_1 = (X_1 - self.mu_1).transpose().dot(X_1 - self.mu_1) / float(X_1.shape[0])
        self.sigma = self.pi * S_1 + (1-self.pi) * S_0
        self.sigma_inv = np.linalg.inv(self.pi * S_1 + (1-self.pi) * S_0)

    def plot_curve(self, legend, start=-5., end=5.):
        x_min, x_max = start, end
        y_min, y_max = start, end
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        z = self.inference(np.c_[xx.ravel(), yy.ravel()])
        z = np.reshape(z, xx.shape)

        plt.contour(xx, yy, z, [0.5], linewidths=1., colors='green')
        legend.append(mlines.Line2D([], [], color='green', label='LDA'))

    def inference(self, X):
        s_0 = np.log(1 - self.pi) + X.dot(self.sigma_inv).dot(self.mu_0) - 0.5 * (self.mu_0).transpose().dot(self.sigma_inv).dot(self.mu_0)
        s_1 = np.log(self.pi) + X.dot(self.sigma_inv).dot(self.mu_1) - 0.5 * (self.mu_1).transpose().dot(self.sigma_inv).dot(self.mu_1)

        return 1. / (1. + np.exp(s_0 - s_1))


class QDA:
    def __init__(self):
        self.mu_1 = None
        self.mu_0 = None
        self.sigma_1 = None
        self.sigma_1_inv = None
        self.sigma_0 = None
        self.sigma_0_inv = None
        self.pi = None

    def __str__(self):
        pi = 'pi: ' + repr(self.pi)
        mu_0 = 'mu_0: ' + repr(self.mu_0)
        mu_1 = 'mu_1: ' + repr(self.mu_1)
        sigma_0 = 'sigma_0: ' + repr(self.sigma_0)
        sigma_1 = 'sigma_1: ' + repr(self.sigma_1)
        return '\n'.join([pi, mu_0, mu_1, sigma_0, sigma_1])

    def train(self, X, Y):
        self.mu_1 = np.mean(X[Y>0.5,:], axis=0)
        self.mu_0 = np.mean(X[Y<0.5,:], axis=0)
        self.pi = np.sum(Y > 0.5) / float(Y.shape[0])

        X_0 = X[Y<0.5,:]
        self.sigma_0 = (X_0 - self.mu_0).transpose().dot(X_0 - self.mu_0) / float(X_0.shape[0])
        self.sigma_0_inv = np.linalg.inv(self.sigma_0)
        X_1 = X[Y>0.5,:]
        self.sigma_1 = (X_1 - self.mu_1).transpose().dot(X_1 - self.mu_1) / float(X_1.shape[0])
        self.sigma_1_inv = np.linalg.inv(self.sigma_1)

    def plot_curve(self, legend, start=-5., end=5.):
        x_min, x_max = start, end
        y_min, y_max = start, end
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        z = self.inference(np.c_[xx.ravel(), yy.ravel()])
        z = np.reshape(z, xx.shape)

        plt.contour(xx, yy, z, [0.5], linewidths=1., colors='k')
        legend.append(mlines.Line2D([], [], color='black', label='QDA'))


    def inference(self, X):
        s_0 = np.log(1 - self.pi) - 0.5 * np.log(np.linalg.det(self.sigma_0)) - 0.5 * np.sum((X - self.mu_0).dot(self.sigma_0_inv) * (X - self.mu_0), axis=1)
        s_1 = np.log(self.pi) - 0.5 * np.log(np.linalg.det(self.sigma_1)) - 0.5 * np.sum((X - self.mu_1).dot(self.sigma_1_inv) * (X - self.mu_1), axis=1)

        return 1. / (1. + np.exp(s_0 - s_1))


if __name__=='__main__':
    for idx in ['A', 'B', 'C']:
        print 'Analyzing dataset {}'.format(idx)
        X_train, Y_train = load_data(idx=idx, type='train')
        X_test, Y_test = load_data(idx=idx, type='test')

        # QDA
        print '--Quadratic Discriminant Analysis--'
        qda = QDA()
        qda.train(X_train, Y_train)
        print qda
        print 'Train Classification'
        confusion_matrix(Y_train, qda.inference(X_train))
        print 'Test Classification'
        confusion_matrix(Y_test, qda.inference(X_test))
        plot_curve(X_train, Y_train, idx, [qda], 'qda')

        # LDA
        print '--Linear Discriminant Analysis--'
        lda = LDA()
        lda.train(X_train, Y_train)
        print lda
        print 'Train Classification'
        confusion_matrix(Y_train, lda.inference(X_train))
        print 'Test Classification'
        confusion_matrix(Y_test, lda.inference(X_test))
        plot_curve(X_train, Y_train, idx, [lda], 'lda')

        # Linear Regression
        print '--Linear Regression--'
        lin_reg = LinearRegression()
        lin_reg.train(X_train, Y_train)
        print lin_reg
        print 'Train Classification'
        confusion_matrix(Y_train, lin_reg.inference(X_train))
        print 'Test Classification'
        confusion_matrix(Y_test, lin_reg.inference(X_test))
        plot_curve(X_train, Y_train, idx, [lin_reg], 'lin_reg')

        # Logistic Regression
        print '--Logistic Regression--'
        log_reg = LogisticRegression()
        log_reg.train(X_train, Y_train)
        print log_reg
        Y_test_hat = log_reg.inference(X_test)
        print 'Train Classification'
        confusion_matrix(Y_train, log_reg.inference(X_train))
        print 'Test Classification'
        confusion_matrix(Y_test, log_reg.inference(X_test))
        plot_curve(X_train, Y_train, idx, [log_reg], 'log_reg')

        # Plot all classification curves
        plot_curve(X_train, Y_train, idx, [lin_reg, log_reg, lda, qda], 'all')
