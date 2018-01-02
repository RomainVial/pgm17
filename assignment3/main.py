import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from clustering import GaussianMixture, plot_ellipse
from scipy.stats import multivariate_normal

mpl.rcParams.update({'font.size': 30})


def load_data(type='train'):
    with open('data/EMGaussian.{}'.format(type)) as f:
        x = []
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            x_i = [float(line[0]), float(line[1])]
            x.append(x_i)
    return np.asarray(x)


def log_dot(a, log_p):
    log_a = np.log(a)
    max_log = np.amax(log_a + log_p)
    return max_log + np.log(np.sum(np.exp(log_a + log_p - max_log)))


def log_matrix_dot(A, log_p):
    res = np.zeros((log_p.size,))
    for k in range(res.size):
        res[k] = log_dot(A[k, :], log_p)
    return res


class HMM:
    def __init__(self, K, mu, sigma):
        self.K = K
        self.iterations = 5
        # Markov chain parameters
        self.pi = [0.25, 0.25, 0.25, 0.25]
        self.A = 1. / 6. * np.ones((4, 4)) + 2. / 6. * np.eye(4)
        # Gaussian parameters
        self.mu = mu
        self.sigma = sigma

    def _alpha_recursion(self, X):
        T = X.shape[0]
        log_alphas = np.zeros((T, self.K))
        log_multivariate_normal = lambda x: np.array(
            [multivariate_normal.logpdf(x, self.mu[i], self.sigma[i]) for i in range(self.K)])

        log_alphas[0, :] = np.log(self.pi) + log_multivariate_normal(X[0, :])

        for t in range(1, T):
            log_alphas[t, :] = log_matrix_dot(self.A, log_alphas[t - 1, :]) + log_multivariate_normal(X[t, :])

        return log_alphas

    def _beta_recursion(self, X):
        T = X.shape[0]
        log_betas = np.zeros((T, self.K))
        log_multivariate_normal = lambda x: np.array(
            [multivariate_normal.logpdf(x, self.mu[i], self.sigma[i]) for i in range(self.K)])

        log_betas[-1, :] = 0.
        for t in range(T - 2, -1, -1):
            log_betas[t, :] = log_matrix_dot(self.A, log_betas[t + 1, :] + log_multivariate_normal(X[t + 1, :]))

        return log_betas

    def _decoding(self, X):
        T = X.shape[0]

        # Compute alpha and beta recursion
        log_alphas = self._alpha_recursion(X)
        log_betas = self._beta_recursion(X)

        # Compute the log partition function
        max_log = np.amax(log_alphas + log_betas, axis=1, keepdims=True)
        log_part = max_log + np.log(np.sum(np.exp(log_alphas + log_betas - max_log), axis=1, keepdims=True))

        # Compute the probability of q_t given the observed variables
        log_gamma = log_alphas + log_betas - log_part

        # Compute the probability of (q_t, q_t+1) given the observed variables
        log_chsi = np.zeros((T - 1, self.K, self.K))
        log_multivariate_normal = lambda x, k: multivariate_normal.logpdf(x, self.mu[k], self.sigma[k])
        for t in range(T - 1):
            for i in range(self.K):
                for j in range(self.K):
                    log_chsi[t, i, j] = log_alphas[t, i] + log_betas[t + 1, j] + np.log(
                        self.A[i, j]) + log_multivariate_normal(X[t + 1, :], j) - log_part[t, 0]

        return np.exp(log_gamma), np.exp(log_chsi)

    def _compute_loglikelihood(self, X):
        gamma, chsi = self._decoding(X)

        log_multivariate_normal = lambda x: np.array(
            [multivariate_normal.logpdf(x, self.mu[i], self.sigma[i]) for i in range(self.K)])
        ll = np.sum(gamma * np.swapaxes(log_multivariate_normal(X), 0, 1)) + np.sum(
            chsi * np.log(self.A[np.newaxis, :, :]))
        return ll

    def fit(self, X_train, X_test, render=True):
        """
        Run Expectation-Maximization algorithm to learn the parameters
        """
        T = X_train.shape[0]
        ll = {'train': [], 'test': []}
        for step in range(self.iterations):
            # Compute log-likelihood
            ll['train'].append(self._compute_loglikelihood(X_train))
            ll['test'].append(self._compute_loglikelihood(X_test))

            # E-step
            gamma, chsi = self._decoding(X_train)

            # M-step
            self.pi = gamma[0, :]

            for i in range(self.K):
                # Update A
                for j in range(self.K):
                    self.A[i, j] = np.sum(chsi[:, i, j]) / np.sum(chsi[:, i, :])

                # Update mu and sigma
                self.mu[i] = np.sum(gamma[:, i, np.newaxis] * X_train, axis=0) / np.sum(gamma[:, i])

                # Update sigma
                self.sigma[i] = sum(
                    gamma[t, i] * np.outer(X_train[t, :] - self.mu[i], X_train[t, :] - self.mu[i]) for t in
                    range(T)) / np.sum(gamma[:, i])

        if render:
            self.plot_likelihood(ll)

    def _viterbi_decoding(self, X):
        T = X.shape[0]
        log_multivariate_normal = lambda x: np.array(
            [multivariate_normal.logpdf(x, self.mu[i], self.sigma[i]) for i in range(self.K)])

        log_v = np.log(self.pi) + log_multivariate_normal(X[0, :])
        paths = {k: {'log_v': log_v[k], 'states': [k]} for k in range(self.K)}

        for t in range(1, T):
            for k in range(self.K):
                prev_state, prev_log_v = paths[k]['states'][-1], paths[k]['log_v']
                log_v = log_multivariate_normal(X[t, :]) + np.log(self.A[prev_state, :]) + prev_log_v

                paths[k]['states'].append(np.argmax(log_v))
                paths[k]['log_v'] = np.amax(log_v)

        best_path_key = np.argmax([paths[key]['log_v'] for key in paths])
        return np.asarray(paths[best_path_key]['states'])

    def plot(self, X):
        colors = ['blue', 'red', 'purple', 'darkred']
        gamma, chsi = self._decoding(X)
        z = np.argmax(gamma, axis=-1)

        # Plot the data points, ellipses and centroids
        splot = plt.subplot()
        for k in range(self.K):
            plt.scatter(X[z == k, 0], X[z == k, 1], marker='^', c=colors[k])
            plot_ellipse(splot, self.mu[k], self.sigma[k], colors[k])
        plt.scatter(self.mu[:, 0], self.mu[:, 1], marker='x', c='white', s=400)
        plt.axis('equal')
        plt.show()
        plt.close()

    def plot_gamma(self, X, len=100, opt='proba'):
        gamma, _ = self._decoding(X)

        if opt == 'proba':
            f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
            ax1.plot(gamma[:len, 0])
            ax2.plot(gamma[:len, 1])
            ax3.plot(gamma[:len, 2])
            ax4.plot(gamma[:len, 3])
        elif opt == 'states':
            z = np.argmax(gamma, axis=-1)
            plt.plot(z, 'ro')

        plt.show()

    def plot_viterbi(self, X):
        colors = ['blue', 'red', 'purple', 'darkred']
        z = self._viterbi_decoding(X)

        # Plot the data points, ellipses and centroids
        splot = plt.subplot()
        for k in range(self.K):
            plt.scatter(X[z == k, 0], X[z == k, 1], marker='^', c=colors[k])
            plot_ellipse(splot, self.mu[k], self.sigma[k], colors[k])
        plt.scatter(self.mu[:, 0], self.mu[:, 1], marker='x', c='white', s=400)
        plt.axis('equal')
        plt.show()
        plt.close()

    def plot_viterbi_states(self, X, len=100, opt='proba'):
        states = self._viterbi_decoding(X)

        if opt == 'proba':
            f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
            ax1.plot(np.where(states[:len] == 0, 1, 0))
            ax2.plot(np.where(states[:len] == 1, 1, 0))
            ax3.plot(np.where(states[:len] == 2, 1, 0))
            ax4.plot(np.where(states[:len] == 3, 1, 0))
        elif opt == 'states':
            plt.plot(states, 'ro')

        plt.show()

    def plot_likelihood(self, ll):
        plt.plot(ll['test'], linewidth=3, label='Test')
        plt.plot(ll['train'], linewidth=3, label='Train')
        plt.title('Likelihood against the number of iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Likelihood')
        plt.legend()
        plt.show()
        plt.close()


if __name__ == '__main__':
    X_train = load_data()
    X_test = load_data(type='test')

    print "----- GMM -----"
    gm = GaussianMixture(4, 'general')
    gm.train(X_train, render=True)

    print "\n----- HMM -----"
    hmm = HMM(4, gm.mu, gm.sigma)

    # Plot gamma before training
    hmm.plot_gamma(X_test, opt='proba')

    # Learn the model
    hmm.fit(X_train, X_test, render=True)

    # Plot data with Viterbi decoding
    hmm.plot_viterbi(X_test)

    # Plot the marginal probabilities
    hmm.plot_gamma(X_test, opt='proba')

    # Plot the most likely states
    hmm.plot_gamma(X_test, opt='states')
    hmm.plot_viterbi_states(X_test, opt='states')
