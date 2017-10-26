import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import copy

def load_data(type='train'):
    with open('data/EMGaussian.{}'.format(type)) as f:
        x = []
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            x_i = [float(line[0]), float(line[1])]
            x.append(x_i)
    return np.asarray(x)

def plot_ellipse(splot, mean, cov, color):
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                              180 + angle, facecolor=color,
                              edgecolor='yellow',
                              linewidth=2, zorder=2)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)


class KMeans:
    # Do several random initializations
    def __init__(self, K):
        self.K = K
        self.mu = None
        self.maxstep = 100

    def distortion(self, X, z, mu):
        J = 0
        for k in range(self.K):
            J += np.sum(np.square(X[z == k,:] - mu[k,:]))
        return J

    def plot_distortion(self, J):
        plt.plot(J)
        plt.title('Distortion against the number of iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Distortion')
        plt.show()
        plt.close()

    def train(self, X, render=False):
        J = []
        mu = np.random.rand(self.K, X.shape[1])
        mu_old = copy.copy(mu)
        term = False
        step = 0

        while step < self.maxstep and not term:
            distances = np.sum(np.square(np.expand_dims(X, axis=1) - mu), axis=-1)
            z = np.argmin(distances, axis=-1)
            J.append(self.distortion(X, z, mu))
            for k in range(self.K):
                mu[k] = np.sum(X[z == k,:], axis=0) / (np.sum(z == k) + 1e-12)
            term = np.all(np.equal(mu, mu_old))
            mu_old = copy.copy(mu)
            step += 1
        self.mu = mu

        if render:
            print 'Fitted after {} iterations'.format(step)
            print 'Final distortion : J = {:.2f}'.format(J[-1])
            self.plot_distortion(J)


    def inference(self, X):
        distances = np.sum(np.square(np.expand_dims(X, axis=1) - self.mu), axis=-1)
        return np.argmin(distances, axis=-1)

    def plot(self, X):
        colors = ['blue', 'red', 'green', 'purple']
        z = self.inference(X)
        for k in range(self.K):
            plt.scatter(X[z == k,0], X[z == k,1], marker='^', c=colors[k])
        plt.scatter(self.mu[:, 0], self.mu[:, 1], marker='*', c='orange', s=200)
        plt.axis('equal')
        plt.show()
        plt.close()


class GaussianMixture:
    def __init__(self, K, covariance_hypothesis='spherical'):
        self.K = K
        self.covariance_hypothesis = covariance_hypothesis
        if self.covariance_hypothesis == 'spherical':
            self.compute_tau = self._compute_tau_spherical
            self.compute_sigma = self._compute_sigma_spherical
        elif self.covariance_hypothesis == 'general':
            self.compute_tau = self._compute_tau_general
            self.compute_sigma = self._compute_sigma_general
        self.pi = None
        self.mu = None
        self.sigma = None

    def _compute_sigma_spherical(self, X, tau, mu):
        #return 0.5 * np.sum(tau * ((X-mu) * (X-mu)).sum(axis=1)) / np.sum(tau)
        S = self._compute_sigma_general(X, tau, mu)
        return 0.5 * (S[0,0] + S[1,1])

    def _compute_tau_spherical(self, X, pi, mu, sigma):
        #return (pi / sigma**2) * np.exp(-0.5 * (1. / sigma**2) * np.sum(np.square(X - mu), axis=-1))
        return self._compute_tau_general(X, pi, mu, sigma * np.eye(2))

    def _compute_sigma_general(self, X, tau, mu):
        A = np.sqrt(np.expand_dims(tau, axis=-1)) * (X - mu)
        return A.transpose().dot(A) / np.sum(tau)

    def _compute_tau_general(self, X, pi, mu, sigma):
        tau = np.zeros(X.shape[0])
        A = np.linalg.inv(sigma)
        for i in range(X.shape[0]):
            tau[i] = (pi / np.sqrt(np.linalg.det(sigma))) * np.exp(-0.5 * (X[i, :] - mu).transpose().dot(A).dot(X[i, :] - mu))
        return tau

    def initialize_with_kmeans(self, X):
        # Initialize mu with K-Means
        kmeans = KMeans(self.K)
        kmeans.train(X)
        mu = kmeans.mu

        # Initialize sigma and pi using cluster assignment of K-Means
        z = kmeans.inference(X)
        pi = np.bincount(z).astype(float) / z.size
        if self.covariance_hypothesis == 'spherical':
            sigma = np.zeros((self.K,))
        elif self.covariance_hypothesis == 'general':
            sigma = np.zeros((self.K, X.shape[1], X.shape[1]))

        for k in range(self.K):
            sigma[k] = self.compute_sigma(X, z==k, mu[k])

        return pi, mu, sigma

    def likelihood(self, X, pi, mu, sigma):
        ll = 0
        for k in range(self.K):
            ll += np.sum(self.compute_tau(X, pi[k], mu[k], sigma[k]))
        return ll

    def plot_likelihood(self, ll):
        plt.plot(ll)
        plt.title('Likelihood against the number of iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Likelihood')
        plt.show()
        plt.close()

    def train(self, X, render=True):
        ll = []
        pi, mu, sigma = self.initialize_with_kmeans(X)
        tau = np.zeros((X.shape[0], self.K))

        for step in range(15): # Convergence ?
            ll.append(self.likelihood(X, pi, mu, sigma))

            # Compute soft assignment to cluster k, tau[i,k]
            for k in range(self.K):
                tau[:,k] = self.compute_tau(X, pi[k], mu[k], sigma[k])
            tau = tau / np.sum(tau, axis=-1, keepdims=True)

            # Compute estimates for pi, mu, sigma based on the soft assignment
            pi = np.mean(tau, axis=0)
            for k in range(self.K):
                mu[k] = np.sum(np.expand_dims(tau[:,k], axis=-1) * X, axis=0) / np.sum(tau[:,k])
                sigma[k] = self.compute_sigma(X, tau[:,k], mu[k])

        self.pi = pi
        self.mu = mu
        self.sigma = sigma

        if render:
            print 'Final likelihood : J = {:.2f}'.format(ll[-1])
            self.plot_likelihood(ll)

    def inference(self, X):
        tau = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            tau[:, k] = self.compute_tau(X, self.pi[k], self.mu[k], self.sigma[k])
        tau = tau / np.sum(tau, axis=-1, keepdims=True)
        return np.argmax(tau, axis=-1)

    def plot(self, X):
        colors = ['blue', 'red', 'green', 'purple']
        z = self.inference(X)
        splot = plt.subplot()
        for k in range(self.K):
            plt.scatter(X[z == k, 0], X[z == k, 1], marker='^', c=colors[k])
            if self.covariance_hypothesis == 'spherical':
                plot_ellipse(splot, self.mu[k], self.sigma[k] * np.eye(2), 'blue')
            elif self.covariance_hypothesis == 'general':
                plot_ellipse(splot, self.mu[k], self.sigma[k], 'blue')
        plt.scatter(self.mu[:, 0], self.mu[:, 1], marker='*', c='orange', s=200)
        plt.axis('equal')
        plt.show()
        plt.close()

if __name__=='__main__':
    X_train = load_data()
    X_test = load_data(type='test')

    # kmeans = KMeans(4)
    # kmeans.train(X_train)
    # kmeans.plot(X_train)
    # kmeans.plot(X_test)

    gm = GaussianMixture(4, 'spherical')
    gm.train(X_train)
    gm.plot(X_train)
    # gm.plot(X_test)

    gm = GaussianMixture(4, 'general')
    # gm.train(X_train)
    # gm.plot(X_train)
    # gm.plot(X_test)

