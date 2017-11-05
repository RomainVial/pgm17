import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import copy

mpl.rcParams.update({'font.size': 30})


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
    def __init__(self, K, initializer='zero'):
        self.K = K
        self.mu = None
        self.maxstep = 100
        self.nb_restart = 10
        if initializer == 'zero':
            self.initialize = self.zero_initializer
        elif initializer == 'random':
            self.initialize = self.random_initializer
        elif initializer == 'random_from_data':
            self.initialize = self.random_from_data_initializer

    def zero_initializer(self, X):
        return np.zeros((self.K, X.shape[1]))

    def random_initializer(self, X):
        x_min, x_max = np.min(X[:,0]), np.max(X[:,0])
        y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
        return [(x_max - x_min), (y_max - y_min)] * np.random.rand(self.K, X.shape[1]) + [x_min, y_min]

    def random_from_data_initializer(self, X):
        indexes = np.random.choice(X.shape[0], size=self.K, replace=False)
        return X[indexes,:]

    def distortion(self, X, z, mu):
        J = 0
        for k in range(self.K):
            J += np.sum(np.square(X[z == k,:] - mu[k,:]))
        return J

    def train(self, X, render=True):
        J = [ [] for _ in range(self.nb_restart) ]
        mu = np.zeros((self.nb_restart, self.K, X.shape[1]))
        for restart in range(self.nb_restart):
            mu[restart,:] = self.initialize(X)
            mu_old = copy.copy(mu[restart,:])
            term = False
            step = 0
            while step < self.maxstep and not term:
                # Compute distances then assign to corresponding centroids
                distances = np.sum(np.square(np.expand_dims(X, axis=1) - mu[restart]), axis=-1)
                z = np.argmin(distances, axis=-1)

                # Compute distortion
                J[restart].append(self.distortion(X, z, mu[restart]))

                # Update the centroids based on the assignment
                for k in range(self.K):
                    mu[restart,k] = np.sum(X[z == k,:], axis=0) / (np.sum(z == k) + 1e-12)

                # Terminate the computation if we reach a stationary point
                term = np.all(np.equal(mu[restart], mu_old))
                mu_old = copy.copy(mu[restart])
                step += 1
        best_restart = np.argmin(J[restart][-1] for restart in range(self.nb_restart))
        self.mu = mu[best_restart]

        if render:
            print 'Fitted after {} iterations'.format(len(J[best_restart][: ]))
            print 'Initial distortion : J = {:.2f}'.format(J[best_restart][0])
            print 'Final distortion : J = {:.2f}'.format(J[best_restart][-1])
            self.plot_distortion(J[best_restart])

    def inference(self, X):
        distances = np.sum(np.square(np.expand_dims(X, axis=1) - self.mu), axis=-1)
        return np.argmin(distances, axis=-1)

    def plot(self, X):
        colors = ['blue', 'red', 'purple', 'darkred']
        z = self.inference(X)

        # Plot the decision boundary. For that, we will assign a color to each
        # points of a meshgrid.
        h = .02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.inference(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        # Plot the data points and centroids
        for k in range(self.K):
            plt.scatter(X[z == k,0], X[z == k,1], marker='^', c=colors[k])
        plt.scatter(self.mu[:, 0], self.mu[:, 1], marker='x', c='white', s=400)
        plt.show()
        plt.close()

    def plot_distortion(self, J):
        plt.plot(J, linewidth=3)
        plt.xlabel('Iterations')
        plt.ylabel('Distortion')
        plt.show()
        plt.close()


class GaussianMixture:
    def __init__(self, K, covariance_hypothesis='spherical', iterations=50):
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
        self.iterations = iterations

    def _compute_sigma_spherical(self, X, tau, mu):
        return 0.5 * np.sum(tau * ((X-mu) * (X-mu)).sum(axis=1)) / np.sum(tau)

    def _compute_tau_spherical(self, X, pi, mu, sigma):
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
        kmeans = KMeans(self.K, initializer='random')
        kmeans.train(X, render=False)
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

    def log_likelihood(self, X, pi, mu, sigma):
        ll = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            ll[:, k] = self.compute_tau(X, pi[k], mu[k], sigma[k])
        ll = np.sum(np.log(np.sum(ll, axis=1)))
        return ll

    def train(self, X, render=True):
        ll = []
        pi, mu, sigma = self.initialize_with_kmeans(X)
        tau = np.zeros((X.shape[0], self.K))

        for step in range(self.iterations):
            # Compute log-likelihood
            ll.append(self.log_likelihood(X, pi, mu, sigma))

            # Compute soft assignment to cluster k, tau[i,k] (Expectation step)
            for k in range(self.K):
                tau[:,k] = self.compute_tau(X, pi[k], mu[k], sigma[k])
            tau = tau / np.sum(tau, axis=-1, keepdims=True)

            # Compute estimates for pi, mu, sigma based on the soft assignment (Maximization step)
            pi = np.mean(tau, axis=0)
            for k in range(self.K):
                mu[k] = np.sum(np.expand_dims(tau[:,k], axis=-1) * X, axis=0) / np.sum(tau[:,k])
                sigma[k] = self.compute_sigma(X, tau[:,k], mu[k])

        self.pi = pi
        self.mu = mu
        self.sigma = sigma

        if render:
            print 'Initial likelihood : J = {:.2f}'.format(ll[0])
            print 'Final likelihood : J = {:.2f}'.format(ll[-1])
            self.plot_likelihood(ll)

    def inference(self, X):
        tau = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            tau[:, k] = self.compute_tau(X, self.pi[k], self.mu[k], self.sigma[k])
        tau = tau / np.sum(tau, axis=-1, keepdims=True)
        return tau

    def plot(self, X):
        colors = ['blue', 'red', 'purple', 'darkred']
        z = np.argmax(self.inference(X), axis=-1)

        # Compute the likelihood on the set
        ll = self.log_likelihood(X, self.pi, self.mu, self.sigma)
        print 'Likelihood : J = {:.2f}'.format(ll)

        # Plot the decision boundary. For that, we will assign a color to each points of a meshgrid.
        h = .1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = np.argmax(self.inference(np.c_[xx.ravel(), yy.ravel()]), axis=-1)
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired,
                   aspect='auto', origin='lower')

        # Plot the data points, ellipses and centroids
        splot = plt.subplot()
        for k in range(self.K):
            plt.scatter(X[z == k, 0], X[z == k, 1], marker='^', c=colors[k])
            if self.covariance_hypothesis == 'spherical':
                plot_ellipse(splot, self.mu[k], self.sigma[k] * np.eye(2), colors[k])
            elif self.covariance_hypothesis == 'general':
                plot_ellipse(splot, self.mu[k], self.sigma[k], colors[k])
        plt.scatter(self.mu[:, 0], self.mu[:, 1], marker='x', c='white', s=400)
        plt.axis('equal')
        plt.show()
        plt.close()

    def plot_likelihood(self, ll):
        plt.plot(ll, linewidth=3)
        plt.title('Likelihood against the number of iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Likelihood')
        plt.show()
        plt.close()

if __name__=='__main__':
    X_train = load_data()
    X_test = load_data(type='test')

    print "----- K-MEANS -----"
    kmeans = KMeans(4, initializer='random')
    kmeans.train(X_train)
    kmeans.plot(X_train)
    kmeans.plot(X_test)

    print "\n----- SPHERICAL GMM -----"
    gm = GaussianMixture(4, 'spherical')
    gm.train(X_train)
    gm.plot(X_train)
    gm.plot(X_test)

    print "\n----- GENERAL GMM -----"
    gm = GaussianMixture(4, 'general')
    gm.train(X_train)
    gm.plot(X_train)
    gm.plot(X_test)
