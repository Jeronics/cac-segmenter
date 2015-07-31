import numpy as np

from sklearn import mixture


def get_mixture_gaussian(X, n_components, cv_type):
    gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
    gmm.fit(X)
    return gmm


def get_number_of_components(X, maximum_n_components=7):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, maximum_n_components + 1)
    cv_types = ['full'] #,'full', 'diag', 'spherical', 'tied']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = get_mixture_gaussian(X, n_components, cv_type)
            aux_score = gmm.bic(X)
            bic.append(aux_score)
            print bic[-1], lowest_bic
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]-bic[-1]*0.0
                best_gmm = gmm

    # bic = np.array(bic)
    # color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
    # clf = best_gmm
    # bars = []
    #
    # Plot the BIC scores
    # spl = plt.subplot(2, 1, 1)
    # for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    # xpos = np.array(n_components_range) + .2 * (i - 2)
    #     bars.append(plt.bar(xpos, bic[i * len(n_components_range):
    #     (i + 1) * len(n_components_range)],
    #                         width=.2, color=color))
    # plt.xticks(n_components_range)
    # plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    # plt.title('BIC score per model')
    # xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    # .2 * np.floor(bic.argmin() / len(n_components_range))
    # plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    # spl.set_xlabel('Number of components')
    # spl.legend([b[0] for b in bars], cv_types)
    # plt.show()
    # plt.cla()
    # plt.clf()

    return best_gmm


if __name__ == '__main__':
    # Number of samples per component
    n_samples = 500

    # Generate random sample, two components
    np.random.seed(0)
    X_1 = np.ones([1, 100]) * 10
    X_2 = np.ones([1, 100]) * 30
    X = np.concatenate([X_1, X_2], axis=1)
    X = np.concatenate([X, X, X * 2 / 3.]).T

    print X.shape
    # # Fit a mixture of Gaussians with EM using five components
    # n_components_range = range(1, 7)
    # cv_types = ['spherical', 'tied', 'diag', 'full']
    # gmm = mixture.GMM(n_components=2, covariance_type='full')
    # gmm.fit(X)

    # # Fit a Dirichlet process mixture of Gaussians using five components
    # dpgmm = mixture.DPGMM(n_components=5, covariance_type='full')
    # dpgmm.fit(X)
    best_gmm = get_number_of_components(X)


