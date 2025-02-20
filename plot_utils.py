import numpy as np
from matplotlib import pyplot as plt


def scree_plot(pca):
    """
    Creates a scree plot associated with the princiapl components

    :param df: pca - the results of a pca object that has been fit_transform by pca
    :return:

    """
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_

    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals, width=0.8)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        if((i%5==0)):
            ax.annotate(r'%s%%' % ((str(vals[i]*100)[:4])),
                (ind[i]+0.2, vals[i]), va='bottom', ha='center', fontsize=10)
    # ax.set_xticks(ind)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Explained Variance')
    plt.title('Explained Variance Per Principal Component')
    plt.show()
