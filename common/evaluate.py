from __future__ import division

import mxnet as mx
import numpy as np
import os
from tqdm import tqdm
import sklearn.cluster
import sklearn.metrics
from sklearn.metrics.pairwise import cosine_distances


def cluster_by_kmeans(X, nb_clusters, minibatch=False, use_threads=True):
    """
    xs : embeddings with shape [nb_samples, nb_features]
    nb_clusters : in this case, must be equal to number of classes
    """
    try:
        # There is a numerical instability issue in scikit-learn, see:
        # https://github.com/scikit-learn/scikit-learn/issues/8583
        if minibatch:
            cluster_labels = sklearn.cluster.MiniBatchKMeans(nb_clusters).fit(X).labels_
        else:
            cluster_labels = sklearn.cluster.KMeans(nb_clusters, n_jobs=(-1 if use_threads else 1)).fit(X).labels_
    except IndexError:
        cluster_labels = None
    return cluster_labels


def calc_normalized_mutual_information(ys, xs_clustered):
    return sklearn.metrics.cluster.normalized_mutual_info_score(xs_clustered, ys, average_method='geometric')


def get_distance_matrix(x, similarity='euclidean', use_mxnet=False):
    """Get distance matrix given a matrix. Used in testing."""
    if similarity == 'euclidean':
        if use_mxnet:
            pairwise_distances_squared = x.square().sum(axis=1, keepdims=True) + x.transpose().square().sum(axis=0,
                                                                                                            keepdims=True) - 2.0 * (
                                             mx.nd.dot(x, x.transpose()))
            pairwise_distances_squared = pairwise_distances_squared.asnumpy()
        else:
            x = x.asnumpy()
            pairwise_distances_squared = np.sum(np.square(x), axis=1, keepdims=True) + np.sum(np.square(np.transpose(x)), axis=0, keepdims=True) - 2.0 * np.dot(x, np.transpose(x))
    elif similarity == 'cosine':
        pairwise_distances_squared = cosine_distances(x.asnumpy())
    else:
        raise RuntimeError('Unknown distance metric: %s' % similarity)
    return pairwise_distances_squared


def _silhouette(d_mat, labels, validation_pairs):
    """
    Compute Silhouette coefficient.
    https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
    """
    silhouette_score = sklearn.metrics.silhouette_score(d_mat, labels, metric='euclidean')

    validation_pairs.append(('Silhouette coefficient', silhouette_score))

    return validation_pairs


def _ir_precision(emb, d_mat, labels, validation_pairs):
    """Compute IR precision."""
    for k in tqdm([1, 2, 4, 8, 16], desc='Computing recall'):
        correct_precision, cnt_precision = 0.0, 0.0
        for i in range(emb.shape[0]):
            d_mat[i, i] = 1e10
            nns = np.argpartition(d_mat[i], k)[:k]

            for nn in nns:
                if labels[i] == labels[nn]:
                    correct_precision += 1
                cnt_precision += 1

        validation_pairs.append(('IR Precision@%d' % k, correct_precision / cnt_precision * 100.0))
    return validation_pairs


def evaluate(emb, labels, num_classes, nmi=True, similarity='euclidean',
             logger=None, use_threads=True, get_detailed_metrics=False):
    """Evaluate embeddings."""
    d_mat = get_distance_matrix(emb, similarity)
    labels = labels.asnumpy()

    validation_pairs = []

    if get_detailed_metrics:
        validation_pairs = _silhouette(d_mat, labels, validation_pairs)
        validation_pairs = _ir_precision(emb, d_mat, labels, validation_pairs)

    # Compute recall
    safeprint(logger, 'Computing recall')
    np.fill_diagonal(d_mat, 1e10)
    d_mat_sorted = np.argpartition(d_mat, range(1, 17))[:, :16]
    labels_matching = (labels[d_mat_sorted] == np.expand_dims(labels, 1))
    for k in [1, 2, 4, 8, 16]:
        correct = np.sum(np.any(labels_matching[:, :k], axis=1))
        validation_pairs.append(('Recall@%d' % k, correct / float(d_mat.shape[0]) * 100.0))

    # Compute NMI
    if nmi:
        if num_classes > 1000:
            safeprint(logger, 'Computing minibatch kmeans')
            cluster_labels = cluster_by_kmeans(emb.asnumpy(), num_classes, minibatch=True, use_threads=use_threads)
        else:
            safeprint(logger, 'Computing kmeans')
            cluster_labels = cluster_by_kmeans(emb.asnumpy(), num_classes, use_threads=use_threads)
        if cluster_labels is not None:
            safeprint(logger, 'Computing NMI')
            nmi = calc_normalized_mutual_information(labels, cluster_labels)
        else:
            nmi = 0.0
        validation_pairs.append(('NMI', nmi * 100))

    return validation_pairs


def safeprint(logger, text):
    if logger is not None:
        logger.info(text)
    else:
        print(text)
