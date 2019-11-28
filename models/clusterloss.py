import mxnet as mx
from mxnet.gluon import loss

from models.loss import pairwise_distance


def unique(F, data):
    """
    Returns the unique elements of a 1D array
    :param F:
    :param data:
    :return:
    """
    sdata = F.reshape(data, (-1,))
    sdata = F.sort(sdata, axis=-1)
    mask = F.concat(F.ones(1, ctx=sdata.context, dtype=sdata.dtype),
                    F.slice(sdata, begin=1, end=(None,)) != F.slice(sdata, begin=(None,), end=(-1,)), dim=0)
    return F.contrib.boolean_mask(sdata, mask)


def cluster_indices(F, input_data, unique_data=None):
    if unique_data is None:
        unique_values = unique(F, input_data)
    else:
        unique_values = unique_data
    return F.broadcast_equal(unique_values.expand_dims(1), input_data.transpose())


def entropy(F, input_data, unique_data=None):
    """
    Computes the entropy of the input data
    :param F:
    :param input_data:
    :param unique_data:
    :return:
    """
    if unique_data is None:
        bins = F.sum(F.broadcast_equal(unique(F, input_data).expand_dims(1), input_data), axis=1)
    else:
        bins = F.sum(F.broadcast_equal(unique_data.expand_dims(1), input_data), axis=1)
    pi = bins / F.sum(bins)
    result = -F.sum(pi * F.log(pi))
    return result


def first_k(F, data, k):
    """
    Returns the first k elements of an array. K is of NDArray type.
    :param F:
    :param data:
    :param k:
    :return:
    """
    mask = (F.arange(start=0, stop=data.size) < k)
    return F.contrib.boolean_mask(data, mask)


def unique_intersection(F, ind1, ind2):
    """
    Computes the intersection of two unique sets
    :param F:
    :param ind1:
    :param ind2:
    :return:
    """
    # must be unique
    a = F.broadcast_equal(ind1, ind2.expand_dims(0).transpose())
    s = F.sum(a, axis=())
    return s


def mutual_information_score(F, data1, data2, unique_data1=None, unique_data2=None):
    cluster_indices1 = cluster_indices(F, data1, unique_data1)  # N1xD
    cluster_indices2 = cluster_indices(F, data2, unique_data2)  # N2xD
    cluster_sizes1 = F.sum(cluster_indices1, axis=1)  # N1
    cluster_sizes2 = F.sum(cluster_indices2, axis=1)  # N2
    cluster_sizes = F.broadcast_mul(cluster_sizes1.expand_dims(1), cluster_sizes2.expand_dims(0))
    N = data1.size
    # assert data1.size == data2.size

    mask = F.broadcast_add(cluster_indices1.expand_dims(1), cluster_indices2.expand_dims(0)) == 2
    num_intersection = F.sum(mask, axis=2)  # N1xN2
    score = num_intersection / N * F.log(num_intersection * N / cluster_sizes)

    return F.nansum(score)


def nmi(F, labels_true, labels_pred):
    unique_labels_true = unique(F, labels_true)
    unique_labels_pred = unique(F, labels_pred)
    h_true = F.maximum(entropy(F, labels_true, unique_labels_true), 0)
    h_pred = F.maximum(entropy(F, labels_pred, unique_labels_pred), 0)

    mi = mutual_information_score(F, labels_true, labels_pred, unique_labels_true, unique_labels_pred)

    _nmi = mi / F.maximum(F.sqrt(h_true * h_pred), 1e-10)
    return _nmi


def get_cluster_assignment(F, pairwise_distances, centroid_ids):
    """Assign data points to the neareset centroids.
      Due to computational instability for each centroid in centroid_ids,
      explicitly assign the centroid itself as the nearest centroid.
      This is done through the mask tensor and the constraint_vect tensor.
    Args:
      pairwise_distances: 2-D Tensor of pairwise distances.
      centroid_ids: 1-D Tensor of centroid indices.
    Returns:
      y_fixed: 1-D tensor of cluster assignment.
    """
    predictions = F.topk(-F.take(pairwise_distances, centroid_ids, axis=0), k=1, ret_typ='indices', axis=0).squeeze()

    batch_size = pairwise_distances.shape[0]

    # Deal with numerical instability
    mask = F.clip(F.sum(F.one_hot(centroid_ids, batch_size), axis=0), 0, 1)

    constraint_one_hot = F.one_hot(centroid_ids, batch_size).transpose() * F.arange(centroid_ids.shape[0],
                                                                                    ctx=pairwise_distances.context)

    constraint_vect = F.sum(constraint_one_hot.transpose(), axis=0)

    y_fixed = F.where(mask, constraint_vect, predictions)

    return y_fixed


def compute_clustering_score(F, labels, predictions, margin_type='nmi'):
    """Computes the clustering score via sklearn.metrics functions.
    There are various ways to compute the clustering score. Intuitively,
    we want to measure the agreement of two clustering assignments (labels vs
    predictions) ignoring the permutations and output a score from zero to one.
    (where the values close to one indicate significant agreement).
    This code supports following scoring functions:
      nmi: normalized mutual information
    Args:
      labels: 1-D Tensor. ground truth cluster assignment.
      predictions: 1-D Tensor. predicted cluster assignment.
      margin_type: Type of structured margin to use. Default is nmi.
    Returns:
      clustering_score: dtypes.float32 scalar.
        The possible valid values are from zero to one.
        Zero means the worst clustering and one means the perfect clustering.
    Raises:
      ValueError: margin_type is not recognized.
    """
    margin_type_to_func = {
        'nmi': nmi,
        # 'nmi': metrics.normalized_mutual_info_score,
        # 'ami': _compute_ami_score,
        # 'ari': _compute_ari_score,
        # 'vmeasure': _compute_vmeasure_score,
        # 'const': _compute_zeroone_score
    }

    if margin_type not in margin_type_to_func:
        raise ValueError('Unrecognized margin_type: %s' % margin_type)
    return margin_type_to_func[margin_type](F, labels, predictions)


def _find_loss_augmented_facility_idx(F, pairwise_distances, labels, chosen_ids,
                                      candidate_ids, margin_multiplier,
                                      margin_type):
    """Find the next centroid that maximizes the loss augmented inference.
    This function is a subroutine called from compute_augmented_facility_locations
    Args:
      pairwise_distances: 2-D Tensor of pairwise distances.
      labels: 1-D Tensor of ground truth cluster assignment.
      all_ids: All indices in a sorted 1-D array
      chosen_ids: 1-D Tensor of current centroid indices. Can be None when empty.
      candidate_ids: 1-D Tensor of candidate indices.
      margin_multiplier: multiplication constant.
      margin_type: Type of structured margin to use. Default is nmi.
    Returns:
      integer index.
    """
    num_candidates = candidate_ids.shape[0]

    pairwise_distances_candidate = F.take(pairwise_distances, candidate_ids, axis=0)

    def then_func():
        pairwise_distances_chosen = F.take(pairwise_distances, chosen_ids, axis=0)
        pairwise_distances_chosen_tile = F.tile(pairwise_distances_chosen, reps=(1, num_candidates))
        return F.concat(pairwise_distances_chosen_tile, F.reshape(pairwise_distances_candidate, (1, -1)), dim=0)

    def else_func():
        return F.reshape(pairwise_distances_candidate, (1, -1))

    chosen_m = F.contrib.cond(chosen_ids.size_array() > 0, then_func, else_func)

    candidate_scores = -1.0 * F.sum(F.reshape(F.min(chosen_m, axis=0, keepdims=True), (num_candidates, -1)), axis=1)

    nmi_scores = F.zeros((num_candidates,), ctx=pairwise_distances.context)
    iteration = F.zeros((1,), ctx=pairwise_distances.context)

    for iteration_sc in range(num_candidates):
        # Cannot concat 0 sized tensors as of 1.5
        def then_func():
            return get_cluster_assignment(F, pairwise_distances,
                                          F.concat(chosen_ids, F.take(candidate_ids, iteration, axis=0), dim=0))

        def else_func():
            return get_cluster_assignment(F, pairwise_distances, F.take(candidate_ids, iteration, axis=0))

        predictions = F.contrib.cond(chosen_ids.size_array() > 0, then_func, else_func)

        nmi_score_i = compute_clustering_score(F, labels, predictions, margin_type)

        score = 1.0 - nmi_score_i
        score = F.one_hot(iteration, nmi_scores.size).squeeze() * score

        nmi_scores = nmi_scores + score
        iteration = iteration + 1

    candidate_scores = candidate_scores + (margin_multiplier * nmi_scores)
    argmax_index = F.topk(candidate_scores, k=1, ret_typ='indices', axis=0).squeeze()

    return F.take(candidate_ids, argmax_index, axis=0)


def diff1d(F, x, y, max_classes):
    """Given two 1D tensors x and y, this operation returns a 1D tensor
       that represents all values that are in x but not in y
    """
    def _diff1d(x, y):
        chosen_mask = F.sum(F.one_hot(y, max_classes), axis=0)
        all_mask = F.sum(F.one_hot(x, max_classes), axis=0)
        remaining_mask = all_mask - chosen_mask
        remaining_mask = F.slice_like(remaining_mask, x)
        return F.contrib.boolean_mask(x, remaining_mask)

    def then_func_both():
        def thenx():
            return x

        def theny():
            return y
        return F.contrib.cond(x.size_array() == 0, theny, thenx)

    def else_both_func():
        return _diff1d(x, y)

    return F.contrib.cond((y.size_array() * x.size_array()) == 0, then_func_both, else_both_func)


def concat2(F, x, y, dim):
    def then_func():
        return F.concat(x, y, dim=dim)

    def else_func():
        return y

    return F.contrib.cond(x.size_array() > 0, then_func, else_func)


def compute_augmented_facility_locations(F, pairwise_distances, labels, unique_labels, all_ids,
                                         margin_multiplier, num_classes, margin_type='nmi'):

    def func_body_iteration(_, states):
        chosen_ids, all_ids = states
        # we need all ID's that are not in chosen_ids
        candidate_ids = diff1d(F, all_ids, chosen_ids, num_classes)

        new_chosen_idx = _find_loss_augmented_facility_idx(F, pairwise_distances,
                                                           labels,
                                                           chosen_ids,
                                                           candidate_ids,
                                                           margin_multiplier,
                                                           margin_type)

        chosen_ids = concat2(F, chosen_ids, new_chosen_idx, dim=0)

        return chosen_ids, (chosen_ids, all_ids)

    # crashes in 1.5.1 but not in 1.4
    if mx.__version__.split('.')[0:2] == ['1', '5']:
        chosen_ids = mx.nd.array([]) # not hybridizable
    else:
        chosen_ids = F.zeros((0,), ctx=pairwise_distances.context)
    _, states = F.contrib.foreach(func_body_iteration, unique_labels, (chosen_ids, all_ids))

    return states[0]


def compute_facility_energy(F, pairwise_distances, centroid_ids):
    """Compute the average travel distance to the assigned centroid.
    Args:
      pairwise_distances: 2-D Tensor of pairwise distances.
      centroid_ids: 1-D Tensor of indices.
    Returns:
      facility_energy: [1]
    """
    return -1.0 * F.sum(F.min(F.take(pairwise_distances, centroid_ids, axis=0), axis=0))


def update_1d_tensor(F, y, index, value):
    """Updates 1d tensor y so that y[index] = value.

    Args:
      y: 1-D Tensor.
      index: index of y to modify.
      value: new value to write at y[index].

    Returns:
      y_mod: 1-D Tensor. Tensor y after the update.
    """
    value = value.squeeze()

    o = F.one_hot(index, y.size).squeeze()
    r = y * (1 - o)
    return r + (o * value)


def update_medoid_per_cluster(F, pairwise_distances, pairwise_distances_subset,
                              labels, chosen_ids, cluster_member_ids,
                              cluster_idx, margin_multiplier, margin_type):
    """Updates the cluster medoid per cluster.
    Args:
      pairwise_distances: 2-D Tensor of pairwise distances.
      pairwise_distances_subset: 2-D Tensor of pairwise distances for one cluster.
      labels: 1-D Tensor of ground truth cluster assignment.
      chosen_ids: 1-D Tensor of cluster centroid indices.
      cluster_member_ids: 1-D Tensor of cluster member indices for one cluster.
      cluster_idx: Index of this one cluster.
      margin_multiplier: multiplication constant.
      margin_type: Type of structured margin to use. Default is nmi.
    Returns:
      chosen_ids: Updated 1-D Tensor of cluster centroid indices.
    """

    # pairwise_distances_subset is of size [p, 1, 1, p],
    #   the intermediate dummy dimensions at
    #   [1, 2] makes this code work in the edge case where p=1.
    #   this happens if the cluster size is one.
    scores_fac = -1.0 * F.sum(F.squeeze(pairwise_distances_subset, axis=(1, 2)), axis=0)

    iteration = F.zeros((1,), ctx=pairwise_distances.context)
    num_candidates = cluster_member_ids.size
    scores_margin = F.zeros((num_candidates,), ctx=pairwise_distances.context)

    for it in range(num_candidates):
        candidate_medoid = F.take(cluster_member_ids, iteration, axis=0)
        tmp_chosen_ids = update_1d_tensor(F, chosen_ids, cluster_idx, candidate_medoid)
        predictions = get_cluster_assignment(F, pairwise_distances, tmp_chosen_ids)
        metric_score = compute_clustering_score(F, labels, predictions, margin_type)

        if it > 0:
            scores_m = F.concat(F.zeros((it,), ctx=pairwise_distances.context), 1.0 - metric_score, dim=0)
        else:
            scores_m = 1.0 - metric_score

        if it < num_candidates - 1:
            scores_m = F.concat(scores_m, F.zeros((num_candidates - 1 - it,), ctx=pairwise_distances.context), dim=0)

        iteration = iteration + 1
        scores_margin = scores_margin + scores_m

    candidate_scores = scores_fac + (margin_multiplier * scores_margin)

    argmax_index = F.topk(candidate_scores, k=1, ret_typ='indices', axis=0).squeeze()

    best_medoid = F.take(cluster_member_ids, argmax_index, axis=0)
    chosen_ids = update_1d_tensor(F, chosen_ids, cluster_idx, best_medoid)
    return chosen_ids


def update_all_medoids(F, pairwise_distances, predictions, labels, chosen_ids,
                       margin_multiplier, margin_type):
    """Updates all cluster medoids a cluster at a time.
    Args:
      pairwise_distances: 2-D Tensor of pairwise distances.
      predictions: 1-D Tensor of predicted cluster assignment.
      labels: 1-D Tensor of ground truth cluster assignment.
      chosen_ids: 1-D Tensor of cluster centroid indices.
      margin_multiplier: multiplication constant.
      margin_type: Type of structured margin to use. Default is nmi.
    Returns:
      chosen_ids: Updated 1-D Tensor of cluster centroid indices.
    """

    def func_cond_augmented_pam(iteration, chosen_ids):
        del chosen_ids  # Unused argument.
        return iteration < num_unique

    def func_body_augmented_pam(iteration, chosen_ids):
        """Call the update_medoid_per_cluster subroutine."""
        mask = F.equal(predictions, iteration)
        # Get the index of all ones in mask
        this_cluster_ids = F.argsort(mask, axis=0, is_ascend=False)
        this_cluster_ids = first_k(F, this_cluster_ids, F.sum(mask))
        this_cluster_ids = this_cluster_ids.expand_dims(1)

        pairwise_distances_subset = F.take(F.take(pairwise_distances, this_cluster_ids, axis=0).transpose(),
                                           this_cluster_ids, axis=0).transpose()

        chosen_ids = update_medoid_per_cluster(F, pairwise_distances,
                                               pairwise_distances_subset, labels,
                                               chosen_ids, this_cluster_ids,
                                               iteration, margin_multiplier,
                                               margin_type)
        return None, (iteration + 1, chosen_ids)

    num_unique = unique(F, labels).size
    iteration = F.zeros((1,), ctx=pairwise_distances.context)

    _, chosen_ids = F.contrib.while_loop(
        func_cond_augmented_pam, func_body_augmented_pam, (iteration, chosen_ids), max_iterations=num_unique)
    return chosen_ids[1]


def compute_augmented_facility_locations_pam(F, pairwise_distances,
                                             labels,
                                             margin_multiplier,
                                             margin_type,
                                             chosen_ids,
                                             pam_max_iter=5):
    """Refine the cluster centroids with PAM local search.
    For fixed iterations, alternate between updating the cluster assignment
      and updating cluster medoids.
    Args:
      pairwise_distances: 2-D Tensor of pairwise distances.
      labels: 1-D Tensor of ground truth cluster assignment.
      margin_multiplier: multiplication constant.
      margin_type: Type of structured margin to use. Default is nmi.
      chosen_ids: 1-D Tensor of initial estimate of cluster centroids.
      pam_max_iter: Number of refinement iterations.
    Returns:
      chosen_ids: Updated 1-D Tensor of cluster centroid indices.
    """
    for _ in range(pam_max_iter):
        # update the cluster assignment given the chosen_ids (S_pred)
        predictions = get_cluster_assignment(F, pairwise_distances, chosen_ids)

        # update the medoids per each cluster
        chosen_ids = update_all_medoids(F, pairwise_distances, predictions, labels,
                                        chosen_ids, margin_multiplier, margin_type)

    return chosen_ids


def compute_gt_cluster_score(F, pairwise_distances, labels, unique_labels):
    """Compute ground truth facility location score.

    Loop over each unique classes and compute average travel distances.

    Args:
      pairwise_distances: 2-D Tensor of pairwise distances.
      labels: 1-D Tensor of ground truth cluster assignment.

    Returns:
      gt_cluster_score: dtypes.float32 score.
    """
    num_classes = unique_labels.size

    iteration = F.zeros((1,), ctx=pairwise_distances.context)
    gt_cluster_score = F.zeros((1,), ctx=pairwise_distances.context)

    def func_cond(iteration, gt_cluster_score):
        del gt_cluster_score  # Unused argument.
        return iteration < num_classes

    def func_body(iteration, gt_cluster_score):
        """Per each cluster, compute the average travel distance."""
        mask = F.equal(labels, F.take(unique_labels, iteration, axis=0))
        this_cluster_ids = F.argsort(mask, axis=0, is_ascend=False)
        this_cluster_ids = first_k(F, this_cluster_ids, F.sum(mask))
        this_cluster_ids = this_cluster_ids.expand_dims(1)

        pairwise_distances_subset = F.take(F.take(pairwise_distances, this_cluster_ids, axis=0).transpose(),
                                           this_cluster_ids, axis=0).transpose()

        this_cluster_score = -1.0 * F.min(F.sum(pairwise_distances_subset, axis=0))
        return None, (iteration + 1, gt_cluster_score + this_cluster_score)

    _, gt_cluster_score = F.contrib.while_loop(
        func_cond, func_body, (iteration, gt_cluster_score), max_iterations=num_classes)

    return gt_cluster_score[1]


class ClusterLoss(loss.Loss):
    def __init__(self, num_classes, enable_pam_finetuning=True,
                 margin_type='nmi', margin_multiplier=1.0, weight=None, batch_axis=0, **kwargs):
        super(ClusterLoss, self).__init__(weight, batch_axis, **kwargs)
        self.num_classes = num_classes
        self._enable_pam_finetuning = enable_pam_finetuning
        self._margin_type = margin_type
        self._margin_multiplier = margin_multiplier

    def hybrid_forward(self, F, embeddings, labels, unique_labels, all_ids):
        pairwise_distances = pairwise_distance(F, embeddings)

        # Compute the loss augmented inference and get the cluster centroids.
        chosen_ids = compute_augmented_facility_locations(F, pairwise_distances, labels, unique_labels,
                                                          all_ids, self._margin_multiplier, self.num_classes,
                                                          self._margin_type)

        if self._enable_pam_finetuning:
            # Initialize with augmented facility solution.
            chosen_ids = compute_augmented_facility_locations_pam(F, pairwise_distances,
                                                                  labels,
                                                                  self._margin_multiplier,
                                                                  self._margin_type,
                                                                  chosen_ids)
            score_pred = compute_facility_energy(F, pairwise_distances, chosen_ids)
        else:
            # Given the predicted centroids, compute the clustering score.
            score_pred = compute_facility_energy(F, pairwise_distances, chosen_ids)

        # Given the predicted centroids, compute the cluster assignments.
        predictions = get_cluster_assignment(F, pairwise_distances, chosen_ids)

        # Compute the clustering (i.e. NMI) score between the two assignments.
        clustering_score_pred = compute_clustering_score(F, labels, predictions,
                                                         self._margin_type)

        # Compute the clustering score from labels.
        score_gt = compute_gt_cluster_score(F, pairwise_distances, labels, unique_labels)

        # Compute the hinge loss.
        clustering_loss = F.maximum(score_pred + self._margin_multiplier * (1.0 - clustering_score_pred) - score_gt,
                                    0.0)

        return clustering_loss
