import mxnet as mx
import numpy as np
from mxnet.gluon import loss
from mxnet.gluon.loss import _apply_weighting


class ProxyTripletLoss(loss.Loss):
    """
    Outputs:
        - **loss**: loss tensor with shape (batch_size,).
    """

    def __init__(self, num_classes, margin=1, multiplier=1, weight=None, batch_axis=0, **kwargs):
        super(ProxyTripletLoss, self).__init__(weight, batch_axis, **kwargs)
        self._num_classes = num_classes
        self._margin = margin
        self._multiplier = multiplier

    def hybrid_forward(self, F, pred, positive_proxy, negative_proxies):
        """
        :param F:
        :param pred: BxE
        :param positive_proxy: BxE
        :param negative_proxies: B x (C-1) x E
        :return:
        """
        pred = pred * self._multiplier
        positive_proxy = F.L2Normalization(positive_proxy) * self._multiplier
        pred_b = F.repeat(pred, repeats=self._num_classes - 1, axis=0)  # B*(C-1) x E
        positive_proxy_b = F.repeat(positive_proxy, repeats=self._num_classes - 1, axis=0)  # B*(C-1) x E
        negative_proxies_b = F.reshape_like(negative_proxies, pred_b)  # B*(C-1) x E
        negative_proxies_b = F.L2Normalization(negative_proxies_b) * self._multiplier  # B*(C-1) x E

        loss = F.sum(F.square(pred_b - positive_proxy_b) - F.square(pred_b - negative_proxies_b),
                     axis=self._batch_axis, exclude=True)
        loss = F.relu(loss + self._margin)
        return loss


class ProxyMarginTripletLoss(loss.Loss):
    """
    Outputs:
        - **loss**: loss tensor with shape (batch_size,).
    """

    def __init__(self, num_classes, beta, margin=0.2, nu=0.0, weight=None, batch_axis=0, **kwargs):
        super(ProxyMarginTripletLoss, self).__init__(weight, batch_axis, **kwargs)
        self._num_classes = num_classes
        self._margin = margin
        self._nu = nu
        self._beta = beta

    def hybrid_forward(self, F, pred, labels, positive_proxy, negative_proxies):
        """
        :param F:
        :param pred: BxE
        :param positive_proxy: BxE
        :param negative_proxies: B x (C-1) x E
        :return:
        """
        beta = self._beta(labels).squeeze()  # <B>
        beta_b = F.repeat(beta, repeats=self._num_classes - 1, axis=0)
        beta_reg_loss = F.sum(beta) * self._nu

        positive_proxy = F.L2Normalization(positive_proxy)  # BxE
        pred_b = F.repeat(pred, repeats=self._num_classes - 1, axis=0)  # B*(C-1) x E
        # positive_proxy_b = F.repeat(positive_proxy, repeats=self._num_classes - 1, axis=0)  # B*(C-1) x E
        negative_proxies_b = F.reshape_like(negative_proxies, pred_b)  # B*(C-1) x E
        negative_proxies_b = F.L2Normalization(negative_proxies_b)  # B*(C-1) x E

        d_ap = F.sum(F.square(positive_proxy - pred), axis=1)  # B
        d_ap = F.repeat(d_ap, repeats=self._num_classes - 1, axis=0)  # B*(C-1)
        d_an = F.sum(F.square(negative_proxies_b - pred_b), axis=1)  # B*(C-1)

        pos_loss = F.relu(d_ap - beta_b + self._margin)
        neg_loss = F.relu(beta_b - d_an + self._margin)

        pair_cnt = F.sum((pos_loss > 0.0) + (neg_loss > 0.0))
        loss = (F.sum(pos_loss + neg_loss) + beta_reg_loss) / pair_cnt

        # loss = pos_loss + neg_loss + beta_reg_loss
        # pair_cnt = F.sum(loss > 0.0)
        return _apply_weighting(F, loss, self._weight, None)


def batch_cosine_dist(F, a, b):
    """
    Computes the cosine distance between two batches of vectors.
    :param a: BxN
    :param b: BxN
    :param F:
    :return:
    """
    a1 = F.expand_dims(a, axis=1)
    b1 = F.expand_dims(b, axis=2)
    d = F.batch_dot(a1, b1)[:, 0, 0]
    a_norm = F.sqrt(F.sum((a * a), axis=1))
    b_norm = F.sqrt(F.sum((b * b), axis=1))
    dist = 1.0 - d / (a_norm * b_norm)
    return dist


def log_sum_exp(F, x, axis):
    """ numerically stable log(sum(exp(x))) implementation that prevents overflow

    :param nd or sym F: ndarray or symbol module
    :param NDArray or Symbol x: data input in NCT or NTC layout (depending on axis parameter). Optional
        sample axis can be present (SNCT or SNTC).
    :param int axis: channel axis
    :return: log(sum(exp(x))) in same layout as input x, with channel axis reduced to width of 1 (e.g. NCT -> N1T)
    :rtype: NDArray or Symbol
    """
    m = F.max(x, axis=axis, keepdims=True)
    # The reason for subtracting m first and then adding it back is for numerical stability
    return m + F.log(F.sum(F.exp(F.broadcast_sub(x, m)), axis=axis, keepdims=True))


class ProxyXentropyLoss(loss.Loss):
    def __init__(self, num_classes, label_smooth=0, temperature=0.05, weight=None, batch_axis=0, K=1,
                 reguralization_constant=0., **kwargs):
        super(ProxyXentropyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._num_classes = num_classes
        self._sigma = temperature
        self._label_smooth = label_smooth
        self._K = K
        self._reguralization_constant = reguralization_constant

    def hybrid_forward(self, F, x, proxies, labels, sample_weight=None):
        """
        :param F:
        :param x: BxE
        :param proxies: CxE
        :param negative_labels: B x (C-1) x E
        :return:
        """
        dist = 1 - F.batch_dot(x.expand_dims(1),  # Bx1xE
                               F.broadcast_like(proxies.expand_dims(0), x, lhs_axes=0, rhs_axes=0).transpose((0, 2, 1))
                               # BxExC
                               ).squeeze()  # B x 1 x C

        dist = dist / self._sigma  # add temperature

        labels_onehot = F.one_hot(labels, self._num_classes)  # BxNc
        labels_onehot = F.repeat(labels_onehot, repeats=self._K, axis=1)

        if self._label_smooth > 0:
            # Apply label smoothing
            labels_onehot_pos = (labels_onehot * (1 - self._label_smooth))
            labels_onehot_neg = ((1 - labels_onehot) * (self._label_smooth / (self._num_classes - 1)))
            labels = labels_onehot_pos + labels_onehot_neg
        else:
            labels = labels_onehot

        loss = F.sum(-labels * F.log_softmax(dist, axis=-1), -1)
        if self._reguralization_constant > 0:
            v = F.repeat(F.one_hot(F.arange(0, self._num_classes, repeat=self._K), self._num_classes),
                         repeats=self._K, axis=1)

            loss = loss + self._reguralization_constant * F.sum(v * (1 - F.dot(proxies, F.transpose(proxies))))
        return loss


class ProxyNCALoss(loss.Loss):
    def __init__(self, num_classes, exclude_positives=True, label_smooth=0, multiplier=1, temperature=1, weight=None, batch_axis=0,
                 **kwargs):
        """
        NCA-based loss
        :param num_classes: Number of classes in the training dataset (=number of proxies)
        :param exclude_positives: Use the positives in the NCA denominator. Original NCA excludes positives
        :param label_smooth: Apply label smoothing
        :param weight:
        :param batch_axis:
        :param kwargs:
        """
        super(ProxyNCALoss, self).__init__(weight, batch_axis, **kwargs)
        self._num_classes = num_classes
        self._exclude_positives = exclude_positives
        self._label_smooth = label_smooth
        self._multiplier = multiplier
        self._sigma = temperature

    def hybrid_forward(self, F, pred, proxies, labels, negative_labels):
        """
        :param F:
        :param pred: BxE (normalized)
        :param positive_proxy: BxE
        :param negative_proxies: B x (C-1) x E
        :return:
        """
        pred = pred * self._multiplier
        dist = pairwise_distance(F, pred, proxies * self._multiplier, squared=True)  # B x Nc
        dist = dist / self._sigma  # add temperature

        labels_onehot = F.one_hot(labels, self._num_classes)  # BxNc

        if self._label_smooth > 0:
            # Apply label smoothing
            labels_onehot_pos = (labels_onehot * (1 - self._label_smooth))
            labels_onehot_neg = ((1 - labels_onehot) * (self._label_smooth / (self._num_classes - 1)))
            labels = labels_onehot_pos + labels_onehot_neg
        else:
            labels = labels_onehot

        if self._exclude_positives:
            # This is NCA (excluding positive term)
            negs = []
            for i in range(dist.shape[0]):
                negs.append(F.take(dist[i], negative_labels[i]).expand_dims(0))
            n_dist = F.concat(*negs, dim=0)

            loss = -dist - F.broadcast_like(log_sum_exp(F, -n_dist, axis=1), dist)
            loss = (-labels * loss)

            loss = F.mean(F.sum(loss, axis=-1, keepdims=True), axis=0, exclude=True)
        else:
            loss = F.sum(-labels * F.log_softmax(dist, axis=-1), -1)

        return loss


class StaticProxyLoss(loss.Loss):
    def __init__(self, num_classes, weight=None, batch_axis=0, **kwargs):
        super(StaticProxyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._num_classes = num_classes

    def hybrid_forward(self, F, pred, labels):
        """
        :param F:
        :param pred: BxE (unnormalized)
        :param labels: B
        :return:
        """
        proxies = F.eye(self._num_classes)
        dist = F.dot(pred, proxies)

        dist = - F.log_softmax(dist, 1)
        loss = dist.pick(labels, axis=1)

        return loss


class PrototypeLoss(loss.Loss):
    def __init__(self, nc, ns, nq, axis=-1, weight=None, batch_axis=0, **kwargs):
        super(PrototypeLoss, self).__init__(weight, batch_axis, **kwargs)
        self.nc = nc
        self.ns = ns
        self.nq = nq
        self.axis = axis

    def hybrid_forward(self, F, supports, queries, sample_weight=None):
        """
        Computes prototypical loss
        :param F:
        :param supports:  <Nc*Ns x E>
        :param queries:   <Nc*Nq x E>
        :return:
        """
        supports = F.reshape(supports, (self.nc, self.ns, -1))  # <Nc x Ns x E>
        prototypes = F.mean(supports, axis=1)  # <Nc x E>

        # Compute distance between queries and prototypes
        square_queries = queries.square().sum(axis=1, keepdims=True)
        square_prototypes = prototypes.square().sum(axis=1,
                                                    keepdims=True)  # <Nc*Ns x 1>
        pairwise_distance_square = square_queries + square_prototypes.transpose() - 2.0 * (
            F.dot(queries, prototypes.transpose()))  # <Nc*Nq x Nc>

        # We construct the labels based on sampled clusters
        labels = F.repeat(F.arange(self.nc), self.nq)

        pred = F.log_softmax(-pairwise_distance_square, self.axis)
        loss = -F.pick(pred, labels, axis=self.axis, keepdims=True)

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class NPairsLoss(loss.Loss):
    def __init__(self, l2_reg, symmetric=False, weight=None, batch_axis=0, **kwargs):
        """
        Npair loss as proposed by Kihyuk Sohn in
        Improved Deep Metric Learning with Multi-class N-pair Loss Objective
        (http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)
        :param l2_reg: L2 regularization term (float)
        :param symmetric: Whether to compute symmetric (mean) loss (bool)
        :param weight:
        :param batch_axis:
        :param kwargs:
        """
        super(NPairsLoss, self).__init__(weight, batch_axis, **kwargs)
        self._l2_reg = l2_reg
        self._symmetric = symmetric

    def hybrid_forward(self, F, anchors, positives, labels, sample_weight=None):
        """
        Computes the loss on the given data
        :param F: mx.nd or mx.sym
        :param anchors: anchor embeddings, <BxE> where B: batch size, E: embedding dimension
        :param positives: positive embeddings, same shape and label than anchors <BxE>
        :param labels: Labels of embeddings <B>
        :param sample_weight: weights of logits, see mx.loss
        :return:
        """
        reg_anchor = F.mean(F.sum(anchors.square(), axis=1), axis=self._batch_axis, exclude=True)
        reg_positive = F.mean(F.sum(positives.square(), axis=1), axis=self._batch_axis, exclude=True)
        l2loss = self._l2_reg * (reg_anchor + reg_positive)

        # Get per pair similarities.
        similarity_matrix = F.dot(anchors, positives, transpose_a=False, transpose_b=True)

        labels = labels.expand_dims(1)

        labels_remapped = F.broadcast_equal(labels, labels.transpose())
        labels_remapped = F.broadcast_div(labels_remapped, F.sum(labels_remapped, axis=1, keepdims=True))

        # Add the softmax loss.
        labels_remapped = labels_remapped.astype(dtype='float32')
        xent_loss = F.sum(F.log_softmax(similarity_matrix, -1) * -labels_remapped, axis=-1, keepdims=True)
        xent_loss = _apply_weighting(F, xent_loss, self._weight, sample_weight)
        xent_loss = F.mean(xent_loss, axis=self._batch_axis, exclude=True)

        loss = l2loss + xent_loss

        if self._symmetric:
            similarity_matrix = F.dot(positives, anchors, transpose_a=False, transpose_b=True)
            xent_loss = F.sum(F.log_softmax(similarity_matrix, -1) * -labels_remapped, axis=-1, keepdims=True)
            xent_loss = _apply_weighting(F, xent_loss, self._weight, sample_weight)
            xent_loss = F.mean(xent_loss, axis=self._batch_axis, exclude=True)

            loss = (loss + l2loss + xent_loss) * 0.5

        return loss


class AngluarLossHinge(loss.Loss):
    """
    This is the hinge version of the angular loss
    """

    def __init__(self, alpha, angular_lambda, symmetric=False, npairparams=None, weight=None, batch_axis=0, **kwargs):
        super(AngluarLossHinge, self).__init__(weight, batch_axis, **kwargs)
        self._tansqralpha = np.tan(alpha) ** 2
        self._angular_lambda = angular_lambda
        self._symmetric = symmetric
        if angular_lambda > 0:
            assert npairparams is not None, 'npairparams must not be empty'
            self._npairloss = NPairsLoss(**npairparams)

    def hybrid_forward(self, F, anchors, positives, labels, sample_weight=None):
        """
        Computes the loss on the given data
        :param F: mx.nd or mx.sym
        :param anchors: anchor embeddings, <BxE> where B: batch size, E: embedding dimension
        :param positives: positive embeddings, same shape and label than anchors <BxE>
        :param labels: Labels of embeddings <B>
        :param sample_weight: weights of logits, see mx.loss
        :return:
        """
        # Get per pair similarities.
        labels_exp = labels.expand_dims(1)
        labels_remapped = 1 - F.broadcast_equal(labels_exp, labels_exp.transpose()).astype('float32')

        term1 = 4 * self._tansqralpha * F.dot(anchors + positives, positives, transpose_a=False, transpose_b=True)
        term2 = 2 * (1 + self._tansqralpha) * F.sum(anchors * positives, axis=-1)
        fapn = F.broadcast_sub(term1, term2.expand_dims(1)) * labels_remapped

        loss = F.mean(log_sum_exp(F, fapn, axis=-1))

        if self._angular_lambda > 0:
            loss = F.mean(self._npairloss(anchors, positives, labels, sample_weight)) + self._angular_lambda * loss

        return loss


class AngluarLoss(loss.Loss):
    """
    This is the CE version of the angular loss
    """

    def __init__(self, alpha, l2_reg, symmetric=False, weight=None, batch_axis=0, **kwargs):
        super(AngluarLoss, self).__init__(weight, batch_axis, **kwargs)
        self._tansqralpha = np.tan(alpha) ** 2
        self._symmetric = symmetric
        self._l2_reg = l2_reg

    def hybrid_forward(self, F, anchors, positives, labels, sample_weight=None):
        """
        Computes the loss on the given data
        :param F: mx.nd or mx.sym
        :param anchors: anchor embeddings, <BxE> where B: batch size, E: embedding dimension
        :param positives: positive embeddings, same shape and label than anchors <BxE>
        :param labels: Labels of embeddings <B>
        :param sample_weight: weights of logits, see mx.loss
        :return:
        """
        reg_anchor = F.mean(F.sum(anchors.square(), axis=1), axis=self._batch_axis, exclude=True)
        reg_positive = F.mean(F.sum(positives.square(), axis=1), axis=self._batch_axis, exclude=True)
        l2loss = self._l2_reg * (reg_anchor + reg_positive)

        xaTxp = F.dot(anchors, positives, transpose_a=False, transpose_b=True)
        label_eye = F.broadcast_equal(labels.expand_dims(1), labels.expand_dims(0)).astype('float32')
        sim_matrix_1 = F.broadcast_mul(2.0 * (1.0 + self._tansqralpha) * xaTxp, label_eye)

        xaPxpTxn = F.dot((anchors + positives), positives, transpose_a=False, transpose_b=True)
        sim_matrix_2 = F.broadcast_mul(4.0 * self._tansqralpha * xaPxpTxn, F.ones_like(xaPxpTxn) - label_eye)

        # similarity_matrix
        similarity_matrix = sim_matrix_1 + sim_matrix_2

        # do softmax cross-entropy
        pred = F.log_softmax(similarity_matrix, -1)
        xent_loss = -F.sum(pred * label_eye, axis=-1, keepdims=True)
        xent_loss = F.mean(xent_loss, axis=0, exclude=True)

        loss = l2loss + xent_loss

        if self._symmetric:
            xaTxp = F.dot(positives, anchors, transpose_a=False, transpose_b=True)
            sim_matrix_1 = F.broadcast_mul(2.0 * (1.0 + self._tansqralpha) * xaTxp, label_eye)

            xaPxpTxn = F.dot((positives + anchors), anchors, transpose_a=False, transpose_b=True)
            sim_matrix_2 = F.broadcast_mul(4.0 * self._tansqralpha * xaPxpTxn, F.ones_like(xaPxpTxn) - label_eye)

            similarity_matrix = sim_matrix_1 + sim_matrix_2

            pred = F.log_softmax(similarity_matrix, -1)
            xent_loss = -F.sum(pred * label_eye, axis=-1, keepdims=True)
            xent_loss = F.mean(xent_loss, axis=0, exclude=True)

            loss = (loss + l2loss + xent_loss) * 0.5

        return loss


def pairwise_distance(F, feature, feature2=None, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
        output[i, j] = || feature[i, :] - feature[j, :] ||_2
        Args:
          feature: 2-D NDArray or Symbol of size [number of data, feature dimension].
          squared: Boolean, whether or not to square the pairwise distances.
        Returns:
          pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    if feature2 is None:
        pairwise_distances_squared = (F.broadcast_add(F.sum(feature.square(), axis=1, keepdims=True),
                                                      F.sum(feature.transpose().square(), axis=[0],
                                                            keepdims=True))) - 2.0 * F.dot(
            feature, feature.transpose())
    else:
        pairwise_distances_squared = (F.broadcast_add(F.sum(feature.square(), axis=1, keepdims=True),
                                                      F.sum(feature2.transpose().square(), axis=0,
                                                            keepdims=True))) - 2.0 * F.dot(
            feature, feature2.transpose())

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = F.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared <= 0.0

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = F.sqrt(pairwise_distances_squared + error_mask * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = pairwise_distances * F.logical_not(error_mask)

    # Explicitly set diagonals to zero.
    if feature2 is None:
        mask_offdiagonals = F.ones_like(pairwise_distances) - F.diag(F.diag(F.ones_like(pairwise_distances)))
        pairwise_distances = pairwise_distances * mask_offdiagonals
    return pairwise_distances


def tile_square(F, data):
    """
    Tiles an NxN matrix into N^2xN
    :param F:
    :param data:
    :return:
    """
    e = F.expand_dims(data, axis=0)  # 1xNxN
    f = F.broadcast_like(e, data, lhs_axes=0, rhs_axes=0)  # NxNxN
    g = F.reshape(f, (-3, 0))  # N^2 x N
    return g


def masked_minimum(F, data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.
    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = F.max(data, dim, keepdims=True)
    masked_minimums = F.min((F.broadcast_sub(data, axis_maximums)) * mask, dim, keepdims=True) + axis_maximums
    return masked_minimums


def masked_maximum(F, data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.
    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = F.min(data, dim, keepdims=True)
    masked_maximums = F.max((F.broadcast_sub(data, axis_minimums)) * mask, dim, keepdims=True) + axis_minimums
    return masked_maximums


class TripletSemiHardLoss(loss.Loss):
    def __init__(self, margin=1, weight=None, batch_axis=0, **kwargs):
        super(TripletSemiHardLoss, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin

    def hybrid_forward(self, F, embeddings, labels):
        labels = labels.expand_dims(1)  # <B x 1>

        # Build pairwise squared distance matrix.
        pdist_matrix = pairwise_distance(F, embeddings, squared=True)  # BxB

        # Build pairwise binary adjacency matrix.
        adjacency = F.broadcast_equal(labels, labels.transpose())  # BxB
        # Invert so we can select negatives only.
        adjacency_not = F.logical_not(adjacency)

        # Compute the mask.
        pdist_matrix_tile = tile_square(F, pdist_matrix)  # B^2xB

        mask = F.broadcast_logical_and(tile_square(F, adjacency_not),
                                       (F.broadcast_greater(pdist_matrix_tile,
                                                            F.reshape(pdist_matrix.transpose(), (-1, 1)))).astype(
                                           'int64'))

        mask_final = F.reshape_like(F.sum(mask.astype('float32'), 1, keepdims=True) > 0.0, pdist_matrix)  # BxB
        mask_final = mask_final.transpose()

        adjacency_not = adjacency_not.astype('float32')
        mask = mask.astype('float32')

        # negatives_outside: smallest D_an where D_an > D_ap.
        negatives_outside = F.reshape_like(
            masked_minimum(F, pdist_matrix_tile, mask), pdist_matrix)
        negatives_outside = negatives_outside.transpose()  # BxB

        # negatives_inside: largest D_an.
        negatives_inside = F.broadcast_like(masked_maximum(F, pdist_matrix, adjacency_not), pdist_matrix)
        semi_hard_negatives = F.where(
            mask_final, negatives_outside, negatives_inside)

        loss_mat = self._margin + (pdist_matrix - semi_hard_negatives)

        mask_positives = adjacency.astype('float32') - F.diag(F.diag(F.ones_like(pdist_matrix)))

        # In lifted-struct, the authors multiply 0.5 for upper triangular
        #   in semihard, they take all positive pairs except the diagonal.
        num_positives = F.sum(mask_positives)
        # To avoid division by 0 when no positive pairs are present
        num_positives = F.maximum(num_positives, F.ones_like(num_positives))

        triplet_loss = F.sum(F.maximum(loss_mat * mask_positives, 0.0)) / num_positives

        return triplet_loss


class LiftedStructLoss(loss.Loss):
    def __init__(self, margin=1, weight=None, batch_axis=0, **kwargs):
        super(LiftedStructLoss, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin

    def hybrid_forward(self, F, embeddings, labels):
        labels = labels.expand_dims(1)  # Bx1

        # Build pairwise squared distance matrix.
        pairwise_distances = pairwise_distance(F, embeddings)  # BxB

        # Build pairwise binary adjacency matrix.
        adjacency = F.broadcast_equal(labels, labels.transpose())
        # Invert so we can select negatives only.
        adjacency_not = F.logical_not(adjacency)

        diff = self._margin - pairwise_distances  # BxB
        mask = adjacency_not.astype('float32')

        # Safe maximum: Temporarily shift negative distances
        #   above zero before taking max.
        #     this is to take the max only among negatives.
        row_minimums = F.min(diff, 1, keepdims=True)
        row_negative_maximums = F.max((F.broadcast_sub(diff, row_minimums)) * mask, 1, keepdims=True) + row_minimums

        # Compute the loss.
        # Keep track of matrix of maximums where M_ij = max(m_i, m_j)
        # where m_i is the max of alpha - negative D_i's.

        max_elements = F.broadcast_maximum(row_negative_maximums, row_negative_maximums.transpose())
        diff_tiled = tile_square(F, diff)
        mask_tiled = tile_square(F, mask)
        max_elements_vect = F.reshape(max_elements.transpose(), (-1, 1))

        loss_exp_left = F.reshape_like(
            F.sum(F.exp(F.broadcast_sub(diff_tiled, max_elements_vect)) * mask_tiled, 1, keepdims=True),
            pairwise_distances)

        loss_mat = max_elements + F.log(loss_exp_left + loss_exp_left.transpose())
        # Add the positive distance.
        loss_mat = loss_mat + pairwise_distances

        mask_positives = adjacency.astype('float32') - F.diag(F.diag(F.ones_like(pairwise_distances)))

        # *0.5 for upper triangular, and another *0.5 for 1/2 factor for loss^2.
        num_positives = F.sum(mask_positives) / 2.0
        # To avoid division by 0 when no positive pairs are present
        num_positives = F.maximum(num_positives, F.ones_like(num_positives))

        lifted_loss = 0.25 * F.sum(F.maximum(loss_mat * mask_positives, 0.0).square()) / num_positives
        return lifted_loss


class RankedListLoss(loss.Loss):
    def __init__(self, margin=1, alpha=1.2, temperature=10, weight=None, batch_axis=0, **kwargs):
        super(RankedListLoss, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin
        self._alpha = alpha
        self._temperature = temperature

    def hybrid_forward(self, F, embeddings, labels):
        labels = labels.expand_dims(1)  # <B x 1>

        # Build pairwise squared distance matrix.
        pdist_matrix = pairwise_distance(F, embeddings, squared=False)  # BxB

        # Build pairwise binary adjacency matrix.
        adjacency = F.broadcast_equal(labels, labels.transpose())  # BxB

        # Mine non-trivial positives (d > (alpha - margin))
        nontrivial_positives_mask = (
                    F.broadcast_greater(pdist_matrix, F.ones_like(pdist_matrix) * (self._alpha - self._margin)).astype(
                        'int32') * adjacency).astype('float32')  # BxB

        loss_positive = F.relu((nontrivial_positives_mask * pdist_matrix) - (self._alpha - self._margin))
        loss_positive = F.broadcast_div(F.sum(loss_positive, axis=1),
                                        F.maximum(F.sum(nontrivial_positives_mask, axis=1), 1))  # B

        # Mine non-trivial negatives (d < alpha)
        adjacency_not = F.logical_not(adjacency)
        nontrivial_negatives_mask = (F.broadcast_lesser(pdist_matrix, F.ones_like(pdist_matrix) * self._alpha).astype(
            'int32') * adjacency_not).astype('float32')  # BxB
        nontrivial_negatives = nontrivial_negatives_mask * pdist_matrix
        loss_negative = F.relu((self._alpha - pdist_matrix) * nontrivial_negatives_mask)

        # apply weighting
        weights = F.exp((self._alpha - nontrivial_negatives) * self._temperature) * nontrivial_negatives_mask
        sum_weights = F.sum(weights, axis=1, keepdims=True)
        weights = F.broadcast_div(weights, F.maximum(sum_weights, 1))
        loss_negative = F.sum(weights * loss_negative, axis=1)  # B

        loss = loss_positive + loss_negative
        num_nonnegative = F.sum(F.broadcast_greater(loss, F.zeros((1,))))
        num_nonnegative = F.maximum(num_nonnegative, 1)
        loss = F.broadcast_div(F.sum(loss), num_nonnegative)

        return loss


class DiscriminativeLoss(loss.Loss):
    def __init__(self, num_classes, num_samples, weight=None, batch_axis=0, **kwargs):
        super(DiscriminativeLoss, self).__init__(weight, batch_axis, **kwargs)
        self.centeroids = self.params.get_constant('centeroids', mx.nd.one_hot(mx.nd.arange(0, num_classes), num_classes))
        self.num_classes = num_classes
        self.num_samples = num_samples

    def hybrid_forward(self, F, embeddings, labels, negative_labels, centeroids):
        c = self.num_classes
        positive_centeroids = F.take(centeroids, labels)  # BxC
        p_dist = F.sum(F.square(embeddings - positive_centeroids), axis=self._batch_axis, exclude=True)  # B

        negative_centeroids = F.take(centeroids, negative_labels)  # Bx(C-1) x C
        embeddings_b = F.repeat(embeddings, repeats=c - 1, axis=0)  # B*(C-1) x C
        negative_centeroids_b = F.reshape_like(negative_centeroids, embeddings_b)  # B*(C-1) x C
        n_dist = F.sum(F.square(embeddings_b - negative_centeroids_b), axis=self._batch_axis, exclude=True)  # B*(C-1)
        n_dist = F.reshape_like(n_dist, negative_centeroids, lhs_begin=0, lhs_end=1, rhs_begin=0, rhs_end=2)  # Bx(C-1)
        n_dist = F.sum(n_dist, axis=self._batch_axis, exclude=True)  # B

        loss = F.sum(p_dist - (1 / (3 * (c - 1)) * n_dist))  # 1
        n = F.size_array(p_dist).astype('float32')  # self.num_samples
        g = 3 * (c - 1) * ((n / c) - 1) * (n / c)
        loss = g * loss
        return loss
