from mxnet.gluon.loss import Loss, _apply_weighting


class PerceptualTripletLoss(Loss):
    def __init__(self, simblocks, l2_reg, symmetric=False, weight=None, batch_axis=0, **kwargs):
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
        super(PerceptualTripletLoss, self).__init__(weight, batch_axis, **kwargs)
        self._simblocks = simblocks
        self._l2_reg = l2_reg
        self._symmetric = symmetric

    def hybrid_forward(self, F, labels, *args, **kwargs):
        """
        Computes the loss on the given data
        :param F: mx.nd or mx.sym
        :param anchors: anchor embeddings, <BxE> where B: batch size, E: embedding dimension
        :param positives: positive embeddings, same shape and label than anchors <BxE>
        :param labels: Labels of embeddings <B>
        :param sample_weight: weights of logits, see mx.loss
        :return:
        """
        block_size = len(args) / 2
        anchors = list(args[:block_size])
        positives = list(args[block_size:])

        # flatten last here
        reg_anchor = F.mean(F.sum(anchors[-1].square(), axis=1), axis=self._batch_axis, exclude=True)
        reg_positive = F.mean(F.sum(positives[-1].square(), axis=1), axis=self._batch_axis, exclude=True)
        l2loss = self._l2_reg * (reg_anchor + reg_positive)

        # Get per pair similarities.
        perceptual_similarity_matrix = [self._simblocks[i](a, p).expand_dims(0) for i, (a, p) in enumerate(zip(anchors[:-1], positives[:-1]))]
        perceptual_similarity_matrix = F.concat(*perceptual_similarity_matrix, dim=0)
        perceptual_similarity_matrix = F.sum(perceptual_similarity_matrix, axis=0)

        # Get npairs similarity matrix
        similarity_matrix = F.dot(anchors[-1], positives[-1], transpose_a=False, transpose_b=True)

        labels = labels.expand_dims(1)

        labels_remapped = F.broadcast_equal(labels, labels.transpose())
        labels_remapped = F.broadcast_div(labels_remapped, F.sum(labels_remapped, axis=1, keepdims=True))
        labels_remapped = labels_remapped.astype(dtype='float32')

        # Add the softmax loss
        xent_loss = F.sum(F.log_softmax(similarity_matrix, -1) * -labels_remapped, axis=-1, keepdims=True)
        xent_loss = _apply_weighting(F, xent_loss, self._weight, kwargs.get('sample_weight'))
        xent_loss = F.mean(xent_loss, axis=self._batch_axis, exclude=True)

        # Add the perceptual softmax loss
        perc_xent_loss = F.sum(F.log_softmax(perceptual_similarity_matrix, -1) * -labels_remapped, axis=-1, keepdims=True)
        perc_xent_loss = _apply_weighting(F, perc_xent_loss, self._weight, kwargs.get('sample_weight'))
        perc_xent_loss = F.mean(perc_xent_loss, axis=self._batch_axis, exclude=True)

        loss = (xent_loss + perc_xent_loss) * 0.5

        if self._symmetric:
            perceptual_similarity_matrix = [self._simblocks[i](a, p).expand_dims(0) for i, (a, p) in
                                            enumerate(zip(positives[:-1], anchors[:-1]))]
            perceptual_similarity_matrix = F.concat(*perceptual_similarity_matrix, dim=0)
            perceptual_similarity_matrix = F.sum(perceptual_similarity_matrix, axis=0)

            similarity_matrix = F.dot(positives, anchors, transpose_a=False, transpose_b=True)

            xent_loss = F.sum(F.log_softmax(similarity_matrix, -1) * -labels_remapped, axis=-1, keepdims=True)
            xent_loss = _apply_weighting(F, xent_loss, self._weight, kwargs.get('sample_weight'))
            xent_loss = F.mean(xent_loss, axis=self._batch_axis, exclude=True)

            perc_xent_loss = F.sum(F.log_softmax(perceptual_similarity_matrix, -1) * -labels_remapped, axis=-1,
                                   keepdims=True)
            perc_xent_loss = _apply_weighting(F, perc_xent_loss, self._weight, kwargs.get('sample_weight'))
            perc_xent_loss = F.mean(perc_xent_loss, axis=self._batch_axis, exclude=True)

            loss = loss + (xent_loss + perc_xent_loss) * 0.5
            loss = loss * 0.5

        return loss + l2loss


# def perceptual_pairwise_similarity(F, x, y):
#     """
#     Computes the perceptual similarity matrix
#     :param F:
#     :param x: tuple of L1
#     :param y: tuple of L2
#     :return:
#     """
#     # # Unit normalize in the channel dimension
#     # for i in range(len(x)):
#     #     x[i] = x[i] / F.norm(x[i], ord=1, axis=1).expand_dims(1)
#     # for i in range(len(y)):
#     #     y[i] = y[i] / F.norm(y[i], ord=1, axis=1).expand_dims(1)
#
#     # compute pairwise difference
#     diff = []
#     for a, b in zip(x,y):
#         aa = a.expand_dims(1)  # Bx1xCxHxW
#         bb = b.expand_dims(0)  # 1xBxCxHxW
#         cc = F.broadcast_sub(aa, bb)  # BxBxCxHxW
#         diff.append(cc)
#
#     # multiply with w
#     # for i in range(len(diff)):
#     #     diff[i] = F.broadcast_mul(diff[i], F.reshape(w[i],(1,1,-1,1,1)))
#
#     # compute l2 norm
#     for i in range(len(diff)):
#         diff[i] = F.norm(diff[i], ord=2, axis=2)
#
#     # spatial average
#     for i in range(len(diff)):
#         d_reshaped = F.reshape(F.reshape(diff[i], (0, 0, -1)), (-1, 0), reverse=True)  # (B*B)x(H*W)
#         d_mean = F.mean(d_reshaped, axis=1)  # B*B
#         diff[i] = F.reshape_like(d_mean, diff[i], lhs_begin=0, lhs_end=1, rhs_begin=0, rhs_end=2).expand_dims(0)
#
#     diff = F.sum(F.concat(*diff, dim=0), axis=0)
#
#     return diff
