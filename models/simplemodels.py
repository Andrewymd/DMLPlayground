import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.nn import Dropout

from common.initializers import TruncNorm


class ScaleLayer(HybridBlock):
    def __init__(self, init_value=1, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        self.beta = self.params.get('beta', init=mx.init.Constant(init_value), shape=(1,))

    def hybrid_forward(self, F, x, beta):
        return F.broadcast_mul(x, beta)


class L2Normalization(HybridBlock):
    r"""Applies L2 Normalization to input.

    Parameters
    ----------
    mode : str
        Mode of normalization.
        See :func:`~mxnet.ndarray.L2Normalization` for available choices.

    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, mode, **kwargs):
        self._mode = mode
        super(L2Normalization, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.L2Normalization(x, mode=self._mode, name='l2_norm')

    def __repr__(self):
        s = '{name}({_mode})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)


class EmbeddingNet(HybridBlock):
    def __init__(self, base_net, emb_dim, normalize=True, dropout=True, **kwargs):
        super(EmbeddingNet, self).__init__(**kwargs)

        if isinstance(emb_dim, (tuple, list)):
            assert len(emb_dim) > 0
            if len(emb_dim) == 1:
                emb_dim = emb_dim[0]

        with self.name_scope():
            self.base_net = base_net
            self.encoder = nn.HybridSequential()
            if isinstance(emb_dim, (tuple, list)):
                for i, dim in enumerate(emb_dim[:-1]):
                    self.encoder.add(nn.Dense(dim, prefix='bottlneckLayer%d' % i))
                    if dropout:
                        self.encoder.add(nn.Dropout(0.5))
                self.encoder.add(nn.Dense(emb_dim[-1], prefix='embeddingLayer', use_bias=not normalize))
            else:
                self.encoder.add(nn.Dense(emb_dim, prefix='embeddingLayer', use_bias=not normalize))
            if normalize:
                self.encoder.add(L2Normalization(mode='instance'))

    def hybrid_forward(self, F, x):
        x = self.base_net(x)
        return self.encoder(x)

    def init(self, initializer=TruncNorm(stdev=0.001), init_basenet=True, ctx=mx.cpu()):
        if init_basenet:
            self.initialize(mx.init.Xavier(magnitude=0.2), ctx=ctx, force_reinit=True)
        else:
            self.encoder.initialize(initializer, ctx=ctx, force_reinit=True)


class SoftmaxNet(HybridBlock):
    def __init__(self, base_net, emb_dim, classes, dropout=True, **kwargs):
        super(SoftmaxNet, self).__init__(**kwargs)

        with self.name_scope():
            self.base_net = base_net
            self.encoder = nn.HybridSequential()
            self.encoder.add(nn.Dense(emb_dim, prefix='embeddingLayer'))

            self.predictor = nn.HybridSequential()
            self.predictor.add(nn.Activation('relu'))
            if dropout:
                self.predictor.add(nn.Dropout(0.5))
            self.predictor.add(nn.Dense(classes))

    def hybrid_forward(self, F, x):
        x = self.base_net(x)
        emb = self.encoder(x)
        pred = self.predictor(emb)
        return pred, emb

    def init(self, initializer=TruncNorm(stdev=0.001), init_basenet=True, ctx=mx.cpu()):
        if init_basenet:
            self.initialize(mx.init.Xavier(magnitude=0.2), ctx=ctx, force_reinit=True)
        else:
            self.encoder.initialize(initializer, ctx=ctx, force_reinit=True)
            self.predictor.initialize(initializer, ctx=ctx, force_reinit=True)


class WeightNormalizedDense(HybridBlock):
    """
    Dense layer with weight normalization
    """
    def __init__(self, units, activation=None, use_bias=True, flatten=True,
                 dtype='float32', weight_initializer=None, bias_initializer='zeros',
                 in_units=0, **kwargs):
        super(WeightNormalizedDense, self).__init__(**kwargs)
        self._flatten = flatten
        with self.name_scope():
            self._units = units
            self._in_units = in_units
            self.weight_v = self.params.get('weight_v', shape=(units, in_units),
                                            init=weight_initializer, dtype=dtype,
                                            allow_deferred_init=True)
            self.weight_g = self.params.get('weight_g', shape=(1,),
                                            init='ones', dtype=dtype,
                                            allow_deferred_init=True)
            if use_bias:
                self.bias = self.params.get('bias', shape=(units,),
                                            init=bias_initializer, dtype=dtype,
                                            allow_deferred_init=True)
            else:
                self.bias = None
            if activation is not None:
                self.act = nn.Activation(activation, prefix=activation + '_')
            else:
                self.act = None

    def hybrid_forward(self, F, x, weight_v, weight_g, bias=None):
        weight = F.broadcast_like(F.expand_dims(weight_g / F.sqrt(F.sum(weight_v.square()) + 1e-7), 0), weight_v) * weight_v
        act = F.FullyConnected(x, weight, bias, no_bias=bias is None, num_hidden=self._units,
                               flatten=self._flatten, name='fwd')
        # do normalization after FC so shape can be inferend. Do the math.
        if self.act is not None:
            act = self.act(act)
        return act

    def __repr__(self):
        s = '{name}({layout}, {act})'
        shape = self.weight_v.shape
        return s.format(name=self.__class__.__name__,
                        act=self.act if self.act else 'linear',
                        layout='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]))
