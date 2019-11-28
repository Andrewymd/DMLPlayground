import mxnet as mx
from mxnet.gluon import nn, HybridBlock

from common.initializers import TruncNorm, NormalScaled
from models.simplemodels import L2Normalization, WeightNormalizedDense, ScaleLayer


class ProxyNet(HybridBlock):
    def __init__(self, base_net, emb_dim, num_classes, normalize=True, K=1,  **kwargs):
        super(ProxyNet, self).__init__(**kwargs)
        self._num_classes = num_classes
        self._K = K

        with self.name_scope():
            self.base_net = base_net
            self.encoder = nn.HybridSequential()
            self.encoder.add(nn.Dense(emb_dim, prefix='embeddingLayer'))
            if normalize:
                self.encoder.add(L2Normalization(mode='instance'))

            self.proxies = nn.Embedding(K * num_classes, emb_dim)

    def hybrid_forward(self, F, x, label, negative_labels):
        z = self.base_net(x)
        y = self.encoder(z)
        proxies = self.proxies(F.arange(0, self._K * self._num_classes))
        proxies = F.L2Normalization(proxies, mode='instance')
        return y, self.proxies(label), self.proxies(negative_labels), proxies

    def init(self, initializer=TruncNorm(stdev=0.001), init_basenet=True, ctx=mx.cpu()):
        if init_basenet:
            self.initialize(initializer, ctx=ctx)
        else:
            self.encoder.initialize(initializer, ctx=ctx, force_reinit=True)
            self.proxies.initialize(NormalScaled(8), ctx=ctx, force_reinit=True)


class NormProxyNet(HybridBlock):
    def __init__(self, base_net, emb_dim, num_classes, feature_size, no_fc=False, dropout=True, static_proxies=False, **kwargs):
        super(NormProxyNet, self).__init__(**kwargs)
        self._num_classes = num_classes
        self._static_proxies = static_proxies
        self._embed_dim = emb_dim

        with self.name_scope():
            self.base_net = base_net
            self.encoder = nn.HybridSequential()
            self.encoder.add(nn.LayerNorm())
            if not no_fc:
                self.encoder.add(WeightNormalizedDense(emb_dim, prefix='embeddingLayer', use_bias=False, in_units=feature_size))
                if dropout:
                    self.encoder.add(nn.Dropout(0.5))
            self.encoder.add(L2Normalization(mode='instance'))
            if not static_proxies:
                self.proxies = nn.Embedding(num_classes, emb_dim)

    def hybrid_forward(self, F, x):
        z = self.base_net(x)
        if self._static_proxies:
            proxies = F.one_hot(F.arange(0, self._num_classes), self._embed_dim)
            proxies = F.L2Normalization(proxies)
            return self.encoder(z), proxies
        proxies = self.proxies(F.arange(0, self._num_classes))
        proxies = F.L2Normalization(proxies)
        return self.encoder(z), proxies

    def init(self, initializer=mx.init.Xavier(magnitude=0.2), init_basenet=True, ctx=mx.cpu()):
        if init_basenet:
            self.initialize(mx.init.Xavier(magnitude=0.2), ctx=ctx)
        else:
            # do not initialize base network
            self.encoder.initialize(initializer, ctx=ctx, force_reinit=True)
            if not self._static_proxies:
                self.proxies.initialize(NormalScaled(8), ctx=ctx, force_reinit=True)

    def get_proxies(self, F=mx.nd, normalized=True):
        proxies = self.proxies(F.arange(0, self._num_classes))
        if normalized:
            proxies = F.L2Normalization(proxies)
        return proxies
