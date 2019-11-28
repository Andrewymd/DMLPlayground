from models.backbones import get_inception_bn, get_googlenet
from models.marginmodels import MarginNet, MarginLoss
from models.proxynca import ProxyNet, NormProxyNet
from models.simplemodels import EmbeddingNet, SoftmaxNet


def get_feature_model(model_name, **kwargs):
    if model_name == 'inception-bn':
        return get_inception_bn()
    elif model_name == 'googlenet':
        return get_googlenet()
    else:
        raise RuntimeError('Unknown model: %s' % model_name)
