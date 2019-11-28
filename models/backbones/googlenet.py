from os.path import dirname, realpath, join

import mxnet as mx


def get_googlenet():
    """Model is converted from ONNX using the source here:
        https://github.com/onnx/models/tree/master/vision/classification/inception_and_googlenet/googlenet
    """
    sym = mx.symbol.load(join(dirname(realpath(__file__)), '../../symbols/GoogleNet/googlenet.sym'))

    new_sym = sym.get_internals()['flatten0_output']
    net = mx.gluon.nn.SymbolBlock(outputs=new_sym, inputs=mx.sym.var('data_0', dtype='float32'))

    return net, '../symbols/GoogleNet/googlenet.params'
