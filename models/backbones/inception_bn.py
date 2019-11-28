from os.path import dirname, realpath, join

import mxnet as mx
from mxnet.gluon import HybridBlock, SymbolBlock

__all__ = ['get_inception_bn']


def get_inception_bn():
    param_path = join(dirname(realpath(__file__)), '../../symbols/Inception-BN/Inception-BN')
    sym, arg_params, aux_params = mx.model.load_checkpoint(param_path, 126)
    new_sym = sym.get_internals()['flatten_output']
    net = mx.gluon.nn.SymbolBlock(outputs=new_sym, inputs=mx.sym.var('data', dtype='float32'))
    return net, param_path + '-0126.params'


def get_inception_all_blocks():
    net = InceptionBNStages()
    return net, 'symbols/Inception-BN-0126.params'


eps = 1e-10 + 1e-5
bn_mom = 0.9
fix_gamma = False


def get_conv(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix='', attr={}):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    bn = mx.symbol.BatchNorm(data=conv, fix_gamma=fix_gamma, eps=eps, momentum=bn_mom, name='bn_%s%s' %(name, suffix))
    act = mx.symbol.Activation(data=bn, act_type='relu', name='relu_%s%s' %(name, suffix), attr=attr)
    return act


def get_inception_a(data, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3, pool, proj, name):
    # 1x1
    c1x1 = get_conv(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = get_conv(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = get_conv(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = get_conv(data=data, num_filter=num_d3x3red, kernel=(1, 1), name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = get_conv(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = get_conv(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = get_conv(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd3x3, cproj], name='ch_concat_%s_chconcat' % name)
    return concat


def get_inception_b(data, num_3x3red, num_3x3, num_d3x3red, num_d3x3, name):
    # 3x3 reduce + 3x3
    c3x3r = get_conv(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = get_conv(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = get_conv(data=data, num_filter=num_d3x3red, kernel=(1, 1),  name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = get_conv(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = get_conv(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="max", name=('max_pool_%s_pool' % name))
    # concat
    concat = mx.symbol.Concat(*[c3x3, cd3x3, pooling], name='ch_concat_%s_chconcat' % name)
    return concat


def get_inception_symbol(num_classes=1000, return_stages=False):
    # data
    data = mx.symbol.Variable(name="data")

    # stage 1
    conv1 = get_conv(data=data, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3), name='1')
    pool1 = mx.symbol.Pooling(data=conv1, kernel=(3, 3), stride=(2, 2), name='pool_1', pool_type='max')
    # stage 2
    conv2red = get_conv(data=pool1, num_filter=64, kernel=(1, 1), stride=(1, 1), name='2_red')
    conv2 = get_conv(data=conv2red, num_filter=192, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='2')
    pool2 = mx.symbol.Pooling(data=conv2, kernel=(3, 3), stride=(2, 2), name='pool_2', pool_type='max')
    # stage 2
    in3a = get_inception_a(pool2, 64, 64, 64, 64, 96, "avg", 32, '3a')
    in3b = get_inception_a(in3a, 64, 64, 96, 64, 96, "avg", 64, '3b')
    in3c = get_inception_b(in3b, 128, 160, 64, 96, '3c')
    # stage 3
    in4a = get_inception_a(in3c, 224, 64, 96, 96, 128, "avg", 128, '4a')
    in4b = get_inception_a(in4a, 192, 96, 128, 96, 128, "avg", 128, '4b')
    in4c = get_inception_a(in4b, 160, 128, 160, 128, 160, "avg", 128, '4c')
    in4d = get_inception_a(in4c, 96, 128, 192, 160, 192, "avg", 128, '4d')
    in4e = get_inception_b(in4d, 128, 192, 192, 256, '4e')
    # stage 4
    in5a = get_inception_a(in4e, 352, 192, 320, 160, 224, "avg", 128, '5a')
    in5b = get_inception_a(in5a, 352, 192, 320, 192, 224, "max", 128, '5b')
    # global avg pooling
    pool = mx.symbol.Pooling(data=in5b, kernel=(7, 7), stride=(1, 1), name="global_pool", pool_type='avg')

    # linear classifier
    flatten = mx.symbol.Flatten(data=pool)
    if num_classes > 0:
        fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes)
        softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
        if return_stages:
            return softmax, pool1, pool2, in3c, in4e, in5b
        else:
            return softmax
    else:
        if return_stages:
            #return pool1, pool2, in3c, in4e, in5b, flatten
            return mx.symbol.Group([pool1, pool2, in3c, in4e, in5b, flatten])
        else:
            return flatten


class InceptionBNStages(HybridBlock):
    def __init__(self, **kwargs):
        super(InceptionBNStages, self).__init__(**kwargs)

        # data
        data = mx.symbol.Variable(name="data")

        # stage 1
        conv1 = get_conv(data=data, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3), name='1')
        pool1 = mx.symbol.Pooling(data=conv1, kernel=(3, 3), stride=(2, 2), name='pool_1', pool_type='max')
        conv2red = get_conv(data=pool1, num_filter=64, kernel=(1, 1), stride=(1, 1), name='2_red')
        conv2 = get_conv(data=conv2red, num_filter=192, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='2')

        # stage 2
        pool2 = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), name='pool_2', pool_type='max')
        in3a = get_inception_a(pool2, 64, 64, 64, 64, 96, "avg", 32, '3a')
        in3b = get_inception_a(in3a, 64, 64, 96, 64, 96, "avg", 64, '3b')
        in3c = get_inception_b(in3b, 128, 160, 64, 96, '3c')

        # stage 3
        in4a = get_inception_a(data, 224, 64, 96, 96, 128, "avg", 128, '4a')
        in4b = get_inception_a(in4a, 192, 96, 128, 96, 128, "avg", 128, '4b')
        in4c = get_inception_a(in4b, 160, 128, 160, 128, 160, "avg", 128, '4c')
        in4d = get_inception_a(in4c, 96, 128, 192, 160, 192, "avg", 128, '4d')
        in4e = get_inception_b(in4d, 128, 192, 192, 256, '4e')

        # stage 4
        in5a = get_inception_a(data, 352, 192, 320, 160, 224, "avg", 128, '5a')
        in5b = get_inception_a(in5a, 352, 192, 320, 192, 224, "max", 128, '5b')

        # stage 5
        pool = mx.symbol.Pooling(data=data, kernel=(7, 7), stride=(1, 1), name="global_pool", pool_type='avg')
        flatten = mx.symbol.Flatten(data=pool)

        self._stage_1 = SymbolBlock(outputs=conv2, inputs=mx.sym.var('data'))
        self._stage_2 = SymbolBlock(outputs=in3c, inputs=mx.sym.var('data'))
        self._stage_3 = SymbolBlock(outputs=in4e, inputs=mx.sym.var('data'))
        self._stage_4 = SymbolBlock(outputs=in5b, inputs=mx.sym.var('data'))
        self._stage_5 = SymbolBlock(outputs=flatten, inputs=mx.sym.var('data'))

    def hybrid_forward(self, F, x):
        stage1 = self._stage_1(x)
        stage2 = self._stage_2(stage1)
        stage3 = self._stage_3(stage2)
        stage4 = self._stage_4(stage3)
        stage5 = self._stage_5(stage4)

        return stage1, stage2, stage3, stage4, stage5

    def load(self, param_file, ctx=mx.cpu()):
        self._stage_1.collect_params().load(param_file, ctx=ctx, ignore_extra=True)
        self._stage_2.collect_params().load(param_file, ctx=ctx, ignore_extra=True)
        self._stage_3.collect_params().load(param_file, ctx=ctx, ignore_extra=True)
        self._stage_4.collect_params().load(param_file, ctx=ctx, ignore_extra=True)
        self._stage_5.collect_params().load(param_file, ctx=ctx, ignore_extra=True)
