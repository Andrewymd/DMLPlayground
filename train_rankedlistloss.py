from __future__ import division

import logging

import mxnet as mx
import numpy as np
from mxnet import autograd as ag
from tqdm import tqdm

from common.evaluate import evaluate
from common.parser import TrainingParser
from common.utils import average_results, get_lr, get_context, parse_steps, format_results, append_postfix
from dataset import get_dataset, get_dataset_iterator
from dataset.dataloader import DatasetIterator
from models import get_feature_model, EmbeddingNet
from models.loss import RankedListLoss


def parse_args():
    parser = TrainingParser(description='Distance metric learning using ranked list loss with semihard mining',
                            default_logfile='train_rankedlist.log',
                            default_model_prefix='rankedlist_model')
    parser.add_argument('--batch-size', type=int, default=180,
                        help='Number of samples in a batch. Default is 180')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of training epochs. default is 60.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate. default is 0.0001.')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='learning rate schedule factor. default is 0.5.')
    parser.add_argument('--wd', type=float, default=0.00001,
                        help='weight decay rate. default is 0.00001.')
    parser.add_argument('--steps', type=str, default='20,30,40',
                        help='epochs to update learning rate. default is 20,30,40.')
    parser.add_argument('--batch-k', type=int, default=3, help='Number of images per class')
    parser.add_argument('--alpha', type=float, default=1.2, help='Margin for negatives')
    parser.add_argument('--margin', type=float, default=0.4, help='Margin for positives')
    parser.add_argument('--temperature', type=float, default=10, help='Temperature for negatives')
    parser.add_argument('--iteration-per-epoch', type=int, default=200,
                        help='Number of iterations per epoch for iteration-based training')
    parser.add_argument('--bottleneck-layers', type=str, default='')

    opt = parser.parse_args()

    assert opt.batch_size % opt.batch_k == 0, 'Batch size must be divisible by batch-k'

    if opt.logfile.lower() != 'none':
        logging.basicConfig(filename=append_postfix(opt.logfile, opt.log_postfix), level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())

    return opt


def validate(net, val_data, ctx, use_threads=True, nmi=True):
    """Test a model."""
    outputs = []
    labels = []
    ctx_cpu = mx.cpu()

    for batch in tqdm(val_data, desc='Computing test embeddings'):
        data = mx.gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = mx.gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

        for x in data:
            outputs.append(net(x).as_in_context(ctx_cpu))
        labels += [l.as_in_context(ctx_cpu) for l in label]

    outputs = mx.nd.concatenate(outputs, axis=0)
    labels = mx.nd.concatenate(labels, axis=0)
    return evaluate(outputs, labels, val_data._dataset.num_classes(), use_threads=use_threads, nmi=nmi)


def train(net, opt, train_dataloader, val_dataloader, context, run_id):
    """Training function."""
    if not opt.skip_pretrain_validation:
        validation_results = validate(net, val_dataloader, context, use_threads=opt.num_workers > 0, nmi=opt.nmi)
        for name, val_acc in validation_results:
            logging.info('Pre-train validation: %s=%f' % (name, val_acc))

    steps = parse_steps(opt.steps, opt.epochs, logging)

    opt_options = {'learning_rate': opt.lr,
                   'wd': opt.wd,
                   }
    if opt.optimizer == 'sgd':
        opt_options['momentum'] = 0.9
    if opt.optimizer == 'adam':
        opt_options['epsilon'] = 1e-7
    trainer = mx.gluon.Trainer(net.collect_params(), opt.optimizer, opt_options, kvstore=opt.kvstore)

    L = RankedListLoss(margin=opt.margin, alpha=opt.alpha, temperature=opt.temperature)
    if not opt.disable_hybridize:
        L.hybridize()

    smoothing_constant = .01  # for tracking moving losses
    moving_loss = 0
    best_results = []  # R@1, NMI

    for epoch in range(1, opt.epochs + 1):
        p_bar = tqdm(enumerate(train_dataloader), desc='[Run %d/%d] Epoch %d' % (run_id, opt.number_of_runs, epoch),
                     total=len(train_dataloader))
        trainer.set_learning_rate(get_lr(opt.lr, epoch, steps, opt.factor))

        for i, (data, labels) in p_bar:
            data = data[0].as_in_context(context[0])
            labels = labels[0].astype('int32').as_in_context(context[0])

            with ag.record():
                losses = []
                embs = net(data)
                losses.append(L(embs, labels))
            for l in losses:
                l.backward()

            trainer.step(1)

            #  Keep a moving average of the losses
            curr_loss = mx.nd.mean(mx.nd.concatenate(losses)).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (epoch == 1))  # starting value
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)  # add current
            p_bar.set_postfix_str('Moving loss: %.4f' % moving_loss)

        logging.info('Moving loss: %.4f' % moving_loss)
        validation_results = validate(net, val_dataloader, context, use_threads=opt.num_workers > 0, nmi=opt.nmi)
        for name, val_acc in validation_results:
            logging.info('[Epoch %d] validation: %s=%f' % (epoch, name, val_acc))

        if (len(best_results) == 0) or (validation_results[0][1] > best_results[0][1]):
            best_results = validation_results
            if opt.save_model_prefix.lower() != 'none':
                filename = '%s.params' % opt.save_model_prefix
                logging.info('Saving %s.' % filename)
                net.save_parameters(filename)
            logging.info('New best validation: R@1: %f NMI: %f' % (best_results[0][1], best_results[-1][1]))

    return best_results


def train_rankedlist(opt):
    logging.info(opt)

    # Settings.
    mx.random.seed(opt.seed)
    np.random.seed(opt.seed)

    # Setup computation context
    context = get_context(opt.gpus, logging)

    run_results = []

    # Get model
    if opt.model == 'inception-bn':
        feature_net, feature_params = get_feature_model(opt.model, ctx=context)
        feature_net.collect_params().load(feature_params, ctx=context, ignore_extra=True)
        data_shape = 224
        scale_image_data = False
    elif opt.model == 'resnet50_v2':
        feature_net = mx.gluon.model_zoo.vision.resnet50_v2(pretrained=True, ctx=context).features
        data_shape = 224
        scale_image_data = True
        feature_params = None
    else:
        raise RuntimeError('Unsupported model: %s' % opt.model)

    if opt.bottleneck_layers != '':
        embedding_layers = [int(x) for x in opt.bottleneck_layers.split(',')] + [opt.embed_dim]
    else:
        embedding_layers = [opt.embed_dim]
    logging.info('Embedding layers: [%s]' % ','.join([str(x) for x in embedding_layers]))
    if len(embedding_layers) == 1:
        embedding_layers = embedding_layers[0]

    net = EmbeddingNet(feature_net, embedding_layers, normalize=True, dropout=False)
    logging.info(net)

    if opt.model == 'resnet50_v2':
        # Use a smaller learning rate for pre-trained convolutional layers.
        for v in net.base_net.collect_params().values():
            if 'conv' in v.name:
                setattr(v, 'lr_mult', 0.01)
            elif 'batchnorm' in v.name or 'bn_' in v.name:
                v.grad_req = 'null'
    else:
        for v in net.encoder.collect_params().values():
            setattr(v, 'lr_mult', 10.)

    # Get data iterators
    train_dataset = DatasetIterator(get_dataset_iterator(opt.dataset, opt.data_path, batch_k=opt.batch_k,
                                                         batch_size=opt.batch_size, batchify=False,
                                                         data_shape=data_shape, use_crops=opt.use_crops,
                                                         scale_image_data=scale_image_data)[0], opt.iteration_per_epoch,
                                    'next')

    train_dataiterator = mx.gluon.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                                  num_workers=opt.num_workers,
                                                  last_batch='keep')

    val_dataset = get_dataset(opt.dataset, opt.data_path, data_shape=data_shape, use_crops=opt.use_crops,
                              use_aug=True,
                              scale_image_data=scale_image_data)[1]
    logging.info(
        'Training with %d classes, validating with %d classes' % (
            train_dataset.data_iterator.num_classes(), val_dataset.num_classes()))

    val_dataloader = mx.gluon.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                                              num_workers=opt.num_workers,
                                              last_batch='keep')

    # main run loop for multiple training runs
    for run in range(1, opt.number_of_runs + 1):
        logging.info('Starting run %d/%d' % (run, opt.number_of_runs))

        net.init(mx.init.Xavier(magnitude=0.2), ctx=context, init_basenet=False)

        if opt.model == 'inception-bn':
            net.base_net.collect_params().load(feature_params, ctx=context, ignore_extra=True)
        elif opt.model == 'resnet50_v2':
            net.base_net = mx.gluon.model_zoo.vision.resnet50_v2(pretrained=True, ctx=context).features
        else:
            raise RuntimeError('Unsupported model: %s' % opt.model)
        if not opt.disable_hybridize:
            net.hybridize()

        run_result = train(net, opt, train_dataiterator, val_dataloader, context, run)
        run_results.append(run_result)
        logging.info('Run %d finished with %f' % (run, run_result[0][1]))

    logging.info(
        'Average validation of %d runs:\n%s' % (opt.number_of_runs, format_results(average_results(run_results))))


if __name__ == '__main__':
    train_rankedlist(parse_args())
