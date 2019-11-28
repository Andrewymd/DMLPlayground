import logging

import mxnet as mx
import numpy as np
from mxnet import autograd as ag
from tqdm import tqdm

from common.evaluate import evaluate
from common.parser import TrainingParser
from common.utils import average_results, get_context, parse_steps, get_lr, format_results, append_postfix
from dataset import get_dataset, get_dataset_iterator
from dataset.dataloader import DatasetIterator
from models import get_feature_model, ProxyNet
from models.loss import ProxyNCALoss, ProxyTripletLoss
from models.simplemodels import TruncNorm


def parse_args():
    parser = TrainingParser(description='Distance metric learning using proxies',
                            default_logfile='train_proxy.log',
                            default_model_prefix='proxy_model')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU). default is 32.')
    parser.add_argument('--batch-k', type=int, default=5,
                        help='Number of images per class in a batch. Used only if iteration-per-epoch > 0. Default is 5.')
    parser.add_argument('--loss', type=str, default='triplet',
                        help='Which loss to use: [nca, triplet, proxymargin, xentropy]')
    parser.add_argument('--epsilon', type=float, default=1e-2,
                        help='Optimizer epsilon. default is 0.01.')
    parser.add_argument('--lr', default=None, type=float,
                        help='Learning rate for the whole model. Overwrites specific learning rates.')
    parser.add_argument('--lr-embedding', default=1e-5, type=float,
                        help='Learning rate for embedding.')
    parser.add_argument('--lr-inception', default=1e-3, type=float,
                        help='Learning rate for Inception, excluding embedding layer.')
    parser.add_argument('--lr-proxynca', default=1e-3, type=float,
                        help='Learning rate for proxies of Proxy NCA.')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='weight decay rate. default is 5e-4.')
    parser.add_argument('--factor', type=float, default=1e-1,
                        help='learning rate schedule factor. default is 1e-1.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs. default is 20.')
    parser.add_argument('--steps', type=str, default='3,10,16',
                        help='Epochs to update learning rate. Negative number represents periodic decrease.'
                             'Zero means no steps. Default is 3,10,16')
    parser.add_argument('--iteration-per-epoch', type=int, default=0,
                        help='Number of iterations per epoch for iteration-based training')
    parser.add_argument('--label-smooth', type=float, default=0,
                        help='Label smoothing. Default is 0')
    parser.add_argument('--embedding-multiplier', type=float, default=3,
                        help='Multiplies normalized embeddings and proxies. Default is 3')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature scaling for NCA and XEntropy losses. Default is 1')

    opt = parser.parse_args()

    if opt.logfile.lower() != 'none':
        logging.basicConfig(filename=append_postfix(opt.logfile, opt.loss, opt.log_postfix), level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())

    return opt


def validate(net, val_data, ctx, use_threads=True):
    """Test a model."""
    outputs = []
    labels = []
    ctx_cpu = mx.cpu()

    for batch in tqdm(val_data, desc='Computing test embeddings'):
        data = mx.gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = mx.gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        neg_labels = mx.gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0, even_split=False)

        for x, l, nl in zip(data, label, neg_labels):
            embedding = net(x, l, nl)[0]
            outputs.append(embedding.as_in_context(ctx_cpu))
        labels += [x.as_in_context(ctx_cpu) for x in label]

    outputs = mx.nd.concatenate(outputs, axis=0)
    labels = mx.nd.concatenate(labels, axis=0)
    return evaluate(outputs, labels, val_data._dataset.num_classes(), use_threads=use_threads)


def train(net, opt, train_dataloader, val_dataloader, context, run_id):
    """Training function."""

    if not opt.skip_pretrain_validation:
        validation_results = validate(net, val_dataloader, context, use_threads=opt.num_workers > 0)
        for name, val_acc in validation_results:
            logging.info('Pre-train validation: %s=%f' % (name, val_acc))

    # Calculate decay steps
    steps = parse_steps(opt.steps, opt.epochs, logger=logging)

    # Init optimizer
    opt_options = {'learning_rate': opt.lr, 'wd': opt.wd, 'clip_gradient': 10.}
    if opt.optimizer == 'sgd':
        opt_options['momentum'] = 0.9
    elif opt.optimizer in ['adam', 'radam']:
        opt_options['epsilon'] = opt.epsilon
    elif opt.optimizer == 'rmsprop':
        opt_options['gamma1'] = 0.9
        opt_options['epsilon'] = opt.epsilon

    trainer = mx.gluon.Trainer(net.collect_params(), opt.optimizer, opt_options, kvstore=opt.kvstore)

    # Init loss function
    if opt.loss == 'nca':
        logging.info('Using NCA loss')
        proxyloss = ProxyNCALoss(train_dataloader._dataset.num_classes(), exclude_positives=True,
                                 label_smooth=opt.label_smooth,
                                 multiplier=opt.embedding_multiplier, temperature=opt.temperature)
    elif opt.loss == 'triplet':
        logging.info('Using triplet loss')
        proxyloss = ProxyTripletLoss(train_dataloader._dataset.num_classes())
    elif opt.loss == 'xentropy':
        logging.info('Using NCA loss without excluding positives')
        proxyloss = ProxyNCALoss(train_dataloader._dataset.num_classes(), exclude_positives=False,
                                 label_smooth=opt.label_smooth,
                                 multiplier=opt.embedding_multiplier, temperature=opt.temperature)
    else:
        raise RuntimeError('Unknown loss function: %s' % opt.loss)

    smoothing_constant = .01  # for tracking moving losses
    moving_loss = 0
    best_results = []  # R@1, NMI

    for epoch in range(1, opt.epochs + 1):
        p_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                     desc=('[Run %d/%d] Epoch %d' % (run_id, opt.number_of_runs, epoch)))

        new_lr = get_lr(opt.lr, epoch, steps, opt.factor)
        logging.info('Setting LR to %f' % new_lr)
        trainer.set_learning_rate(new_lr)
        if opt.optimizer == 'rmsprop':
            # exponential decay of gamma
            if epoch != 1:
                trainer._optimizer.gamma1 *= .94
                logging.info('Setting rmsprop gamma to %f' % trainer._optimizer.gamma1)

        for (i, batch) in p_bar:
            if opt.iteration_per_epoch > 0:
                for b in range(len(batch)):
                    batch[b] = batch[b][0]
            data = mx.gluon.utils.split_and_load(batch[0], ctx_list=context, batch_axis=0, even_split=False)
            label = mx.gluon.utils.split_and_load(batch[1], ctx_list=context, batch_axis=0, even_split=False)
            negative_labels = mx.gluon.utils.split_and_load(batch[2], ctx_list=context, batch_axis=0,
                                                            even_split=False)

            with ag.record():
                losses = []
                for x, y, nl in zip(data, label, negative_labels):
                    embs, positive_proxy, negative_proxies, proxies = net(x, y, nl)
                    if opt.loss in ['nca', 'xentropy']:
                        losses.append(proxyloss(embs, proxies, y, nl))
                    else:
                        losses.append(proxyloss(embs, positive_proxy, negative_proxies))
            for l in losses:
                l.backward()

            trainer.step(data[0].shape[0])

            #  Keep a moving average of the losses
            curr_loss = mx.nd.mean(mx.nd.maximum(mx.nd.concatenate(losses), 0)).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (epoch == 1))  # starting value
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
            p_bar.set_postfix_str('Moving loss: %.4f' % moving_loss)

        logging.info('Moving loss: %.4f' % moving_loss)
        validation_results = validate(net, val_dataloader, context, use_threads=opt.num_workers > 0)
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


def train_proxy(opt):
    logging.info(opt)

    # Set random seed
    mx.random.seed(opt.seed)
    np.random.seed(opt.seed)

    # Setup computation context
    context = get_context(opt.gpus, logging)

    run_results = []

    # Adjust batch size to each compute context
    batch_size = opt.batch_size * len(context)

    # Prepare feature extractor
    if opt.model == 'inception-bn':
        feature_net, feature_params = get_feature_model(opt.model, ctx=context)
        data_shape = 224
        scale_image_data = False
    elif opt.model == 'resnet50_v2':
        feature_net = mx.gluon.model_zoo.vision.resnet50_v2(pretrained=True, ctx=context).features
        data_shape = 224
        scale_image_data = True
    else:
        raise RuntimeError('Unsupported model: %s' % opt.model)

    # Prepare datasets
    train_dataset, val_dataset = get_dataset(opt.dataset, opt.data_path, data_shape=data_shape, use_crops=opt.use_crops,
                                             use_aug=True, with_proxy=True, scale_image_data=scale_image_data)
    logging.info(
        'Training with %d classes, validating with %d classes' % (
            train_dataset.num_classes(), val_dataset.num_classes()))

    if opt.iteration_per_epoch > 0:
        train_dataset, _ = get_dataset_iterator(opt.dataset, opt.data_path,
                                                batch_k=(opt.batch_size // 3) if opt.loss == 'xentropy' else opt.batch_k,
                                                batch_size=opt.batch_size,
                                                data_shape=data_shape, use_crops=opt.use_crops,
                                                scale_image_data=scale_image_data,
                                                batchify=False)
        train_dataloader = mx.gluon.data.DataLoader(DatasetIterator(train_dataset, opt.iteration_per_epoch,
                                                                    'next_proxy_sample',
                                                                    call_params={
                                                                        'sampled_classes': (opt.batch_size // opt.batch_k) if (opt.batch_k is not None) else None,
                                                                        'chose_classes_randomly': True,
                                                                    }),
                                                    batch_size=1, shuffle=False, num_workers=opt.num_workers,
                                                    last_batch='keep')
    else:
        train_dataloader = mx.gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                    num_workers=opt.num_workers, last_batch='rollover')
    val_dataloader = mx.gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=opt.num_workers, last_batch='keep')

    # Prepare proxy model
    net = ProxyNet(feature_net, opt.embed_dim, num_classes=train_dataset.num_classes())

    if opt.lr is None:
        logging.info('Using variable learning rate')
        opt.lr = max([opt.lr_proxynca, opt.lr_embedding, opt.lr_inception])

        for p, v in net.encoder.collect_params().items():
            v.lr_mult = opt.lr_embedding / opt.lr

        for p, v in net.base_net.collect_params().items():
            v.lr_mult = opt.lr_inception / opt.lr

        for p, v in net.proxies.collect_params().items():
            v.lr_mult = opt.lr_proxynca / opt.lr
    else:
        logging.info('Using single learning rate: %f' % opt.lr)

    for run in range(1, opt.number_of_runs + 1):
        logging.info('Starting run %d/%d' % (run, opt.number_of_runs))

        # reset networks
        if opt.model == 'inception-bn':
            net.base_net.collect_params().load(feature_params, ctx=context, ignore_extra=True)

            if opt.dataset == 'CUB':
                for v in net.base_net.collect_params().values():
                    if v.name in ['batchnorm', 'bn_']:
                        v.grad_req = 'null'

        elif opt.model == 'resnet50_v2':
            logging.info('Lowering LR for Resnet backbone')
            net.base_net = mx.gluon.model_zoo.vision.resnet50_v2(pretrained=True, ctx=context).features

            # Use a smaller learning rate for pre-trained convolutional layers.
            for v in net.base_net.collect_params().values():
                if 'conv' in v.name:
                    setattr(v, 'lr_mult', 0.01)
        else:
            raise NotImplementedError('Unknown model: %s' % opt.model)

        if opt.loss == 'triplet':
            net.encoder.initialize(mx.init.Xavier(magnitude=0.2), ctx=context, force_reinit=True)
            net.proxies.initialize(mx.init.Xavier(magnitude=0.2), ctx=context, force_reinit=True)
        else:
            net.init(TruncNorm(stdev=0.001), ctx=context, init_basenet=False)
        if not opt.disable_hybridize:
            net.hybridize()

        run_result = train(net, opt, train_dataloader, val_dataloader, context, run)
        run_results.append(run_result)
        logging.info('Run %d finished with %f' % (run, run_result[0][1]))

    logging.info(
        'Average validation of %d runs:\n%s' % (opt.number_of_runs, format_results(average_results(run_results))))


if __name__ == '__main__':
    train_proxy(parse_args())
