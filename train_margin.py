from __future__ import division

import logging

import mxnet as mx
import numpy as np
from mxnet import autograd as ag, nd
from mxnet import gluon
from tqdm import tqdm

from common.evaluate import evaluate
from common.parser import TrainingParser
from common.utils import average_results, format_results, get_context, parse_steps, get_lr, append_postfix
from dataset import get_dataset_iterator, get_dataset
from dataset.dataloader import DatasetIterator
from models import get_feature_model
from models.marginmodels import MarginNet, MarginLoss


def parse_args():
    parser = TrainingParser(description='Distance metric learning with marginloss and distance-weighted sampling.',
                            default_logfile='train_margin.log',  default_model_prefix='margin_loss_model')
    parser.add_argument('--batch-size', type=int, default=125,
                        help='Number of samples in a batch per compute unit. Default is 125.'
                             'Must be divisible with batch-k.')
    parser.add_argument('--batch-k', type=int, default=5,
                        help='number of images per class in a batch. default is 5.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs. default is 20.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate. default is 0.0001.')
    parser.add_argument('--lr-beta', type=float, default=0.1,
                        help='learning rate for the beta in margin based loss. default is 0.1.')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='margin for the margin based loss. default is 0.2.')
    parser.add_argument('--beta', type=float, default=1.2,
                        help='initial value for beta. default is 1.2.')
    parser.add_argument('--nu', type=float, default=0.0,
                        help='regularization parameter for beta. default is 0.0.')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='learning rate schedule factor. default is 0.5.')
    parser.add_argument('--steps', type=str, default='12,14,16,18',
                        help='epochs to update learning rate. default is 12,14,16,18.')
    parser.add_argument('--wd', type=float, default=0.00001,
                        help='weight decay rate. default is 0.00001.')
    parser.add_argument('--iteration-per-epoch', type=int, default=200,
                        help='Number of iteration per epoch. default=200.')

    opt = parser.parse_args()

    if opt.logfile.lower() != 'none':
        logging.basicConfig(filename=append_postfix(opt.logfile, opt.log_postfix), level=logging.INFO)
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

        for x in data:
            outputs.append(net(x).as_in_context(ctx_cpu))
        labels += [l.as_in_context(ctx_cpu) for l in label]

    outputs = mx.nd.concatenate(outputs, axis=0)
    labels = mx.nd.concatenate(labels, axis=0)
    return evaluate(outputs, labels, val_data._dataset.num_classes(), use_threads=use_threads)


def train(net, beta, opt, train_dataloader, val_dataloader, batch_size, context, run_id):
    """Training function."""

    if not opt.skip_pretrain_validation:
        validation_results = validate(net, val_dataloader, context, use_threads=opt.num_workers > 0)
        for name, val_acc in validation_results:
            logging.info('Pre-train validation: %s=%f' % (name, val_acc))

    steps = parse_steps(opt.steps, opt.epochs, logging)

    opt_options = {'learning_rate': opt.lr, 'wd': opt.wd}
    if opt.optimizer == 'sgd':
        opt_options['momentum'] = 0.9
    if opt.optimizer == 'adam':
        opt_options['epsilon'] = 1e-7

    trainer = gluon.Trainer(net.collect_params(), opt.optimizer, opt_options, kvstore=opt.kvstore)

    train_beta = not isinstance(beta, float)

    if train_beta:
        # Jointly train class-specific beta
        beta.initialize(mx.init.Constant(opt.beta), ctx=context)
        trainer_beta = gluon.Trainer(beta.collect_params(), 'sgd',
                                     {'learning_rate': opt.lr_beta, 'momentum': 0.9}, kvstore=opt.kvstore)
    loss = MarginLoss(batch_size, opt.batch_k, beta, margin=opt.margin, nu=opt.nu, train_beta=train_beta)
    if not opt.disable_hybridize:
        loss.hybridize()

    best_results = []  # R@1, NMI

    for epoch in range(1, opt.epochs + 1):
        prev_loss, cumulative_loss = 0.0, 0.0

        # Learning rate schedule.
        trainer.set_learning_rate(get_lr(opt.lr, epoch, steps, opt.factor))
        logging.info('Epoch %d learning rate=%f', epoch, trainer.learning_rate)
        if train_beta:
            trainer_beta.set_learning_rate(get_lr(opt.lr_beta, epoch, steps, opt.factor))
            logging.info('Epoch %d beta learning rate=%f', epoch, trainer_beta.learning_rate)

        p_bar = tqdm(train_dataloader, desc='[Run %d/%d] Epoch %d' % (run_id, opt.number_of_runs, epoch),
                     total=opt.iteration_per_epoch)
        for batch in p_bar:
            data = gluon.utils.split_and_load(batch[0][0], ctx_list=context, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1][0].astype('float32'), ctx_list=context, batch_axis=0)

            Ls = []
            with ag.record():
                for x, y in zip(data, label):
                    embedings = net(x)
                    L = loss(embedings, y)

                    Ls.append(L)
                    cumulative_loss += nd.mean(L).asscalar()

                for L in Ls:
                    L.backward()

            trainer.step(batch[0].shape[1])
            if opt.lr_beta > 0.0:
                trainer_beta.step(batch[0].shape[1])

            p_bar.set_postfix({'loss': cumulative_loss - prev_loss})
            prev_loss = cumulative_loss

        logging.info('[Epoch %d] training loss=%f' % (epoch, cumulative_loss))

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


def train_margin(opt):
    logging.info(opt)

    # Set random seed
    mx.random.seed(opt.seed)
    np.random.seed(opt.seed)

    # Setup computation context
    context = get_context(opt.gpus, logging)

    # Adjust batch size to each compute context
    batch_size = opt.batch_size * len(context)

    run_results = []

    # Get model
    if opt.model == 'inception-bn':
        feature_net, feature_params = get_feature_model(opt.model, ctx=context)
        feature_net.collect_params().load(feature_params, ctx=context, ignore_extra=True)
        data_shape = 224
        scale_image_data = False
    elif opt.model == 'resnet50_v2':
        feature_params = None
        feature_net = mx.gluon.model_zoo.vision.resnet50_v2(pretrained=True, ctx=context).features
        data_shape = 224
        scale_image_data = True
    else:
        raise RuntimeError('Unsupported model: %s' % opt.model)

    net = MarginNet(feature_net, opt.embed_dim)

    if opt.model == 'resnet50_v2':
        # Use a smaller learning rate for pre-trained convolutional layers.
        for v in net.base_net.collect_params().values():
            if 'conv' in v.name:
                setattr(v, 'lr_mult', 0.01)

    # Get data iterators
    train_dataset, val_dataset = get_dataset(opt.dataset, opt.data_path, data_shape=data_shape, use_crops=opt.use_crops,
                                             use_aug=True, scale_image_data=scale_image_data)
    train_dataiter, _ = get_dataset_iterator(opt.dataset, opt.data_path, batch_k=opt.batch_k, batch_size=batch_size,
                                             data_shape=data_shape, use_crops=opt.use_crops,
                                             scale_image_data=scale_image_data, batchify=False)
    train_dataloader = mx.gluon.data.DataLoader(DatasetIterator(train_dataiter, opt.iteration_per_epoch, 'next'),
                                                batch_size=1, shuffle=False, num_workers=opt.num_workers,
                                                last_batch='keep')
    val_dataloader = mx.gluon.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False,
                                              num_workers=opt.num_workers, last_batch='keep')

    logging.info('Training with %d classes, validating with %d classes' % (
            train_dataset.num_classes(), val_dataset.num_classes()))

    # main run loop for multiple training runs
    for run in range(1, opt.number_of_runs + 1):
        logging.info('Starting run %d/%d' % (run, opt.number_of_runs))

        # Re-init embedding layers and reload pretrained layers
        if opt.model == 'inception-bn':
            net.init(mx.init.Xavier(magnitude=0.2), ctx=context, init_basenet=False)
            net.base_net.collect_params().load(feature_params, ctx=context, ignore_extra=True)
        elif opt.model == 'resnet50_v2':
            net.init(mx.init.Xavier(magnitude=2), ctx=context, init_basenet=False)
            net.base_net = mx.gluon.model_zoo.vision.resnet50_v2(pretrained=True, ctx=context).features
        else:
            raise RuntimeError('Unknown model type: %s' % opt.model)

        if not opt.disable_hybridize:
            net.hybridize()

        if opt.lr_beta > 0.0:
            logging.info('Learning beta margin')
            beta = mx.gluon.nn.Embedding(train_dataset.num_classes(), 1)
        else:
            beta = opt.beta

        run_result = train(net, beta, opt, train_dataloader, val_dataloader, batch_size, context, run)
        run_results.append(run_result)
        logging.info('Run %d finished with %f' % (run, run_result[0][1]))

    logging.info(
        'Average validation of %d runs:\n%s' % (opt.number_of_runs, format_results(average_results(run_results))))


if __name__ == '__main__':
    train_margin(parse_args())
