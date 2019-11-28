from __future__ import division

import logging

import mxnet as mx
import numpy as np
from mxnet import autograd as ag
from tqdm import tqdm

from common.evaluate import evaluate
from common.parser import TrainingParser
from common.utils import average_results, get_lr, get_context, parse_steps, format_results, append_postfix
from dataset import get_prototype_iterators, get_dataset
from dataset.dataloader import DatasetIterator
from models import get_feature_model, EmbeddingNet
from models.loss import PrototypeLoss


def parse_args():
    parser = TrainingParser(description='Distance metric learning using prototypical loss',
                            default_logfile='train_prototype.log',
                            default_model_prefix='prototype_model')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Number of samples in a batch per device. Default is 32')
    parser.add_argument('--nc', type=int, default=12,
                        help='Number of classes in each episode. Default is 12')
    parser.add_argument('--nq', type=int, default=5,
                        help='Number of query examples in each episode. Default is 5.')
    parser.add_argument('--ns', type=int, default=5,
                        help='Number of support examples in each episode. Default is 5.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs. default is 30.')
    parser.add_argument('--iteration-per-epoch', type=int, default=100,
                        help='Number of iterations per epoch. Default is 100')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate. default is 0.0001.')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='learning rate schedule factor. default is 0.5.')
    parser.add_argument('--wd', type=float, default=0.00001,
                        help='weight decay rate. default is 0.00001.')
    parser.add_argument('--steps', type=str, default='12,14,16,18',
                        help='epochs to update learning rate. default is 12,14,16,18.')

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


def train(net, opt, train_dataloader, val_dataloader, context, run_id):
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
    trainer = mx.gluon.Trainer(net.collect_params(), opt.optimizer,
                               opt_options,
                               kvstore=opt.kvstore)

    L = PrototypeLoss(opt.nc, opt.ns, opt.nq)

    data_size = opt.nc * (opt.ns + opt.nq)

    best_results = []  # R@1, NMI
    for epoch in range(1, opt.epochs + 1):

        prev_loss, cumulative_loss = 0.0, 0.0

        trainer.set_learning_rate(get_lr(opt.lr, epoch, steps, opt.factor))
        logging.info('Epoch %d learning rate=%f', epoch, trainer.learning_rate)

        p_bar = tqdm(train_dataloader, desc=('[Run %d/%d] Epoch %d' % (run_id, opt.number_of_runs, epoch)))
        for batch in p_bar:
            supports_batch, queries_batch, labels_batch = [x[0] for x in batch]
            # supports_batch: <Nc x Ns x I>
            # queries_batch: <Nc x Nq x I>
            # labels_batch: <Nc x 1>

            supports_batch = mx.nd.reshape(supports_batch, (-1, 0, 0, 0), reverse=True)  # <(Nc * Ns) x I>
            queries_batch = mx.nd.reshape(queries_batch, (-1, 0, 0, 0), reverse=True)

            queries = mx.gluon.utils.split_and_load(queries_batch, ctx_list=context, batch_axis=0)
            supports = mx.gluon.utils.split_and_load(supports_batch, ctx_list=context, batch_axis=0)

            support_embs = []
            queries_embs = []
            with ag.record():
                for s in supports:
                    s_emb = net(s)
                    support_embs.append(s_emb)
                supports = mx.nd.concat(*support_embs, dim=0)  # <Nc*Ns x E>

                for q in queries:
                    q_emb = net(q)
                    queries_embs.append(q_emb)
                queries = mx.nd.concat(*queries_embs, dim=0)  # <Nc*Nq x E>

                loss = L(supports, queries)

            loss.backward()
            cumulative_loss += mx.nd.mean(loss).asscalar()
            trainer.step(data_size)

            p_bar.set_postfix({'loss': cumulative_loss - prev_loss})
            prev_loss = cumulative_loss

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


def train_prototype(opt):
    logging.info(opt)

    # Set random seed
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
    else:
        raise RuntimeError('Unsupported model: %s' % opt.model)

    net = EmbeddingNet(feature_net, opt.embed_dim, normalize=False)

    if opt.model == 'resnet50_v2':
        # Use a smaller learning rate for pre-trained convolutional layers.
        logging.info('Using smaller conv learning rates')
        for v in net.base_net.collect_params().values():
            if 'conv' in v.name:
                setattr(v, 'lr_mult', 0.01)

    # Get data iterators
    train_dataset, val_dataset = get_dataset(opt.dataset, opt.data_path, data_shape=data_shape, use_crops=opt.use_crops,
                                             use_aug=True, scale_image_data=scale_image_data)
    train_dataiter, _ = get_prototype_iterators(opt.dataset, opt.data_path, Nc=opt.nc, Ns=opt.ns, Nq=opt.nq,
                                                data_shape=data_shape,
                                                test_batch_size=len(context) * opt.batch_size, use_crops=opt.use_crops,
                                                scale_image_data=scale_image_data)
    train_dataloader = mx.gluon.data.DataLoader(DatasetIterator(train_dataiter, opt.iteration_per_epoch, 'next'),
                                                batch_size=1, shuffle=False, num_workers=opt.num_workers,
                                                last_batch='keep')
    val_dataloader = mx.gluon.data.DataLoader(val_dataset, batch_size=len(context) * opt.batch_size, shuffle=False,
                                              num_workers=opt.num_workers, last_batch='keep')
    logging.info('Training with %d classes, validating with %d classes' % (
        train_dataset.num_classes(), val_dataset.num_classes()))

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

        run_result = train(net, opt, train_dataloader, val_dataloader, context, run)
        run_results.append(run_result)
        logging.info('Run %d finished with %f' % (run, run_result[0][1]))

    logging.info(
        'Average validation of %d runs:\n%s' % (opt.number_of_runs, format_results(average_results(run_results))))


if __name__ == '__main__':
    train_prototype(parse_args())

