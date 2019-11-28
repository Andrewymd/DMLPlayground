from __future__ import division

import logging
from math import ceil

import mxnet as mx
import numpy as np
from mxnet import autograd as ag
from tqdm import tqdm

from common.evaluate import evaluate
from common.parser import TrainingParser
from common.utils import average_results, get_context, parse_steps, get_lr, format_results, append_postfix
from dataset import get_dataset, get_dataset_iterator
from models import get_feature_model
from models.loss import ProxyXentropyLoss
from models.proxynca import NormProxyNet
from models.simplemodels import TruncNorm


def parse_args():
    parser = TrainingParser(description='Distance metric learning using normproxies',
                            default_logfile='train_normproxy.log', default_model_prefix='normproxy_model')
    parser.add_argument('--batch-size', type=int, default=75,
                        help='Number of samples in a batch. Default is 75')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs. default is 60.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate. default is 0.0001.')
    parser.add_argument('--factor', type=float, default=0.1,
                        help='learning rate schedule factor. default is 0.5.')
    parser.add_argument('--wd', type=float, default=0.00001,
                        help='weight decay rate. default is 0.00001.')
    parser.add_argument('--steps', type=str, default='15',
                        help='Epochs to update learning rate. Negative number represents periodic decrease.'
                             'Zero means no steps. Default is -1')
    parser.add_argument('--binarize', action="store_true",
                        help='Thresholds the embedding into a binary vector')
    parser.add_argument('--epsilon', type=float, default=1e-2,
                        help='Optimizer epsilon. default is 0.01.')
    parser.add_argument('--label-smooth', type=float, default=0.1,
                        help='Label smoothing. Default is 0')
    parser.add_argument('--temperature', type=float, default=0.05,
                        help='Sigma temperature constant. Default is 0.05')
    parser.add_argument('--batch-k', type=int, default=0,
                        help='Number of images per class for episodic sampling. 0 Will turn it off.')
    parser.add_argument('--start-epoch', type=int, default=1,
                        help='Epoch to start at, >1 means loading parameters')
    parser.add_argument('--no-fc', action="store_true", help='Skips the fully-connected layer in the model.')
    parser.add_argument('--static-proxies', action="store_true", help='Proxies will not be learned.')
    parser.add_argument('--no-dropout', dest='dropout', action="store_false", help='Do not add dropout layer to the model.')
    parser.add_argument('--similarity', type=str, choices=['euclidean', 'cosine'], default='euclidean')
    parser.set_defaults(
        embed_dim=2048,
        lr=0.001,
        wd=1e-4,
    )
    opt = parser.parse_args()

    if opt.logfile.lower() != 'none':
        logging.basicConfig(filename=append_postfix(opt.logfile, opt.log_postfix), level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())

    return opt


def validate(net, val_data, ctx, binarize=True, nmi=True, similarity='euclidean'):
    """Test a model."""
    outputs = []
    labels = []
    ctx_cpu = mx.cpu()

    for batch in tqdm(val_data, desc='Computing test embeddings'):
        data = mx.gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = mx.gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

        for x in data:
            embs = net(x)[0]
            if binarize:
                embs = embs > 0
            outputs.append(embs.as_in_context(ctx_cpu))
        labels += [x.as_in_context(ctx_cpu) for x in label]

    outputs = mx.nd.concatenate(outputs, axis=0)
    labels = mx.nd.concatenate(labels, axis=0)
    return evaluate(outputs, labels, val_data._dataset.num_classes(), similarity=similarity,
                    get_detailed_metrics=False, nmi=nmi)


def train(net, opt, train_data, val_data, num_train_classes, context, run_id):
    """Training function"""

    if not opt.skip_pretrain_validation:
        validation_results = validate(net, val_data, context, binarize=opt.binarize, nmi=opt.nmi, similarity=opt.similarity)
        for name, val_acc in validation_results:
            logging.info('Pre-train validation: %s=%f' % (name, val_acc))

    # Calculate decay steps
    steps = parse_steps(opt.steps, opt.epochs, logger=logging)

    # Init optimizer
    opt_options = {'learning_rate': opt.lr, 'wd': opt.wd, 'clip_gradient': 10.}
    if opt.optimizer == 'sgd':
        opt_options['momentum'] = 0.9
    elif opt.optimizer == 'adam':
        opt_options['epsilon'] = opt.epsilon
    elif opt.optimizer == 'rmsprop':
        opt_options['gamma1'] = 0.9
        opt_options['epsilon'] = opt.epsilon

    # We train only embedding and proxies initially
    params2train = net.encoder.collect_params()
    if not opt.static_proxies:
        params2train.update(net.proxies.collect_params())

    trainer = mx.gluon.Trainer(params2train, opt.optimizer, opt_options, kvstore=opt.kvstore)

    smoothing_constant = .01  # for tracking moving losses
    moving_loss = 0
    best_results = []  # R@1, NMI

    batch_size = opt.batch_size * len(context)

    proxyloss = ProxyXentropyLoss(num_train_classes, label_smooth=opt.label_smooth, temperature=opt.temperature)

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        if epoch == 2:
            # switch training to all parameters
            logging.info('Switching to train all parameters')
            trainer = mx.gluon.Trainer(net.collect_params(), opt.optimizer, opt_options, kvstore=opt.kvstore)
        if opt.batch_k > 0:
            iterations_per_epoch = int(ceil(train_data.num_training_images() / batch_size))
            p_bar = tqdm(range(iterations_per_epoch), desc='[Run %d/%d] Epoch %d' % (run_id, opt.number_of_runs, epoch),
                         total=iterations_per_epoch)
        else:
            p_bar = tqdm(enumerate(train_data), total=len(train_data),
                         desc=('[Run %d/%d] Epoch %d' % (run_id, opt.number_of_runs, epoch)))

        new_lr = get_lr(opt.lr, epoch, steps, opt.factor)
        logging.info('Setting LR to %f' % new_lr)
        trainer.set_learning_rate(new_lr)
        if opt.optimizer == 'rmsprop':
            # exponential decay of gamma
            if epoch != 1:
                trainer._optimizer.gamma1 *= .94
                logging.info('Setting rmsprop gamma to %f' % trainer._optimizer.gamma1)

        losses = []
        curr_losses_np = []

        for i in p_bar:
            if opt.batch_k > 0:
                num_sampled_classes = batch_size // opt.batch_k
                batch = train_data.next_proxy_sample(sampled_classes=num_sampled_classes, chose_classes_randomly=True).data
            else:
                batch = i[1]
                i = i[0]

            data = mx.gluon.utils.split_and_load(batch[0], ctx_list=context, batch_axis=0, even_split=False)
            label = mx.gluon.utils.split_and_load(batch[1], ctx_list=context, batch_axis=0, even_split=False)

            with ag.record():
                for x, y in zip(data, label):
                    embs, proxies = net(x)
                    curr_loss = proxyloss(embs, proxies, y)
                    losses.append(curr_loss)
                mx.nd.waitall()

            curr_losses_np += [cl.asnumpy() for cl in losses]

            ag.backward(losses)

            trainer.step(batch[0].shape[0])

            #  Keep a moving average of the losses
            curr_loss = np.mean(np.maximum(np.concatenate(curr_losses_np), 0))
            curr_losses_np.clear()

            losses.clear()
            moving_loss = (curr_loss if ((i == 0) and (epoch == 1))  # starting value
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
            p_bar.set_postfix_str('Moving loss: %.4f' % moving_loss)

        logging.info('Moving loss: %.4f' % moving_loss)
        validation_results = validate(net, val_data, context, binarize=opt.binarize, nmi=opt.nmi, similarity=opt.similarity)
        for name, val_acc in validation_results:
            logging.info('[Epoch %d] validation: %s=%f' % (epoch, name, val_acc))

        if (len(best_results) == 0) or (validation_results[0][1] > best_results[0][1]):
            best_results = validation_results
            filename = '%s.params' % opt.save_model_prefix
            logging.info('Saving %s.' % filename)
            net.save_parameters(filename)
            logging.info('New best validation: R@1: %f%s' % (best_results[0][1], (' NMI: %f' % best_results[-1][1]) if opt.nmi else ''))

    return best_results


def train_normproxy(opt):
    logging.info(opt)

    # Set random seed
    mx.random.seed(opt.seed)
    np.random.seed(opt.seed)

    # Setup computation context
    context = get_context(opt.gpus, logging)

    # Adjust batch size to each compute context
    batch_size = opt.batch_size * len(context)

    run_results = []

    # Prepare feature extractor
    if opt.model == 'inception-bn':
        feature_net, feature_params = get_feature_model(opt.model, ctx=context)
        feature_net.collect_params().load(feature_params, ctx=context, ignore_extra=True)
        data_shape = 224
        scale_image_data = False
        feature_size = 1024
    elif opt.model == 'resnet50_v2':
        feature_params = None
        feature_net = mx.gluon.model_zoo.vision.resnet50_v2(pretrained=True, ctx=context).features
        data_shape = 224
        scale_image_data = True
        feature_size = 2048
    else:
        raise RuntimeError('Unsupported model: %s' % opt.model)

    # Prepare datasets
    train_dataset, val_dataset = get_dataset(opt.dataset, opt.data_path, data_shape=data_shape, use_crops=opt.use_crops,
                                             use_aug=True, with_proxy=True, scale_image_data=scale_image_data)
    logging.info(
        'Training with %d classes, validating with %d classes' % (
            train_dataset.num_classes(), val_dataset.num_classes()))

    if opt.batch_k > 0:
        train_dataset, _ = get_dataset_iterator(opt.dataset, opt.data_path, batch_k=opt.batch_k,
                                                batch_size=batch_size, data_shape=data_shape, use_crops=opt.use_crops,
                                                scale_image_data=scale_image_data)
        train_dataloader = train_dataset
    else:
        train_dataloader = mx.gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                    num_workers=opt.num_workers, last_batch='rollover')
    val_dataloader = mx.gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=opt.num_workers, last_batch='keep')
    num_train_classes = train_dataset.num_classes()

    # Prepare proxy model
    net = NormProxyNet(feature_net, opt.embed_dim, num_classes=train_dataset.num_classes(),
                       feature_size=feature_size, no_fc=opt.no_fc, dropout=opt.dropout, static_proxies=opt.static_proxies)

    # main run loop for multiple training runs
    for run in range(1, opt.number_of_runs + 1):
        logging.info('Starting run %d/%d' % (run, opt.number_of_runs))

        # reset networks
        if opt.model == 'inception-bn':
            net.base_net.collect_params().load(feature_params, ctx=context, ignore_extra=True)
        elif opt.model == 'resnet50_v2':
            net.base_net = mx.gluon.model_zoo.vision.resnet50_v2(pretrained=True, ctx=context).features

            # Use a smaller learning rate for pre-trained convolutional layers.
            logging.info('Lowering LR for Resnet backbone by 100x')
            for v in net.base_net.collect_params().values():
                if 'conv' in v.name:
                    setattr(v, 'lr_mult', 0.01)
        else:
            raise NotImplementedError('Unknown model: %s' % opt.model)

        if opt.start_epoch != 1:
            param_file = 'normproxy_model.params'
            logging.info('Loading parameters from %s' % param_file)
            net.load_parameters(param_file, ctx=context)
        else:
            if opt.model == 'resnet50_v2':
                net.init(mx.init.Xavier(magnitude=2), ctx=context, init_basenet=False)
            else:
                net.init(TruncNorm(stdev=0.001), ctx=context, init_basenet=False)
        if not opt.disable_hybridize:
            net.hybridize()

        run_result = train(net, opt, train_dataloader, val_dataloader, num_train_classes, context, run)
        run_results.append(run_result)
        logging.info('Run %d finished with %f' % (run, run_result[0][1]))

    logging.info(
        'Average validation of %d runs:\n%s' % (opt.number_of_runs, format_results(average_results(run_results))))


if __name__ == '__main__':
    train_normproxy(parse_args())
