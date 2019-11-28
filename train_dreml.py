import logging

import mxnet as mx
import numpy as np
from mxnet import autograd as ag
from tqdm import tqdm

from common.evaluate import evaluate
from common.parser import TrainingParser
from common.utils import get_context, parse_steps, get_lr, append_postfix
from dataset import get_dataset
from models import get_feature_model, ProxyNet
from models.loss import ProxyNCALoss, ProxyTripletLoss, StaticProxyLoss
from models.simplemodels import TruncNorm, EmbeddingNet


def parse_args():
    parser = TrainingParser(description='Deep Randomized Ensembles for Metric Learning',
                            default_logfile='train_dreml.log',
                            default_model_prefix='dreml_model')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Number of samples in a batch. Default is 128')
    parser.add_argument('--epochs', type=int, default=12,
                        help='number of training epochs. default is 20.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate. default is 0.0001.')
    parser.add_argument('--factor', type=float, default=0.1,
                        help='learning rate schedule factor. default is 0.1.')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='weight decay rate. default is 5e-4.')
    parser.add_argument('--steps', type=str, default='-4',
                        help='Epochs to update learning rate. Negative number represents periodic decrease.'
                             'Zero means no steps. Default is -4')
    parser.add_argument('--loss', type=str, default='nca',
                        help='Which loss to use: [triplet, nca, xentropy]')
    parser.add_argument('--epsilon', type=float, default=1e-2,
                        help='Optimizer epsilon. default is 0.01.')
    parser.add_argument('--label-smooth', type=float, default=0,
                        help='Label smoothing. Default is 0')
    parser.add_argument('--embedding-multiplier', type=float, default=3,
                        help='Multiplies normalized embeddings and proxies. Default is 3')
    parser.add_argument('-L', '--number-of-ensembles', dest='L', type=int, default=48,
                        help='Number of ensembles.')
    parser.add_argument('-D', '--meta-classes', dest='D', type=int, default=12,
                        help='Number of meta-classes.')
    parser.add_argument('--static-proxies', action="store_true",
                        help='Do not learn proxies, but keep them fixed.')
    parser.add_argument('--data-shape', type=int, default=224,
                        help='Input data size')

    opt = parser.parse_args()

    if opt.logfile.lower() != 'none':
        logging.basicConfig(filename=append_postfix(opt.logfile, opt.log_postfix), level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())

    return opt


def validate(dataloader, models, context, static_proxies, similarity='cosine'):
    outputs = []
    labels = []
    ctx_cpu = mx.cpu()

    for batch in tqdm(dataloader, desc='Computing test embeddings'):
        data = mx.gluon.utils.split_and_load(batch[0], ctx_list=context, batch_axis=0, even_split=False)
        label = mx.gluon.utils.split_and_load(batch[1], ctx_list=context, batch_axis=0, even_split=False)
        neg_labels = mx.gluon.utils.split_and_load(batch[2], ctx_list=context, batch_axis=0, even_split=False)
        for x, l, nl in zip(data, label, neg_labels):
            ensembles = []
            for m in models:
                m.collect_params().reset_ctx(context)  # move model to GPU
                if static_proxies:
                    ensembles.append(m(x).as_in_context(ctx_cpu))
                else:
                    ensembles.append(
                        m(x, mx.nd.zeros_like(l, ctx=x.context), mx.nd.zeros_like(nl, ctx=x.context))[0].as_in_context(
                            ctx_cpu))
                mx.nd.waitall()
                m.collect_params().reset_ctx(ctx_cpu)  # move model to CPU
            outputs.append(mx.nd.concat(*ensembles, dim=1))
        labels += [x.as_in_context(ctx_cpu) for x in label]

    outputs = mx.nd.concatenate(outputs, axis=0)
    labels = mx.nd.concatenate(labels, axis=0)
    logging.info('Evaluating with %s distance' % similarity)
    return evaluate(outputs, labels, dataloader._dataset.num_classes(), similarity=similarity)


def train_dreml(opt):
    logging.info(opt)

    # Set random seed
    mx.random.seed(opt.seed)
    np.random.seed(opt.seed)

    # Setup computation context
    context = get_context(opt.gpus, logging)
    cpu_ctx = mx.cpu()

    # Adjust batch size to each compute context
    batch_size = opt.batch_size * len(context)

    if opt.model == 'inception-bn':
        scale_image_data = False
    elif opt.model in ['resnet50_v2', 'resnet18_v2']:
        scale_image_data = True
    else:
        raise RuntimeError('Unsupported model: %s' % opt.model)

    # Prepare datasets
    train_dataset, val_dataset = get_dataset(opt.dataset, opt.data_path, data_shape=opt.data_shape, use_crops=opt.use_crops,
                                             use_aug=True, with_proxy=True, scale_image_data=scale_image_data,
                                             resize_img=int(opt.data_shape * 1.1))

    # Create class mapping
    mapping = np.random.randint(0, opt.D, (opt.L, train_dataset.num_classes()))

    # Train embedding functions one by one
    trained_models = []
    best_results = []  # R@1, NMI
    for ens in tqdm(range(opt.L), desc='Training model in ensemble'):
        train_dataset.set_class_mapping(mapping[ens], opt.D)
        train_dataloader = mx.gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                    num_workers=opt.num_workers, last_batch='rollover')

        if opt.model == 'inception-bn':
            feature_net, feature_params = get_feature_model(opt.model, ctx=context)
        elif opt.model == 'resnet50_v2':
            feature_net = mx.gluon.model_zoo.vision.resnet50_v2(pretrained=True, ctx=context).features
        elif opt.model == 'resnet18_v2':
            feature_net = mx.gluon.model_zoo.vision.resnet18_v2(pretrained=True, ctx=context).features
        else:
            raise RuntimeError('Unsupported model: %s' % opt.model)

        if opt.static_proxies:
            net = EmbeddingNet(feature_net, opt.D, normalize=False)
        else:
            net = ProxyNet(feature_net, opt.D, num_classes=opt.D)

        # Init loss function
        if opt.static_proxies:
            logging.info('Using static proxies')
            proxyloss = StaticProxyLoss(opt.D)
        elif opt.loss == 'nca':
            logging.info('Using NCA loss')
            proxyloss = ProxyNCALoss(opt.D, exclude_positives=True, label_smooth=opt.label_smooth,
                                     multiplier=opt.embedding_multiplier)
        elif opt.loss == 'triplet':
            logging.info('Using triplet loss')
            proxyloss = ProxyTripletLoss(opt.D)
        elif opt.loss == 'xentropy':
            logging.info('Using NCA loss without excluding positives')
            proxyloss = ProxyNCALoss(opt.D, exclude_positives=False, label_smooth=opt.label_smooth,
                                     multiplier=opt.embedding_multiplier)
        else:
            raise RuntimeError('Unknown loss function: %s' % opt.loss)

        # Init optimizer
        opt_options = {'learning_rate': opt.lr, 'wd': opt.wd}
        if opt.optimizer == 'sgd':
            opt_options['momentum'] = 0.9
        elif opt.optimizer == 'adam':
            opt_options['epsilon'] = opt.epsilon
        elif opt.optimizer == 'rmsprop':
            opt_options['gamma1'] = 0.9
            opt_options['epsilon'] = opt.epsilon

        # Calculate decay steps
        steps = parse_steps(opt.steps, opt.epochs, logger=logging)

        # reset networks
        if opt.model == 'inception-bn':
            net.base_net.collect_params().load(feature_params, ctx=context, ignore_extra=True)
        elif opt.model in ['resnet18_v2', 'resnet50_v2']:
            net.base_net = mx.gluon.model_zoo.vision.get_model(opt.model, pretrained=True, ctx=context).features
        else:
            raise NotImplementedError('Unknown model: %s' % opt.model)

        if opt.static_proxies:
            net.init(mx.init.Xavier(magnitude=0.2), ctx=context, init_basenet=False)
        elif opt.loss == 'triplet':
            net.encoder.initialize(mx.init.Xavier(magnitude=0.2), ctx=context, force_reinit=True)
            net.proxies.initialize(mx.init.Xavier(magnitude=0.2), ctx=context, force_reinit=True)
        else:
            net.init(TruncNorm(stdev=0.001), ctx=context, init_basenet=False)
        if not opt.disable_hybridize:
            net.hybridize()

        trainer = mx.gluon.Trainer(net.collect_params(), opt.optimizer,
                                   opt_options,
                                   kvstore=opt.kvstore)

        smoothing_constant = .01  # for tracking moving losses
        moving_loss = 0

        for epoch in range(1, opt.epochs + 1):
            p_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                         desc=('[Model %d/%d] Epoch %d' % (ens + 1, opt.L, epoch)))

            new_lr = get_lr(opt.lr, epoch, steps, opt.factor)
            logging.info('Setting LR to %f' % new_lr)
            trainer.set_learning_rate(new_lr)

            for i, batch in p_bar:
                data = mx.gluon.utils.split_and_load(batch[0], ctx_list=context, batch_axis=0, even_split=False)
                label = mx.gluon.utils.split_and_load(batch[1], ctx_list=context, batch_axis=0, even_split=False)
                negative_labels = mx.gluon.utils.split_and_load(batch[2], ctx_list=context, batch_axis=0,
                                                                even_split=False)

                with ag.record():
                    losses = []
                    for x, y, nl in zip(data, label, negative_labels):
                        if opt.static_proxies:
                            embs = net(x)
                            losses.append(proxyloss(embs, y))
                        else:
                            embs, positive_proxy, negative_proxies, proxies = net(x, y, nl)
                            if opt.loss in ['nca', 'xentropy']:
                                losses.append(proxyloss(embs, proxies, y, nl))
                            else:
                                losses.append(proxyloss(embs, positive_proxy, negative_proxies))
                for l in losses:
                    l.backward()

                trainer.step(data[0].shape[0])

                ##########################
                #  Keep a moving average of the losses
                ##########################
                curr_loss = mx.nd.mean(mx.nd.maximum(mx.nd.concatenate(losses), 0)).asscalar()
                moving_loss = (curr_loss if ((i == 0) and (epoch == 1))  # starting value
                               else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
                p_bar.set_postfix_str('Moving loss: %.4f' % moving_loss)

            logging.info('Moving loss: %.4f' % moving_loss)

        # move model to CPU
        mx.nd.waitall()
        net.collect_params().reset_ctx(cpu_ctx)
        trained_models.append(net)
        del train_dataloader

        # Run ensemble validation
        logging.info('Running validation with %d models in the ensemble' % len(trained_models))
        val_dataloader = mx.gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=opt.num_workers, last_batch='keep')

        validation_results = validate(val_dataloader, trained_models, context, opt.static_proxies)

        for name, val_acc in validation_results:
            logging.info('Validation: %s=%f' % (name, val_acc))

        if (len(best_results) == 0) or (validation_results[0][1] > best_results[0][1]):
            best_results = validation_results
            logging.info('New best validation: R@1: %f NMI: %f' % (best_results[0][1], best_results[-1][1]))


if __name__ == '__main__':
    train_dreml(parse_args())
