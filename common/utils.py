import mxnet as mx
import numpy as np


def isjpeg(filename):
    """
    Returns true if filename has jpeg file extension
    :param filename:
    :return:
    """
    jpeg_exts = ['jpg', 'jpeg']
    if '.' not in filename:
        return False
    return filename.split('.')[-1].lower() in jpeg_exts


def format_results(results):
    if not isinstance(results, np.ndarray):
        return ''
    labels = ['R@1', 'R@2', 'R@4', 'R@8', 'R@1', 'NMI']
    output = ' ' + ' \t '.join(labels[:len(results[:len(labels)])]) + '\n'
    output += '\t'.join(['%.2f' % r for r in results[:len(labels)]])
    return output


def average_results(results):
    """
    Computes the average of results
    :param results: list of lists of 2-tuples (name-value)
    :return:
    """
    if not isinstance(results, list):
        return None
    if len(results) > 0:
        data = np.zeros((len(results), len(results[0])))
        for i, res in enumerate(results):
            for j, (name, value) in enumerate(res):
                data[i][j] = value
        avg = np.mean(data, axis=0)
        return avg
    return None


def get_context(gpus, logger=None):
    """
    Returns the mxnet compute context based on the passed configuration
    :param gpus: integer, integer list or string of comma separated device-ids. e.g. 0,1,2.
                 -1 means to use all available gpus
                 -2 use only cpu
    :param logger: logger object to write results to. Can be None
    :return: list of mxnet context objects
    """
    if isinstance(gpus, int):
        gpus = [gpus]
    elif (not isinstance(gpus, list)) and (not isinstance(gpus, str)):
        raise Exception('Error: gpus should be an integer list or string of comma separated device-ids. e.g. 0,1,2')

    if isinstance(gpus, str):
        gpus = [int(g.strip()) for g in gpus.split(',')]

    if gpus[0] == -1:  # choose all available gpus
        gpus = mx.context.num_gpus()
        if logger:
            logger.info('Autodetected %s GPUs%s' % (gpus, ' using CPU' if gpus == 0 else ''))
        if gpus == 0:
            context = [mx.cpu()]
        else:
            context = [mx.gpu(i) for i in range(gpus)]
    elif gpus[0] == -2:  # run only on 1 CPU
        context = [mx.cpu()]
    else:  # Run on specific GPUs specified by comma separated device-ids. e.g. 0,1,2
        if logger:
            logger.info('Running on GPU Ids %s' % gpus)
        context = [mx.gpu(i) for i in gpus]
    return context


def parse_steps(steps, epochs, logger=None):
    steps = [int(step) for step in steps.split(',')]
    if steps[0] < 0:
        s = -steps[0]
        if logger:
            logger.info('Dropping LR every %d epochs' % s)
        steps = range(s if s > 1 else s + 1, epochs, s)
    elif steps[0] == 0:
        if logger:
            logger.info('No learning rate decay')
        steps = []
    return steps


def get_lr(lr, epoch, steps, factor):
    """Get learning rate based on schedule."""
    for s in steps:
        if epoch >= s:
            lr *= factor
    return lr


def append_postfix(filename, *args):
    """Appends a set of postfixes to the given filename"""
    file_parts = filename.split('.')
    base = ''.join(file_parts[:-1])
    ext = file_parts[-1]

    for pf in args:
        if not isinstance(pf, str):
            pf = str(pf)
        base += '' if (pf == '' or pf is None) else ('_%s' % pf)
    return '%s.%s' % (base, ext)
