from argparse import ArgumentParser

from dataset import dataloaders


class TrainingParser(ArgumentParser):
    """
    Superclass used for parsing options for DML model training
    """

    def __init__(self, default_logfile='None', default_model_prefix='None', remove_arguments=None, *args, **kwargs):
        super(TrainingParser, self).__init__(*args, **kwargs)
        # Compute context
        self.add(remove_arguments, '--gpus', type=str, default='-1',
                 help='Where to run model gpu/cpu. -1=run on all available gpus (auto-detect),'
                      '-2=run only on 1 cpu, 0,1= run on gpu1,2')
        self.add(remove_arguments, '--kvstore', type=str, default='device',
                 help='kvstore to use for trainer.')

        # Dataset params
        self.add(remove_arguments, '--dataset', type=str, default='CUB', choices=dataloaders.keys(),
                 help='Dataset to run the training on. Default is CUB.')
        self.add(remove_arguments, '--data-path', type=str, default='/DATA/Datastore/CUB_200_2011',
                 help='Path of the logo image data')
        self.add(remove_arguments, '--use-crops', action="store_true",
                 help='Use bounding boxes in the dataset to crop images')
        self.add(remove_arguments, '--num-workers', type=int, default=4,
                 help='Number of worker threads to use')

        # Model params
        self.add(remove_arguments, '--embed-dim', type=int, default=64,
                 help='dimensionality of image embedding. default is 64.')
        self.add(remove_arguments, '--model', type=str, default='inception-bn',
                 choices=['resnet18_v2', 'resnet50_v2', 'inception-bn'],
                 help='Type of model to use. Default is Inception-BN.')

        # Optimizer params
        self.add(remove_arguments, '--optimizer', type=str, default='adam',
                 help='optimizer. default is adam.')

        # Run options
        self.add(remove_arguments, '--seed', type=int, default=123,
                 help='Random seed to use. default=123.')
        self.add(remove_arguments, '--number-of-runs', type=int, default=1,
                 help='Number of runs. Final results will be averaged.')
        self.add(remove_arguments, '--skip-pretrain-validation', action="store_true",
                 help='Skips validation before training')
        self.add(remove_arguments, '--logfile', type=str, default=default_logfile,
                 help='Name of the log file. None will disable file logging. Default is None.')
        self.add(remove_arguments, '--save-model-prefix', type=str, default=default_model_prefix,
                 help='prefix of models to be saved.')
        self.add(remove_arguments, '--log-postfix', type=str, default='', help='Postfix added to the logfile\'s name')
        self.add(remove_arguments, '--no-nmi', dest='nmi', action="store_false", help='Runs evaluation without NMI.')
        self.add(remove_arguments, '--disable-hybridize', action='store_true', help='Disables model and loss hybridization')

        self.add(remove_arguments, '--load-checkpoint', action="store_true", help='Continue from the previous run.')
        # TODO: start epoch

    def add(self, remove_arguments, *args, **kwargs):
        if remove_arguments is not None and args[0] in remove_arguments:
            return
        self.add_argument(*args, **kwargs)
