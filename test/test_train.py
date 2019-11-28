from unittest import TestCase
from os.path import dirname, realpath, join
from os import remove
import glob

try:
    # python 3.4+ should use builtin unittest.mock not mock package
    from unittest.mock import patch
except ImportError:
    from mock import patch

from train_triplet_semihard import train_triplet_semihard, parse_args as parse_triplet_semihard_args
from train_liftedstruct import train_liftedstruct, parse_args as parse_liftedstruct_args
from train_npairs import train_npairs, parse_args as parse_npairs_args
from train_clusterloss import train_clusterloss, parse_args as parse_clusterloss_args
from train_margin import train_margin, parse_args as parse_margin_args
from train_prototype import train_prototype, parse_args as parse_prototype_args
from train_proxy import train_proxy, parse_args as parse_proxy_args
from train_angular import train_angular, parse_args as parse_angular_args
from train_normproxy import train_normproxy, parse_args as parse_normproxy_args
from train_dreml import train_dreml, parse_args as parse_dreml_args
from train_rankedlistloss import train_rankedlist, parse_args as parse_rankedlist_args
from train_discriminative import train_discriminative, parse_args as parse_discriminative_args


class TestTrain_triplet_semihard(TestCase):
    def setUp(self):
        def cleanup():
            # erase param files created during training
            param_files = glob.glob('unittest_model_triplet_semihard*.params')
            for pfile in param_files:
                remove(pfile)
        self.addCleanup(cleanup)

    def test_train_triplet_semihard(self):
        testargs = ['train_triplet_semihard',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_triplet_semihard',
                    '--batch-size', '4'
                    ]
        with patch('sys.argv', testargs):
            train_triplet_semihard(parse_triplet_semihard_args())

    def test_train_triplet_semihard_resnet(self):
        testargs = ['train_triplet_semihard',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--logfile', 'None',
                    '--epochs', '1',
                    '--num-workers', '0',
                    '--model', 'resnet50_v2',
                    '--save-model-prefix', 'None',
                    '--number-of-runs', '2',
                    '--batch-size', '4'
                    ]
        with patch('sys.argv', testargs):
            train_triplet_semihard(parse_triplet_semihard_args())


class TestTrain_liftedstruct(TestCase):
    def setUp(self):
        def cleanup():
            # erase param files created during training
            param_files = glob.glob('unittest_model_liftedstruct*.params')
            for pfile in param_files:
                remove(pfile)
        self.addCleanup(cleanup)

    def test_train_liftedstruct(self):
        testargs = ['train_liftedstruct',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_liftedstruct',
                    '--batch-size', '4'
                    ]
        with patch('sys.argv', testargs):
            train_liftedstruct(parse_liftedstruct_args())

    def test_train_liftedstruct_resnet(self):
        testargs = ['train_liftedstruct',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--logfile', 'None',
                    '--epochs', '1',
                    '--num-workers', '0',
                    '--model', 'resnet50_v2',
                    '--save-model-prefix', 'None',
                    '--number-of-runs', '2',
                    '--batch-size', '4'
                    ]
        with patch('sys.argv', testargs):
            train_triplet_semihard(parse_triplet_semihard_args())


class TestTrain_npairs(TestCase):
    def setUp(self):
        def cleanup():
            # erase param files created during training
            param_files = glob.glob('unittest_model_npairs*.params')
            for pfile in param_files:
                remove(pfile)
        self.addCleanup(cleanup)

    def test_train_npairs(self):
        testargs = ['train_npairs',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--epoch-length', '10',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_npairs',
                    '--batch-size', '4'
                    ]
        with patch('sys.argv', testargs):
            train_npairs(parse_npairs_args())

    def test_train_npairs_resnet(self):
        testargs = ['train_npairs',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--logfile', 'None',
                    '--epochs', '1',
                    '--num-workers', '0',
                    '--model', 'resnet50_v2',
                    '--save-model-prefix', 'None',
                    '--number-of-runs', '2',
                    '--batch-size', '4',
                    '--epoch-length', '10',
                    ]
        with patch('sys.argv', testargs):
            train_npairs(parse_npairs_args())


class TestTrain_clusterloss(TestCase):
    def setUp(self):
        def cleanup():
            # erase param files created during training
            param_files = glob.glob('unittest_model_clusterloss*.params')
            for pfile in param_files:
                remove(pfile)
        self.addCleanup(cleanup)

    def test_train_clusterloss(self):
        testargs = ['train_clusterloss',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--iteration-per-epoch', '10',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_clusterloss',
                    '--batch-size', '4',
                    '--batch-k', '2',
                    ]
        with patch('sys.argv', testargs):
            train_clusterloss(parse_clusterloss_args())

    def test_train_clusterloss_resnet50_v2(self):
        testargs = ['train_clusterloss',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--iteration-per-epoch', '0',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_clusterloss',
                    '--batch-size', '4',
                    '--batch-k', '2',
                    '--model', 'resnet50_v2',
                    ]
        with patch('sys.argv', testargs):
            train_clusterloss(parse_clusterloss_args())


class TestTrain_margin(TestCase):
    def setUp(self):
        def cleanup():
            # erase param files created during training
            param_files = glob.glob('unittest_model_margin*.params')
            for pfile in param_files:
                remove(pfile)
        self.addCleanup(cleanup)

    def test_train_margin(self):
        testargs = ['train_margin',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--iteration-per-epoch', '10',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_margin',
                    '--batch-size', '4',
                    '--batch-k', '2',
                    ]
        with patch('sys.argv', testargs):
            train_margin(parse_margin_args())

    def test_train_margin_resnet50_v2(self):
        testargs = ['train_margin',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--iteration-per-epoch', '10',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_margin',
                    '--batch-size', '4',
                    '--batch-k', '2',
                    '--model', 'resnet50_v2',
                    '--lr-beta', '0',
                    ]
        with patch('sys.argv', testargs):
            train_margin(parse_margin_args())


class TestTrain_prototype(TestCase):
    def setUp(self):
        def cleanup():
            # erase param files created during training
            param_files = glob.glob('unittest_model_prototype*.params')
            for pfile in param_files:
                remove(pfile)
        self.addCleanup(cleanup)

    def test_train_prototype(self):
        testargs = ['train_prototype',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--iteration-per-epoch', '5',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_prototype',
                    '--batch-size', '10',
                    '--nc', '3',
                    '--nq', '2',
                    '--ns', '4'
                    ]
        with patch('sys.argv', testargs):
            train_prototype(parse_prototype_args())

    def test_train_prototype_resnet50_v2(self):
        testargs = ['train_prototype',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--iteration-per-epoch', '5',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_prototype',
                    '--batch-size', '10',
                    '--nc', '3',
                    '--nq', '2',
                    '--ns', '4',
                    '--model', 'resnet50_v2',
                    ]
        with patch('sys.argv', testargs):
            train_prototype(parse_prototype_args())


class TestTrain_proxy(TestCase):
    def setUp(self):
        def cleanup():
            # erase param files created during training
            param_files = glob.glob('unittest_model_proxy*.params')
            for pfile in param_files:
                remove(pfile)
        self.addCleanup(cleanup)

    def test_train_proxy(self):
        testargs = ['train_proxy',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--iteration-per-epoch', '0',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_proxy',
                    '--batch-size', '10',
                    ]
        with patch('sys.argv', testargs):
            train_proxy(parse_proxy_args())

    def test_train_proxy_iter(self):
        testargs = ['train_proxy',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--iteration-per-epoch', '5',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_proxy',
                    '--batch-size', '10',
                    ]
        with patch('sys.argv', testargs):
            train_proxy(parse_proxy_args())

    def test_train_proxy_resnet50_v2(self):
        testargs = ['train_proxy',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_proxy',
                    '--batch-size', '10',
                    '--model', 'resnet50_v2',
                    ]
        with patch('sys.argv', testargs):
            train_proxy(parse_proxy_args())


class TestTrain_angular(TestCase):
    def setUp(self):
        def cleanup():
            # erase param files created during training
            param_files = glob.glob('unittest_model_angular*.params')
            for pfile in param_files:
                remove(pfile)
        self.addCleanup(cleanup)

    def test_train_angular(self):
        testargs = ['train_angular',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--epoch-length', '10',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_angular',
                    '--batch-size', '4'
                    ]
        with patch('sys.argv', testargs):
            train_angular(parse_angular_args())

    def test_train_angular_resnet(self):
        testargs = ['train_angular',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--logfile', 'None',
                    '--epochs', '1',
                    '--epoch-length', '10',
                    '--num-workers', '0',
                    '--model', 'resnet50_v2',
                    '--save-model-prefix', 'None',
                    '--number-of-runs', '2',
                    '--batch-size', '4'
                    ]
        with patch('sys.argv', testargs):
            train_angular(parse_angular_args())


class TestTrain_normproxy(TestCase):
    def setUp(self):
        def cleanup():
            # erase param files created during training
            param_files = glob.glob('unittest_model_normproxy*.params')
            for pfile in param_files:
                remove(pfile)
        self.addCleanup(cleanup)

    def test_train_normproxy(self):
        testargs = ['train_normproxy',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_normproxy',
                    '--batch-size', '10',
                    ]
        with patch('sys.argv', testargs):
            train_normproxy(parse_normproxy_args())

    def test_train_normproxy_resnet(self):
        testargs = ['train_normproxy',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_normproxy',
                    '--batch-size', '10',
                    '--model', 'resnet50_v2',
                    '--number-of-runs', '2',
                    ]
        with patch('sys.argv', testargs):
            train_normproxy(parse_normproxy_args())


class TestTrain_dreml(TestCase):
    def setUp(self):
        def cleanup():
            # erase param files created during training
            param_files = glob.glob('unittest_model_dreml*.params')
            for pfile in param_files:
                remove(pfile)
        self.addCleanup(cleanup)

    def test_train_dreml(self):
        testargs = ['train_dreml',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_dreml',
                    '--batch-size', '10',
                    '--number-of-ensembles', '3',
                    '--meta-classes', '3',
                    ]
        with patch('sys.argv', testargs):
            train_dreml(parse_dreml_args())

    def test_train_dreml_resnet(self):
        testargs = ['train_dreml',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_dreml',
                    '--batch-size', '10',
                    '--model', 'resnet50_v2',
                    '--number-of-ensembles', '3',
                    '--meta-classes', '3',
                    '--number-of-runs', '2',
                    ]
        with patch('sys.argv', testargs):
            train_dreml(parse_dreml_args())


class TestTrain_rankedlistloss(TestCase):
    def setUp(self):
        def cleanup():
            # erase param files created during training
            param_files = glob.glob('unittest_model_rankedlistloss*.params')
            for pfile in param_files:
                remove(pfile)
        self.addCleanup(cleanup)

    def test_train_rankedlistloss(self):
        testargs = ['train_rankedlist',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_rankedlistloss',
                    '--batch-size', '12',
                    '--batch-k', '3',
                    '--iteration-per-epoch', '10',
                    ]
        with patch('sys.argv', testargs):
            train_rankedlist(parse_rankedlist_args())

    def test_train_rankedlistloss_resnet(self):
        testargs = ['train_rankedlist',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_rankedlistloss',
                    '--batch-size', '12',
                    '--batch-k', '3',
                    '--model', 'resnet50_v2',
                    '--number-of-runs', '2',
                    '--iteration-per-epoch', '10',
                    ]
        with patch('sys.argv', testargs):
            train_rankedlist(parse_rankedlist_args())


class TestTrain_discriminative(TestCase):
    def setUp(self):
        def cleanup():
            # erase param files created during training
            param_files = glob.glob('unittest_model_discriminative*.params')
            for pfile in param_files:
                remove(pfile)
        self.addCleanup(cleanup)

    def test_train_discriminative(self):
        testargs = ['train_discriminative',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_discriminative',
                    '--batch-size', '12',
                    ]
        with patch('sys.argv', testargs):
            train_discriminative(parse_discriminative_args())

    def test_train_discriminative_resnet(self):
        testargs = ['train_discriminative',
                    '--dataset', 'miniCUB',
                    '--data-path', join(dirname(realpath(__file__)), 'microcub'),
                    '--epochs', '1',
                    '--num-workers', '0',
                    '--logfile', 'None',
                    '--save-model-prefix', 'unittest_model_discriminative',
                    '--batch-size', '12',
                    '--model', 'resnet50_v2',
                    '--number-of-runs', '2',
                    ]
        with patch('sys.argv', testargs):
            train_discriminative(parse_discriminative_args())
