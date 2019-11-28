from unittest import TestCase

from models import get_googlenet, get_inception_bn


class TestBackbones(TestCase):

    def test_get_googlenet(self):
        net, params = get_googlenet()
        net.load_parameters(params, ignore_extra=True)

    def test_get_inceptionbn(self):
        net, params = get_inception_bn()
        net.load_parameters(params, ignore_extra=True)
