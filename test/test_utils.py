import unittest
from unittest import TestCase

import numpy as np
import mxnet as mx

from common.utils import isjpeg, average_results, format_results, get_context


class TestUtils(TestCase):
    def test_isjpeg(self):
        assert isjpeg('dummy.jpg')
        assert isjpeg('dummy.JPG')
        assert isjpeg('dummy.JpG')
        assert isjpeg('dummy.jpeg')
        assert isjpeg('dummy.JPEG')
        assert isjpeg('dummy.JPeG')

        assert not isjpeg('dummy.jpa')
        assert not isjpeg('dummy.jpA')
        assert not isjpeg('jpg')

    def test_average_results(self):
        results = [
            [('value1', 0), ('value2', 100)],
            [('value1', 100), ('value2', 0)]
        ]
        assert np.all(average_results(results) == np.array([50, 50]))
        assert average_results(None) is None
        assert average_results([]) is None

    def test_format_results(self):
        result = np.array([1,2])
        assert format_results(result) == ' R@1 \t R@2\n1.00\t2.00'
        assert format_results(None) == ''


class TestContextUtils(TestCase):
    def test_get_context_cpu(self):
        assert get_context(-1) == [mx.cpu()]

    @unittest.skipUnless(mx.context.num_gpus() > 0, 'Host has no GPU, skipping test')
    def test_get_context_gpu(self):
        assert get_context(0) == [mx.gpu(0)]
