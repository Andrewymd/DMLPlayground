import itertools

from nose.loader import TestLoader
from nose import run
from nose.suite import LazySuite


def run_my_tests():
    all_tests = ()
    all_tests = itertools.chain(all_tests, TestLoader().loadTestsFromDir('.'))
    suite = LazySuite(all_tests)
    run(suite=suite, argv=['run_all.py', '--cover-branches', '--with-coverage', '--cover-html'])


if __name__ == '__main__':
    run_my_tests()
