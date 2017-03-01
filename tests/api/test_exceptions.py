"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the core module
"""
import unittest2 as unittest


class TestException(unittest.TestCase):
    """
    Tests the Exceptions
    """
    def test_nothing(self):
        raise NotImplementedError("Test not implemented")

