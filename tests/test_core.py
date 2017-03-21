"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the core module
"""
import unittest2 as unittest
import serene
from .utils import TestWithServer


class TestSerene(TestWithServer):
    """
    Tests the SemanticModeller class
    """
    def test_load(self):

