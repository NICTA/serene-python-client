"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the core module
"""
import unittest2 as unittest
import serene


class TestSchemaMatcher(unittest.TestCase):
    """
    Tests the SchemaMatcher class
    """
    def test_junk(self):
        sn = serene.SchemaMatcher()
        self.assertEqual('a', 'b')
