"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the core module
"""
import unittest2 as unittest
import os

from .utils import SereneTestServer
from serene.endpoints import DataSetEndpoint, OntologyEndpoint
from serene.api.session import Session


class TestEndpoints(unittest.TestCase):
    """
    Tests the Endpoints class
    """
    __server__ = None

    #@classmethod
    #def setUpClass(cls):
    #    cls.__server__ = SereneTestServer()
    #    cls.__server__.setup()
    #    print("DONE WITH SETUP")

    @classmethod
    def tearDownClass(cls):
        cls.__server__.tear_down()

    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        TestEndpoints.__server__ = SereneTestServer()
        TestEndpoints.__server__.setup()
        self._session = Session(
            TestEndpoints.__server__.host,
            TestEndpoints.__server__.port,
            None,
            None,
            False
        )


class TestDataSetEndpoint(TestEndpoints):
    """
    Tests the dataset endpoint
    """
    def __init__(self, method_name="runTest"):
        super().__init__(method_name)

        self._datasets = DataSetEndpoint(self._session)
        self._clear_datasets()

        self._test_file = 'tests/resources/data/businessInfo.csv'
        assert(os.path.isfile(self._test_file))

    def _clear_datasets(self):
        """Removes all datasets"""
        for ds in self._datasets.items:
            self._datasets.remove(ds)

    def tearDown(self):
        """
        Be sure to remove all datasets once a test is finished...
        :return:
        """
        self._clear_datasets()

    def test_upload(self):
        """
        Tests the uploading of a dataset
        :return:
        """
        self.assertEqual(len(self._datasets.items), 0)

        # only a dataset string path should be allowed
        with self.assertRaises(Exception):
            self._datasets.upload(1234)

        # only an existing dataset string path should be allowed
        with self.assertRaises(Exception):
            self._datasets.upload('resources/data/doesnt-exist.csv')

        # only a dataset string path should be allowed
        d = self._datasets.upload(self._test_file)

        self.assertEqual(len(self._datasets.items), 1)
        self.assertEqual(d.filename, os.path.basename(self._test_file))
        self.assertEqual(
            [c.name for c in d.columns],
            ['company', 'ceo', 'city', 'state']
        )