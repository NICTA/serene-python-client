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
    _server = None
    _session = None

    @classmethod
    def setUpClass(cls):
        cls._server = SereneTestServer()
        cls._server.setup()
        cls._session = Session(
            cls._server.host,
            cls._server.port,
            None,
            None,
            False
        )

    @classmethod
    def tearDownClass(cls):
        cls._server.tear_down()


class TestDataSetEndpoint(TestEndpoints):
    """
    Tests the dataset endpoint
    """
    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._datasets = None
        self._test_file = 'tests/resources/data/businessInfo.csv'

    def setUp(self):
        self._datasets = DataSetEndpoint(self._session)
        self._clear_datasets()
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

        # now upload a nice dataset
        d = self._datasets.upload(self._test_file)

        self.assertEqual(len(self._datasets.items), 1)
        self.assertEqual(d.filename, os.path.basename(self._test_file))
        self.assertEqual(
            [c.name for c in d.columns],
            ['company', 'ceo', 'city', 'state']
        )

    def test_remove(self):
        """
        Tests the removal of a dataset
        :return:
        """
        self.assertEqual(len(self._datasets.items), 0)

        # upload some datasets
        d1 = self._datasets.upload(self._test_file)
        d2 = self._datasets.upload(self._test_file)
        self.assertEqual(len(self._datasets.items), 2)

        # now let's remove one
        self._datasets.remove(d1)
        self.assertEqual(len(self._datasets.items), 1)
        self.assertEqual(self._datasets.items[0].id, d2.id)

        # let's remove the other by id
        self._datasets.remove(d2.id)
        self.assertEqual(len(self._datasets.items), 0)

    def test_items(self):
        """
        Tests item manipulation for the dataset list
        :return:
        """
        self.assertEqual(len(self._datasets.items), 0)

        # upload some datasets
        d1 = self._datasets.upload(self._test_file)
        d2 = self._datasets.upload(self._test_file)
        self.assertEqual(len(self._datasets.items), 2)

        # now let's remove one
        self._datasets.remove(d1)
        self.assertEqual(len(self._datasets.items), 1)

        # make sure the right one remains
        self.assertEqual(self._datasets.items[0], d2)

        # now let's remove the last one
        self._datasets.remove(d2)
        self.assertEqual(len(self._datasets.items), 0)


class TestOntologyEndpoint(TestEndpoints):
    """
    Tests the ontology endpoint
    """
    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._datasets = None
        self._test_file = 'tests/resources/owl/dataintegration_report_ontology.owl'

    def setUp(self):
        self._ontologies = OntologyEndpoint(self._session)
        self._clear_ontologies()
        assert(os.path.isfile(self._test_file))

    def _clear_ontologies(self):
        """Removes all ontologies"""
        for on in self._ontologies.items:
            self._ontologies.remove(on)

    def tearDown(self):
        """
        Be sure to remove all datasets once a test is finished...
        :return:
        """
        self._clear_ontologies()

    def test_upload(self):
        """
        Tests the uploading of a ontology
        :return:
        """
        description_string = "description-string"

        self.assertEqual(len(self._ontologies.items), 0)

        # only a ontology string path should be allowed
        with self.assertRaises(Exception):
            self._ontologies.upload(1234)

        # only an existing ontology string path should be allowed
        with self.assertRaises(Exception):
            self._ontologies.upload('resources/owl/doesnt-exist.owl')

        # now upload a nice ontology
        on = self._ontologies.upload(
            self._test_file,
            description=description_string,
            owl_format='owl'
        )

        self.assertEqual(len(self._ontologies.items), 1)
        self.assertEqual(on.name, os.path.basename(self._test_file))
        self.assertEqual(on.description, description_string)
        self.assertEqual(len(on.class_nodes), 6)

    # def test_remove(self):
    #     """
    #     Tests the removal of a dataset
    #     :return:
    #     """
    #     self.assertEqual(len(self._datasets.items), 0)
    #
    #     # upload some datasets
    #     d1 = self._datasets.upload(self._test_file)
    #     d2 = self._datasets.upload(self._test_file)
    #     self.assertEqual(len(self._datasets.items), 2)
    #
    #     # now let's remove one
    #     self._datasets.remove(d1)
    #     self.assertEqual(len(self._datasets.items), 1)
    #     self.assertEqual(self._datasets.items[0].id, d2.id)
    #
    #     # let's remove the other by id
    #     self._datasets.remove(d2.id)
    #     self.assertEqual(len(self._datasets.items), 0)
    #
    # def test_items(self):
    #     """
    #     Tests Item manipulation for the dataset list
    #     :return:
    #     """
    #     self.assertEqual(len(self._datasets.items), 0)
    #
    #     # upload some datasets
    #     d1 = self._datasets.upload(self._test_file)
    #     d2 = self._datasets.upload(self._test_file)
    #     self.assertEqual(len(self._datasets.items), 2)
    #
    #     # now let's remove one
    #     self._datasets.remove(d1)
    #     self.assertEqual(len(self._datasets.items), 1)
    #
    #     # make sure the right one remains
    #     self.assertEqual(self._datasets.items[0], d2.id)
    #
    #     # now let's remove the last one
    #     self._datasets.remove(d2)
    #     self.assertEqual(len(self._datasets.items), 0)
