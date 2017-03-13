"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the ssd module
"""
import unittest2 as unittest
import serene
from serene.matcher.dataset import DataSet
import os
from serene.elements import ClassNode, DataNode, Link
from serene.semantics.ssd import SSD
from ..utils import TestWithServer
from serene.endpoints import DataSetEndpoint, OntologyEndpoint


class TestSSD(TestWithServer):
    """
    Tests the SSD class
    """
    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._datasets = None
        self._ontologies = None

        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
        self._test_file = os.path.join(path, 'data','businessInfo.csv')
        self._test_owl = os.path.join(path, 'owl', 'dataintegration_report_ontology.owl')
        self._test_ssd = os.path.join(path, 'ssd','businessInfo.ssd')

    def setUp(self):
        self._datasets = DataSetEndpoint(self._session)
        self._ontologies = OntologyEndpoint(self._session)
        self._clear_storage()
        assert(os.path.isfile(self._test_file))

    def _clear_storage(self):
        """Removes all server elements"""
        for ds in self._datasets.items:
            self._datasets.remove(ds)

        for on in self._ontologies.items:
            self._ontologies.remove(on)

    def tearDown(self):
        """
        Be sure to remove all storage elements once a test is finished...
        :return:
        """
        self._clear_storage()

    def test_create(self):
        """
        Tests the SSD creation
        :return:
        """
        ds = self._datasets.upload(self._test_file)
        on = self._ontologies.upload(self._test_owl)

        single = SSD(dataset=ds, ontology=on)

        self.assertEqual(len(single.data_nodes), 4)
        self.assertEqual(len(single.links), 0)
        # self.assertIsNotNone(ClassNode.search(single.class_nodes, ClassNode("hello")))
        # self.assertEqual(single.class_nodes[0].name, "hello")
