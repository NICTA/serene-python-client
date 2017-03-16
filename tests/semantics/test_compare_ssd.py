"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the ssd module
"""
import os
import logging
from pprint import pprint

from serene.elements import SSD, Ontology, DataSet
from serene.endpoints import DataSetEndpoint, OntologyEndpoint
from ..utils import TestWithServer

# logging functions...
_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)


class TestEvaluateSSD(TestWithServer):
    """
    Tests the comparison of SSDs
    """
    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._datasets = None
        self._ontologies = None

        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
        self._test_file = os.path.join(path, 'data', 'businessInfo.csv')
        self._test_owl = os.path.join(path, 'owl', 'dataintegration_report_ontology.ttl')
        self._test_ssd = os.path.join(path, 'ssd', 'businessInfo.ssd')

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

    def test_evaluate1(self):
        """
        Tests evaluation for business
        :return:
        """
        ds = self._datasets.upload(self._test_file)
        on = self._ontologies.upload(self._test_owl)

        dataset = self._datasets.items[0]
        print(dataset)
        assert(issubclass(type(dataset), DataSet))

        ontologies = self._ontologies.items

        new_json = dataset.bind_ssd(self._test_ssd, ontologies)

        print("************************")
        print("new json...")
        pprint(new_json)

        empty_ssd = SSD(dataset, on)
        ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)
        print(ssd)

        self.fail()
