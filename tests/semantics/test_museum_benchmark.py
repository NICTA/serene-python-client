"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the museum benchmark
"""
import os
import logging
from pprint import pprint

from serene.elements import SSD, Ontology, DataSet
from serene.endpoints import DataSetEndpoint, OntologyEndpoint, SSDEndpoint
from ..utils import TestWithServer
from serene.elements.semantics.base import KARMA_DEFAULT_NS

# logging functions...
_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)


class TestMuseum(TestWithServer):
    """
    Tests the JsonReader/JsonWriter for SSD
    """
    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._datasets = None
        self._ontologies = None

        benchmark_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "museum_benchmark")
        self._museum_owl_dir = os.path.join(benchmark_path, "owl")
        self._museum_data = os.path.join(benchmark_path, 'dataset')
        self._museum_ssd = os.path.join(benchmark_path, 'ssd')

    def setUp(self):
        self._datasets = DataSetEndpoint(self._session)
        self._ontologies = OntologyEndpoint(self._session)
        self._ssds = SSDEndpoint(self._session, self._datasets, self._ontologies)
        self._clear_storage()

        # add ontologies
        ontologies = []
        for path in os.listdir(self._museum_owl_dir):
            f = os.path.join(self._museum_owl_dir, path)
            ontologies.append(self._ontologies.upload(f))

        self.assertEqual(len(self._ontologies.items), 11)

        # add datasets with their ssds
        for ds in os.listdir(self._museum_data):
            ds_f = os.path.join(self._museum_data, ds)
            ds_name = os.path.splitext(ds)[0]
            ssd_f = os.path.join(self._museum_ssd, ds_name+".ssd")

            _logger.debug("Adding dataset: ", ds_f)
            dataset = self._datasets.upload(ds_f, description="museum_benchmark")

            _logger.debug("Adding ssd: ", ssd_f)
            new_json = dataset.bind_ssd(ssd_f, ontologies, KARMA_DEFAULT_NS)
            empty_ssd = SSD(dataset, ontologies)
            ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)
            self._ssds.upload(ssd)

        _logger.debug("SetUp complete")


    def _clear_storage(self):
        """Removes all server elements"""
        for ssd in self._ssds.items:
            self._ssds.remove(ssd)

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


    def test_evaluate_museum(self):
        """
        Here we have a class node with no data nodes, a list of ontologies, class instance link.
        Not all columns from file get mapped.
        :return:
        """
        self.assertEqual(len(self._ssds.items), 29)
        self.assertEqual(len(self._datasets.items), 29)
        self.assertEqual(len(self._ontologies.items), 11)