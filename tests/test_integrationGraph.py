"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Test class for the IntegrationGraph which uses the chuffed solver
"""

import unittest2 as unittest
import logging
import os
import tarfile

from .utils import TestWithServer
from serene.elements import SSD
from serene.endpoints import DataSetEndpoint, OntologyEndpoint, SSDEndpoint
from serene.elements.semantics.base import KARMA_DEFAULT_NS


# logging functions...
_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)

class TestIntegrationGraph(TestWithServer):
    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._datasets = None
        self._ontologies = None

        self._benchmark_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests","resources", "museum_benchmark")
        self._museum_owl_dir = os.path.join(self._benchmark_path, "owl")
        self._museum_data = os.path.join(self._benchmark_path, 'dataset')
        self._museum_ssd = os.path.join(self._benchmark_path, 'ssd')

    def setUp(self):
        self._datasets = DataSetEndpoint(self._session)
        self._ontologies = OntologyEndpoint(self._session)
        self._ssds = SSDEndpoint(self._session, self._datasets, self._ontologies)
        self._clear_storage()

        _logger.debug("SetUp complete")

    def tearDown(self):
        """
        Be sure to remove all storage elements once a test is finished...
        :return:
        """
        self._clear_storage()

    def _add_owls(self):
        # add ontologies
        ontologies = []
        for path in os.listdir(self._museum_owl_dir):
            f = os.path.join(self._museum_owl_dir, path)
            ontologies.append(self._ontologies.upload(f))
        return ontologies

    def _add_datasets(self, ontologies):
        # we need to unzip first the data
        if os.path.exists(os.path.join(self._museum_data, "data.tar.gz")):
            with tarfile.open(os.path.join(self._museum_data, "data.tar.gz")) as f:
                f.extractall(path=self._museum_data)

        # add datasets with their ssds
        for ds in os.listdir(self._museum_data):
            if ds.endswith(".gz"):
                continue # skip archive
            ds_f = os.path.join(self._museum_data, ds)
            ds_name = os.path.splitext(ds)[0]
            ssd_f = os.path.join(self._museum_ssd, ds_name + ".ssd")

            _logger.debug("Adding dataset: {}".format(ds_f))
            dataset = self._datasets.upload(ds_f, description="museum_benchmark")

            _logger.debug("Adding ssd: {}".format(ssd_f))
            new_json = dataset.bind_ssd(ssd_f, ontologies, KARMA_DEFAULT_NS)
            empty_ssd = SSD(dataset, ontologies)
            ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)
            self._ssds.upload(ssd)
            # we remove the csv dataset
            os.remove(ds_f)

    def _clear_storage(self):
        """Removes all server elements"""
        for oc in self._serene.octopii.items:
            self._serene.octopii.remove(oc)

        for ssd in self._ssds.items:
            self._ssds.remove(ssd)

        for ds in self._datasets.items:
            self._datasets.remove(ds)

        for on in self._ontologies.items:
            self._ontologies.remove(on)

    def test_get_initial_alignment(self):
        ontologies = self._add_owls()
        self._add_datasets(ontologies)
        self.assertEqual(len(self._ssds.items), 29)
        self.assertEqual(len(self._datasets.items), 29)
        self.assertEqual(len(self._ontologies.items), 11)


    def test_read_config(self):
        self.fail()

    def test_add_matches(self):
        self.fail()

    def test__alignment2dzn(self):
        self.fail()

    def test__matches2dzn(self):
        self.fail()

    def test__generate_oznfzn(self):
        self.fail()

    def test__find_link(self):
        self.fail()

    def test__digraph_to_ssd(self):
        self.fail()

    def test__process_solution(self):
        self.fail()

    def test_solve(self):
        self.fail()
