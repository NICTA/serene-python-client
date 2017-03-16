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
        self._test_owl = os.path.join(path, 'owl', 'dataintegration_report_ontology.ttl')
        self._business_file = os.path.join(path, 'data', 'businessInfo.csv')
        self._business_ssd = os.path.join(path, 'ssd', 'businessInfo.ssd')
        self._cities_file = os.path.join(path, 'data', 'getCities.csv')
        self._cities_ssd = os.path.join(path, 'ssd', 'getCities.ssd')
        self._tricky_cities_ssd = os.path.join(path, 'ssd', 'tricky.ssd')
        self._objects_owl = os.path.join(path, 'owl', 'objects.ttl')
        self._paintings_file = os.path.join(path, 'data', 'paintings.csv')
        self._paintings_ssd = os.path.join(path, 'ssd', 'paintings.ssd')

    def setUp(self):
        self._datasets = DataSetEndpoint(self._session)
        self._ontologies = OntologyEndpoint(self._session)
        self._clear_storage()

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

    def test_evaluate_business(self):
        """
        Tests evaluation for business
        :return:
        """
        ds = self._datasets.upload(self._business_file)
        on = self._ontologies.upload(self._test_owl)

        dataset = self._datasets.items[0]
        print(dataset)
        assert(issubclass(type(dataset), DataSet))

        ontology = self._ontologies.items[0]
        print("namespaces: ", ontology._prefixes)
        print("class nodes: ", list(ontology._iclass_nodes()))
        print("data nodes: ", list(ontology._idata_nodes()))
        print("links: ", list(ontology._ilinks()))

        new_json = dataset.bind_ssd(self._business_ssd, [ontology], str(ontology._prefixes['']))

        print("************************")
        print("new json...")
        pprint(new_json)

        empty_ssd = SSD(dataset, on)
        print("***empty class nodes: ", empty_ssd.class_nodes)
        ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)
        print("***class table: ", ssd._semantic_model._class_table)
        print("***class nodes: ", ssd.class_nodes)
        print("***mappings: ", ssd.mappings)
        print("***data nodes: ", ssd.data_nodes)
        pprint(ssd.json)

        self.assertEqual(len(ssd.class_nodes), 4)
        self.assertEqual(len(ssd.data_nodes), 4)
        self.assertEqual(len(ssd.mappings), 4)
        self.assertEqual(len(ssd.links), 3)   # these are only object properties
        self.assertEqual(new_json, ssd.json)  # somehow check that jsons are appx same

    def test_evaluate_tricky_cities(self):
        """
        Here the ssd has two class nodes of the same type
        :return:
        """
        ds = self._datasets.upload(self._cities_file)
        on = self._ontologies.upload(self._test_owl)

        dataset = self._datasets.items[0]
        print(dataset)
        assert (issubclass(type(dataset), DataSet))

        ontology = self._ontologies.items[0]
        print("namespaces: ", ontology._prefixes)
        print("class nodes: ", list(ontology._iclass_nodes()))
        print("data nodes: ", list(ontology._idata_nodes()))
        print("links: ", list(ontology._ilinks()))

        new_json = dataset.bind_ssd(self._tricky_cities_ssd,
                                    [ontology],
                                    str(ontology._prefixes['']))

        print("************************")
        print("new json...")
        pprint(new_json)

        empty_ssd = SSD(dataset, on)
        ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)
        pprint(ssd.json)

        self.assertEqual(len(ssd.class_nodes), 2)
        self.assertEqual(len(ssd.data_nodes), 2)
        self.assertEqual(len(ssd.mappings), 2)
        self.assertEqual(len(ssd.links), 1)     # these are only object properties
        self.assertEqual(new_json, ssd.json)    # somehow check that jsons are appx same

        self.fail()

    def test_evaluate_paintings(self):
        """
        Here we have a class node with no data nodes
        :return:
        """
        ds = self._datasets.upload(self._paintings_file)
        on = self._ontologies.upload(self._objects_owl)

        dataset = self._datasets.items[0]
        print(dataset)
        assert (issubclass(type(dataset), DataSet))

        ontology = self._ontologies.items[0]
        print("namespaces: ", ontology._prefixes)
        print("class nodes: ", list(ontology._iclass_nodes()))
        print("data nodes: ", list(ontology._idata_nodes()))
        print("links: ", list(ontology._ilinks()))

        new_json = dataset.bind_ssd(self._paintings_ssd, [ontology], str(ontology._prefixes['']))

        print("************************")
        print("new json...")
        pprint(new_json)

        empty_ssd = SSD(dataset, on)
        ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)
        pprint(ssd.json)

        self.assertEqual(len(ssd.class_nodes), 3)
        self.assertEqual(len(ssd.links), 2)
        self.assertEqual(len(ssd.data_nodes), 2)
        self.assertEqual(len(ssd.mappings), 2)
        self.assertEqual(new_json, ssd.json)    # somehow check that jsons are appx same

        self.fail()
