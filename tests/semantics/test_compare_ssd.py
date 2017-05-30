"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the ssd module
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
        self._museum_owl_dir = os.path.join(path, 'owl', 'museum_edm')
        self._business_file = os.path.join(path, 'data', 'businessInfo.csv')
        self._business_ssd = os.path.join(path, 'ssd', 'businessInfo.ssd')
        self._cities_file = os.path.join(path, 'data', 'getCities.csv')
        self._cities_ssd = os.path.join(path, 'ssd', 'getCities.ssd')
        self._tricky_cities_ssd = os.path.join(path, 'ssd', 'tricky.ssd')
        self._objects_owl = os.path.join(path, 'owl', 'objects.ttl')
        self._paintings_file = os.path.join(path, 'data', 'paintings.csv')
        self._paintings_ssd = os.path.join(path, 'ssd', 'paintings.ssd')
        self._museum_file = os.path.join(path, 'data', 'museum.csv')
        self._museum_ssd = os.path.join(path, 'ssd', 'museum.ssd')

        self._ssd_path = os.path.join(path, 'ssd')
        self._data_path = os.path.join(path, 'data')

    def setUp(self):
        self._datasets = DataSetEndpoint(self._session)
        self._ontologies = OntologyEndpoint(self._session)
        self._ssds = SSDEndpoint(self._session, self._datasets, self._ontologies)
        self._clear_storage()

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

    def test_evaluate_business(self):
        """
        Tests evaluation for business
        :return:
        """
        dataset = self._datasets.upload(self._business_file)
        on = self._ontologies.upload(self._test_owl)

        assert(issubclass(type(dataset), DataSet))

        ontology = self._ontologies.items[0]

        new_json = dataset.bind_ssd(self._business_ssd,
                                    [ontology],
                                    str(ontology._prefixes['']))

        empty_ssd = SSD(dataset, on)
        ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)

        self.assertEqual(len(ssd.class_nodes), 4)
        self.assertEqual(len(ssd.data_nodes), 4)
        self.assertEqual(len(ssd.data_links), 4)   # these are only data properties
        self.assertEqual(len(ssd.object_links), 3)   # these are only object properties

        res = self._ssds.compare(ssd, ssd)
        print(res)

        self.assertEqual(res['precision'], 1)
        self.assertEqual(res['recall'], 1)
        self.assertEqual(res['jaccard'], 1)

    def test_evaluate_tricky_cities(self):
        """
        Here the ssd has two class nodes of the same type
        :return:
        """
        self._datasets.upload(self._cities_file)
        on = self._ontologies.upload(self._test_owl)

        dataset = self._datasets.items[0]
        assert (issubclass(type(dataset), DataSet))

        ontology = self._ontologies.items[0]

        new_json = dataset.bind_ssd(self._tricky_cities_ssd,
                                    [ontology],
                                    str(ontology._prefixes['']))

        empty_ssd = SSD(dataset, on)
        ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)

        self.assertEqual(len(ssd.class_nodes), 2)
        self.assertEqual(len(ssd.data_nodes), 2)
        self.assertEqual(len(ssd.mappings), 2)
        self.assertEqual(len(ssd.object_links), 1)     # these are only object properties

        res = self._ssds.compare(ssd, ssd)
        print(res)

        self.assertEqual(res['precision'], 1)
        self.assertEqual(res['recall'], 1)
        self.assertEqual(res['jaccard'], 1)

    def test_ssd_evaluate_tricky_cities(self):
        """
        Here the ssd has two class nodes of the same type.
        We use the evaluate method of ssd and not from the server.
        :return:
        """
        self._datasets.upload(self._cities_file)
        on = self._ontologies.upload(self._test_owl)

        dataset = self._datasets.items[0]
        assert (issubclass(type(dataset), DataSet))

        ontology = self._ontologies.items[0]

        new_json = dataset.bind_ssd(self._tricky_cities_ssd,
                                    [ontology],
                                    str(ontology._prefixes['']))

        empty_ssd = SSD(dataset, on)
        ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)

        self.assertEqual(len(ssd.class_nodes), 2)
        self.assertEqual(len(ssd.data_nodes), 2)
        self.assertEqual(len(ssd.mappings), 2)
        self.assertEqual(len(ssd.object_links), 1)     # these are only object properties

        res = ssd.evaluate(ssd)
        print(res)

        self.assertEqual(res['precision'], 1)
        self.assertEqual(res['recall'], 1)
        self.assertEqual(res['jaccard'], 1)

    def test_evaluate_country_names(self):
        """
        Tests evaluation for country_names
        :return:
        """
        path = os.path.join(self._data_path, "country_names.csv")
        dataset = self._datasets.upload(path)
        on = self._ontologies.upload(self._test_owl)

        assert (issubclass(type(dataset), DataSet))

        ssd_path = os.path.join(self._ssd_path, "country_names.ssd")
        new_json = dataset.bind_ssd(ssd_path,
                                    [on],
                                    str(on._prefixes['']))

        empty_ssd = SSD(dataset, on)
        ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)
        pprint(ssd.json)

        self.assertEqual(len(ssd.class_nodes), 1)
        self.assertEqual(len(ssd.data_nodes), 4)
        self.assertEqual(len(ssd.mappings), 4)
        #self.assertEqual(len(ssd.links), 4)  # class and data links
        self.assertEqual(len(ssd.data_links), 4)  # these are only data properties
        self.assertEqual(len(ssd.object_links), 0)  # these are only object properties

        res = self._ssds.compare(ssd, ssd, False, False)
        print(res)

        self.assertEqual(res['precision'], 1)
        self.assertEqual(res['recall'], 1)
        self.assertEqual(res['jaccard'], 1)

    def test_evaluate_country_names_zero(self):
        """
        Tests evaluation for country_names
        If we ignore everything, then it will all 0
        :return:
        """
        path = os.path.join(self._data_path, "country_names.csv")
        dataset = self._datasets.upload(path)
        on = self._ontologies.upload(self._test_owl)

        ssd_path = os.path.join(self._ssd_path, "country_names.ssd")
        new_json = dataset.bind_ssd(ssd_path,
                                    [on],
                                    str(on._prefixes['']))

        empty_ssd = SSD(dataset, on)
        ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)

        res = self._ssds.compare(ssd, ssd, True, True)
        print(res)

        self.assertEqual(res['precision'], 0)
        self.assertEqual(res['recall'], 0)
        self.assertEqual(res['jaccard'], 0)

    def test_evaluate_places_dif(self):
        """
        Tests evaluation for places_dif
        :return:
        """
        path = os.path.join(self._data_path, "places_dif.csv")
        dataset = self._datasets.upload(path)
        on = self._ontologies.upload(self._test_owl)

        assert (issubclass(type(dataset), DataSet))

        ssd_path = os.path.join(self._ssd_path, "places_dif.ssd")
        new_json = dataset.bind_ssd(ssd_path,
                                    [on],
                                    str(on._prefixes['']))

        empty_ssd = SSD(dataset, on)
        ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)
        pprint(ssd.json)

        self.assertEqual(len(ssd.class_nodes), 4)
        self.assertEqual(len(ssd.data_nodes), 4)
        self.assertEqual(len(ssd.mappings), 4)
        self.assertEqual(len(ssd.links), 7)  # class and data links
        self.assertEqual(len(ssd.data_links), 4)  # these are only data properties
        self.assertEqual(len(ssd.object_links), 3)  # these are only object properties

        res = self._ssds.compare(ssd, ssd)
        print(res)

        self.assertEqual(res['precision'], 1)
        self.assertEqual(res['recall'], 1)
        self.assertEqual(res['jaccard'], 1)

    def test_evaluate_places_mix(self):
        """
        Tests evaluation for places_mix
        :return:
        """
        path = os.path.join(self._data_path, "places_mix.csv")
        dataset = self._datasets.upload(path)
        on = self._ontologies.upload(self._test_owl)

        assert (issubclass(type(dataset), DataSet))

        ssd_path = os.path.join(self._ssd_path, "places_mix.ssd")
        new_json = dataset.bind_ssd(ssd_path,
                                    [on],
                                    str(on._prefixes['']))

        empty_ssd = SSD(dataset, on)
        ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)
        pprint(ssd.json)

        self.assertEqual(len(ssd.class_nodes), 2)
        self.assertEqual(len(ssd.data_nodes), 4)
        self.assertEqual(len(ssd.mappings), 4)
        self.assertEqual(len(ssd.links), 5)  # class and data links
        self.assertEqual(len(ssd.data_links), 4)  # these are only data properties
        self.assertEqual(len(ssd.object_links), 1)  # these are only object properties

        res = self._ssds.compare(ssd, ssd, False, True)
        print(res)

        self.assertEqual(res['precision'], 1)
        self.assertEqual(res['recall'], 1)
        self.assertEqual(res['jaccard'], 1)

    def test_evaluate_paintings(self):
        """
        Here we have a class node with no data nodes
        :return:
        """
        self._datasets.upload(self._paintings_file)
        on = self._ontologies.upload(self._objects_owl)

        dataset = self._datasets.items[0]
        #print(dataset)
        assert (issubclass(type(dataset), DataSet))

        ontology = self._ontologies.items[0]
        #print("namespaces: ", ontology._prefixes)
        #print("class nodes: ", list(ontology._iclass_nodes()))
        #print("data nodes: ", list(ontology._idata_nodes()))
        #print("links: ", list(ontology._ilinks()))

        new_json = dataset.bind_ssd(self._paintings_ssd, [ontology], str(ontology._prefixes['']))

        # print("************************")
        # print("new json...")
        # pprint(new_json)

        empty_ssd = SSD(dataset, on)
        ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)
        # pprint(ssd.json)

        self.assertEqual(len(ssd.class_nodes), 3)
        self.assertEqual(len(ssd.links), 4)
        self.assertEqual(len(ssd.data_links), 2)
        self.assertEqual(len(ssd.object_links), 2)
        self.assertEqual(len(ssd.data_nodes), 2)
        self.assertEqual(len(ssd.mappings), 2)

        res = self._ssds.compare(ssd, ssd, True, True)
        print(res)

        self.assertEqual(res['precision'], 1)
        self.assertEqual(res['recall'], 1)
        self.assertEqual(res['jaccard'], 1)

    def test_evaluate_museum(self):
        """
        Here we have a class node with no data nodes, a list of ontologies, class instance link.
        Not all columns from file get mapped.
        :return:
        """
        dataset = self._datasets.upload(self._museum_file)

        ontologies = []
        for path in os.listdir(self._museum_owl_dir):
            f = os.path.join(self._museum_owl_dir, path)
            ontologies.append(self._ontologies.upload(f))

        new_json = dataset.bind_ssd(self._museum_ssd, ontologies, KARMA_DEFAULT_NS)

        empty_ssd = SSD(dataset, ontologies)
        ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)

        self.assertEqual(len(ssd.class_nodes), 6)
        self.assertEqual(len(ssd.links), 14)  # class instance, data property, object property
        self.assertEqual(len(ssd.data_nodes), 10)
        self.assertEqual(len(ssd.mappings), 10)

        res = self._ssds.compare(ssd, ssd, False, False)
        print(res)

        self.assertEqual(res['precision'], 1)
        self.assertEqual(res['recall'], 1)
        self.assertEqual(res['jaccard'], 1)