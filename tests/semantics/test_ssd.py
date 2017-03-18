"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the ssd module
"""
import os

from serene.elements import SSD, Column, DataNode, ClassNode
from serene.endpoints import DataSetEndpoint, OntologyEndpoint
from ..utils import TestWithServer


class TestSSD(TestWithServer):
    """
    Tests the SSD class
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

    def test_create(self):
        """
        Tests the SSD creation
        :return:
        """
        self._build_simple()

    def _build_simple(self):

        ds = self._datasets.upload(self._test_file)
        on = self._ontologies.upload(self._test_owl)

        self.assertEqual(len(self._ontologies.items), 1)

        single = SSD(dataset=ds, ontology=on)

        self.assertEqual(len(single.data_nodes), 0)
        self.assertEqual(len(single.links), 0)
        self.assertEqual(len(single.columns), 4)
        self.assertEqual(str(single.columns),
                         "[Column(company), Column(ceo), Column(city), Column(state)]")

        return single

    def test_map_simple(self):
        """
        Tests the map function for SSD mapping with one map
        :return:
        """
        simple = self._build_simple()

        simple.map(Column("ceo"), DataNode("Person", "name"))

        self.assertEqual(len(simple.class_nodes), 1)
        self.assertEqual(len(simple.data_nodes), 1)
        self.assertEqual(len(simple.data_links), 1)
        self.assertEqual(len(simple.object_links), 0)

    def test_map_full(self):
        """
        Tests the map function for SSD mapping with full map
        :return:
        """
        simple = self._build_simple()

        (simple
         .map(Column("company"), DataNode("Organization", "name"))
         .map(Column("ceo"), DataNode("Person", "name"))
         .map(Column("city"), DataNode("City", "name"))
         .map(Column("state"), DataNode("State", "name")))

        self.assertEqual(len(simple.class_nodes), 4)
        self.assertEqual(len(simple.data_nodes), 4)
        self.assertEqual(len(simple.data_links), 4)
        self.assertEqual(len(simple.object_links), 0)

    def test_map_short_hand(self):
        """
        Tests the map function for SSD mapping with short hand strings
        :return:
        """
        simple = self._build_simple()

        (simple
         .map("company", "Organization.name")
         .map("ceo", "Person.name")
         .map("city", "City.name")
         .map("state", "State.name"))

        self.assertEqual(len(simple.class_nodes), 4)
        self.assertEqual(len(simple.data_nodes), 4)
        self.assertEqual(len(simple.data_links), 4)
        self.assertEqual(len(simple.object_links), 0)

    def test_map_multi_instance(self):
        """
        Tests the map function with multiple instances

        :return:
        """
        simple = self._build_simple()

        (simple
         .map(Column("company"),
              DataNode(ClassNode("Person", 0), "name"))
         .map(Column("ceo"),
              DataNode(ClassNode("Person", 1), "name"))
         .map(Column("city"),
              DataNode(ClassNode("Person", 2), "name"))
         .map(Column("state"),
              DataNode(ClassNode("Person", 3), "name")))

        self.assertEqual(len(simple.class_nodes), 4)
        self.assertEqual(len(simple.data_nodes), 4)
        self.assertEqual(len(simple.data_links), 4)
        self.assertEqual(len(simple.object_links), 0)

    def test_map_multi_instance_links(self):
        """
        Tests the map function with multiple instances in the links

        :return:
        """
        simple = self._build_simple()

        (simple
         .map(Column("company"),
              DataNode(ClassNode("City", 0), "name"))
         .map(Column("ceo"),
              DataNode(ClassNode("City", 1), "name"))
         .map(Column("city"),
              DataNode(ClassNode("City", 2), "name"))
         .map(Column("state"),
              DataNode(ClassNode("City", 3), "name"))
         .link(ClassNode("City", 0), "nearby", ClassNode("City", 1))
         .link(ClassNode("City", 1), "nearby", ClassNode("City", 2))
         .link(ClassNode("City", 2), "nearby", ClassNode("City", 3)))

        self.assertEqual(len(simple.class_nodes), 4)
        self.assertEqual(len(simple.data_nodes), 4)
        self.assertEqual(len(simple.data_links), 4)
        self.assertEqual(len(simple.object_links), 3)

    def test_map_overwrite(self):
        """
        Tests that re-specifying should overwrite
        :return:
        """
        simple = self._build_simple()

        (simple
         .map("company", "Organization.name")
         .map("ceo", "Person.name")
         .map("city", "City.name")
         .map("state", "State.name")
         .map("company", "Organization.name")
         .map("ceo", "Person.name")
         .map("city", "City.name")
         .map("state", "State.name")
         .link("City", "state", "State")
         .link("Organization", "location", "City")
         .link("Person", "worksFor", "Organization")
         .link("City", "state", "State")
         .link("Organization", "location", "City")
         .link("Person", "worksFor", "Organization"))

        self.assertEqual(len(simple.class_nodes), 4)
        self.assertEqual(len(simple.data_nodes), 4)
        self.assertEqual(len(simple.data_links), 4)
        self.assertEqual(len(simple.object_links), 3)

    def test_map_links(self):
        """
        Tests the map function for SSD mapping with full map
        :return:
        """
        simple = self._build_simple()

        (simple
         .map("company", "Organization.name")
         .map("ceo", "Person.name")
         .map("city", "City.name")
         .map("state", "State.name")
         .link("City", "state", "State")
         .link("Organization", "location", "City")
         .link("Person", "worksFor", "Organization"))

        self.assertEqual(len(simple.class_nodes), 4)
        self.assertEqual(len(simple.data_nodes), 4)
        self.assertEqual(len(simple.data_links), 4)
        self.assertEqual(len(simple.object_links), 3)

