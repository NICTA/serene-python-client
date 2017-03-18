"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the core module
"""
import unittest2 as unittest
import os

import serene
from serene.elements import Class, DataProperty


class TestOntology(unittest.TestCase):
    """
    Tests the Ontology class
    """
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
        self._test_owl = os.path.join(path, 'owl', 'dataintegration_report_ontology.ttl')

    @staticmethod
    def empty_ontology():
        return serene.Ontology()

    def test_class_node(self):
        """
        Tests the class node add method
        :return:
        """
        single = self.empty_ontology().owl_class("hello")

        self.assertEqual(len(single.class_nodes), 1)
        self.assertEqual(len(single.data_nodes), 0)
        self.assertEqual(len(single.links), 0)
        self.assertIsNotNone(Class.search(single.class_nodes, Class("hello")))
        self.assertEqual(single.class_nodes[0].label, "hello")

    def test_class_node_one_prop(self):
        """
        Tests adding a class node with one property
        :return:
        """
        one_prop = self.empty_ontology().owl_class("Person", ["name"])

        self.assertEqual(len(one_prop.class_nodes), 1)
        self.assertEqual(len(one_prop.data_nodes), 1)
        self.assertEqual(len(one_prop.links), 1)
        self.assertIsNotNone(
            Class.search(one_prop.class_nodes, Class("Person")))
        self.assertIsNotNone(
            Class.search(one_prop.class_nodes, Class("Person", ["name"])))
        self.assertIsNotNone(
            DataProperty.search(one_prop.data_nodes, DataProperty("Person", "name")))
        self.assertEqual(one_prop.class_nodes[0].label, "Person")
        self.assertEqual(one_prop.class_nodes[0].nodes[0].label, "name")

    def test_class_node_multi_prop(self):
        """
        Tests adding a class node with multiple properties
        :return:
        """
        two_prop = self.empty_ontology().owl_class("Person", ["name", "addr"])

        self.assertEqual(len(two_prop.class_nodes), 1)
        self.assertEqual(len(two_prop.data_nodes), 2)
        self.assertEqual(len(two_prop.links), 2)
        self.assertIsNotNone(
            Class.search(two_prop.class_nodes, Class("Person")))
        self.assertIsNotNone(
            Class.search(two_prop.class_nodes, Class("Person", ["name"])))
        self.assertIsNotNone(
            Class.search(two_prop.class_nodes, Class("Person", ["addr"])))
        self.assertIsNotNone(
            Class.search(two_prop.class_nodes, Class("Person", ["name", "addr"])))
        self.assertIsNotNone(
            DataProperty.search(two_prop.data_nodes, DataProperty("Person", "name")))
        self.assertIsNotNone(
            DataProperty.search(two_prop.data_nodes, DataProperty("Person", "addr")))
        self.assertEqual(two_prop.class_nodes[0].label, "Person")
        self.assertEqual(
            set([x.label for x in two_prop.class_nodes[0].nodes]),
            {"name", "addr"}
        )

    def test_class_node_duplicate_prop(self):
        """
        Create class_node should fail with duplicate properties
        :return:
        """
        with self.assertRaises(Exception):
            self.empty_ontology().owl_class("Person", ["name", "name"])

    def test_class_node_no_parent(self):
        """
        Create class_node should fail if parent does not exist
        :return:
        """
        with self.assertRaises(Exception):
            self.empty_ontology().owl_class("Person", ["name", "addr"], is_a="junk")

    def test_class_node_bad_nodes(self):
        """
        Create class_node should fail with bad nodes
        :return:
        """
        with self.assertRaises(Exception):
            self.empty_ontology().owl_class("Person", nodes=1234)

    def test_class_node_parent(self):
        """
        Create class_node should create the parent link successfully
        :return:
        """
        family = (self.empty_ontology()
                  .owl_class("Parent", ["name", "addr"])
                  .owl_class("Child", ["toys"], is_a="Parent"))

        self.assertEqual(len(family.class_nodes), 2)
        self.assertEqual(len(family.data_nodes), 3)
        self.assertEqual(len(family.links), 3)

        parent = Class.search(family.class_nodes, Class("Parent"))
        child = Class.search(family.class_nodes, Class("Child"))
        links = [link.label for link in family.links]

        self.assertEqual(child.parent, parent)
        self.assertEqual(set(links), {"name", "addr", "toys"})

    def test_add_class_node(self):
        """
        Tests the class node add ClassNode method
        :return:
        """
        node = Class("Person")
        single = self.empty_ontology().add_class_node(node)

        self.assertEqual(len(single.class_nodes), 1)
        self.assertEqual(Class.search(single.class_nodes, Class("Person")), node)

    def test_link(self):
        raise NotImplementedError("Test not implemented")

    def test_add_link(self):
        raise NotImplementedError("Test not implemented")

    def test_find_class_node(self):
        raise NotImplementedError("Test not implemented")

    def test_remove_link(self):
        raise NotImplementedError("Test not implemented")

    def test_is_node(self):
        raise NotImplementedError("Test not implemented")

    def test_is_link(self):
        raise NotImplementedError("Test not implemented")

    def test_remove_node(self):
        raise NotImplementedError("Test not implemented")

    def test_flatten(self):
        raise NotImplementedError("Test not implemented")

    def test_class_nodes(self):
        raise NotImplementedError("Test not implemented")

    def test_data_nodes(self):
        raise NotImplementedError("Test not implemented")

    def test_links(self):
        raise NotImplementedError("Test not implemented")

    def test_class_links(self):
        raise NotImplementedError("Test not implemented")

    def test_summary(self):
        raise NotImplementedError("Test not implemented")

    def test_create(self):
        raise NotImplementedError("Test not implemented")

    def test_update(self):
        raise NotImplementedError("Test not implemented")

    def test_set_stored_flag(self):
        raise NotImplementedError("Test not implemented")

    def test_to_turtle(self):
        raise NotImplementedError("Test not implemented")

    def test_prefix(self):
        raise NotImplementedError("Test not implemented")

    def test_uri(self):
        raise NotImplementedError("Test not implemented")

    def test_prefixes(self):
        raise NotImplementedError("Test not implemented")

    def test_repr(self):
        raise NotImplementedError("Test not implemented")

    def test_iclass_nodes(self):
        raise NotImplementedError("Test not implemented")

    def test_idata_nodes(self):
        raise NotImplementedError("Test not implemented")

    def test_ilinks(self):
        raise NotImplementedError("Test not implemented")
