"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the core module
"""
import unittest2 as unittest
import serene
from serene import ClassNode, DataNode, Link
from serene.semantics import DATA_NODE_LINK_NAME


class TestOntology(unittest.TestCase):
    """
    Tests the Ontology class
    """
    @staticmethod
    def empty_ontology():
        return serene.Ontology()

    def test_class_node(self):
        """
        Tests the class node add method
        :return:
        """
        single = self.empty_ontology().class_node("hello")

        self.assertEqual(len(single.class_nodes), 1)
        self.assertEqual(len(single.data_nodes), 0)
        self.assertEqual(len(single.links), 0)
        self.assertIsNotNone(ClassNode.search(single.class_nodes, ClassNode("hello")))
        self.assertEqual(single.class_nodes[0].name, "hello")

    def test_class_node_one_prop(self):
        """
        Tests adding a class node with one property
        :return:
        """
        one_prop = self.empty_ontology().class_node("Person", ["name"])

        self.assertEqual(len(one_prop.class_nodes), 1)
        self.assertEqual(len(one_prop.data_nodes), 1)
        self.assertEqual(len(one_prop.links), 1)
        self.assertIsNotNone(
            ClassNode.search(one_prop.class_nodes, ClassNode("Person")))
        self.assertIsNotNone(
            ClassNode.search(one_prop.class_nodes, ClassNode("Person", ["name"])))
        self.assertIsNotNone(
            DataNode.search(one_prop.data_nodes, DataNode("Person", "name")))
        self.assertEqual(one_prop.class_nodes[0].name, "Person")
        self.assertEqual(one_prop.class_nodes[0].nodes[0].name, "name")

    def test_class_node_multi_prop(self):
        """
        Tests adding a class node with multiple properties
        :return:
        """
        two_prop = self.empty_ontology().class_node("Person", ["name", "addr"])

        self.assertEqual(len(two_prop.class_nodes), 1)
        self.assertEqual(len(two_prop.data_nodes), 2)
        self.assertEqual(len(two_prop.links), 2)
        self.assertIsNotNone(
            ClassNode.search(two_prop.class_nodes, ClassNode("Person")))
        self.assertIsNotNone(
            ClassNode.search(two_prop.class_nodes, ClassNode("Person", ["name"])))
        self.assertIsNotNone(
            ClassNode.search(two_prop.class_nodes, ClassNode("Person", ["addr"])))
        self.assertIsNotNone(
            ClassNode.search(two_prop.class_nodes, ClassNode("Person", ["name", "addr"])))
        self.assertIsNotNone(
            DataNode.search(two_prop.data_nodes, DataNode("Person", "name")))
        self.assertIsNotNone(
            DataNode.search(two_prop.data_nodes, DataNode("Person", "addr")))
        self.assertEqual(two_prop.class_nodes[0].name, "Person")
        self.assertEqual(
            set([x.name for x in two_prop.class_nodes[0].nodes]),
            {"name", "addr"}
        )

    def test_class_node_duplicate_prop(self):
        """
        Create class_node should fail with duplicate properties
        :return:
        """
        with self.assertRaises(Exception):
            self.empty_ontology().class_node("Person", ["name", "name"])

    def test_class_node_no_parent(self):
        """
        Create class_node should fail if parent does not exist
        :return:
        """
        with self.assertRaises(Exception):
            self.empty_ontology().class_node("Person", ["name", "addr"], is_a="junk")

    def test_class_node_bad_nodes(self):
        """
        Create class_node should fail with bad nodes
        :return:
        """
        with self.assertRaises(Exception):
            self.empty_ontology().class_node("Person", nodes=1234)

    def test_class_node_parent(self):
        """
        Create class_node should create the parent link successfully
        :return:
        """
        family = (self.empty_ontology()
                  .class_node("Parent", ["name", "addr"])
                  .class_node("Child", ["toys"], is_a="Parent"))

        self.assertEqual(len(family.class_nodes), 2)
        self.assertEqual(len(family.data_nodes), 3)
        self.assertEqual(len(family.links), 3)

        parent = ClassNode.search(family.class_nodes, ClassNode("Parent"))
        child = ClassNode.search(family.class_nodes, ClassNode("Child"))
        links = [link.name for link in family.links]

        self.assertEqual(child.parent, parent)
        self.assertEqual(set(links), {DATA_NODE_LINK_NAME})

    def test_add_class_node(self):
        """
        Tests the class node add ClassNode method
        :return:
        """
        node = ClassNode("Person")
        single = self.empty_ontology().add_class_node(node)

        self.assertEqual(len(single.class_nodes), 1)
        self.assertEqual(ClassNode.search(single.class_nodes, ClassNode("Person")), node)

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

