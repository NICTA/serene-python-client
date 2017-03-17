"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Defines the Ontology object
"""
import logging
import networkx as nx
import itertools as it
from serene.elements import Class, DataProperty, ObjectProperty, ObjectPropertyList
from serene.visualizers import BaseVisualizer
from collections import defaultdict

_logger = logging.getLogger()
_logger.setLevel(logging.WARN)

LINK_NAME = "relationship"

# ClassNode -> Class
# Link -> ObjectProperty
# DataNode -> DataProperty

class BaseSemantic(object):
    """
    Semantic Graph is the base object to construct Semantic objects.
    The object allows Class Nodes, DataNodes and Links to be members.
    """
    def __init__(self):
        """

        :param file:
        """
        self._uri = ""
        # we use multidigraph since we can potentially have more than 1 link between 2 nodes
        self._graph = nx.MultiDiGraph()
        self._class_table = {}
        self._links = ObjectPropertyList()

    def owl_class(self, name, nodes=None, prefix=None, is_a=None):
        """
        This function creates a ClassNode on the semantic object.

        :param name:
        :param nodes:
        :param prefix:
        :param is_a: The parent object
        :return: The ontology object
        """
        _logger.debug("Adding class node name={}, "
                      "nodes={}, prefix={}, parent={}"
                      .format(name, nodes, prefix, is_a))

        # set the prefix to the uri if not available...
        if prefix is None:
            prefix = self._uri

        # if the parent does not exist, refuse to create the class
        parent_class = None
        if is_a is not None:
            if is_a not in self._class_table:
                raise Exception("Failed to create class {}. "
                                "Parent class '{}' does not exist"
                                .format(name, is_a))
            else:
                parent_class = self._class_table[is_a]

        # build up the node list...
        if nodes is None:
            node_dict = {}
        elif issubclass(type(nodes), dict):
            node_dict = nodes
        elif issubclass(type(nodes), list):
            assert(len(nodes) == len(set(nodes)))
            node_dict = {k: str for k in nodes}
        else:
            msg = "Unknown nodes argument: {}".format(nodes)
            raise Exception(msg)

        # now we create a class node object...
        cn = Class(name, node_dict, prefix, parent_class)

        self.add_class_node(cn)

        return self

    def add_class_node(self, node, add_data_nodes=True):
        """
        Adds the ClassNode node into the Semantic Model

        :param node:
        :param add_data_nodes:
        :return:
        """
        # if the name is in the class table, then we
        # need to remove it from the graph...
        self._class_table[node.name] = node

        # now add the data property links
        if add_data_nodes and node.nodes is not None:
            for dataNode in node.nodes:
                link = ObjectProperty(dataNode.name, node, dataNode, prefix=dataNode.prefix)
                self.add_link(link)

        self._graph.add_node(node)

        return self

    def link(self, source, link, dest, prefix=None):
        """
        This function adds a Link relationship between two class nodes
        in the SemanticBase object. The ClassNode source and dest must
        already exist in the SemanticBase graph. This method will create
        the Link object.

        :param source: The source ClassNode
        :param link: The name of the Link relationship
        :param dest: The destination ClassNode object
        :param prefix: The ontology namespace for the link
        :return: The ontology object
        """
        _logger.debug("Adding relationship between "
                      "classes {} and {} as '{}'"
                      .format(source, dest, link))

        if prefix is None:
            prefix = self._uri

        if source not in self._class_table:
            msg = "Item {} is not in the class nodes".format(source)
            raise Exception(msg)
        elif dest not in self._class_table:
            msg = "Item {} is not in the class nodes".format(dest)
            raise Exception(msg)

        src = self._class_table[source]
        dst = self._class_table[dest]
        link = ObjectProperty(link, src, dst, prefix)

        self.add_link(link)

        return self

    def add_link(self, link):
        """

        :return:
        """
        if link not in self._links:
            self._links.append(link)

            # we also add the link to the graph...
            self._graph.add_edge(
                link.src,
                link.dst,
                link)
        return self

    def find_class_node(self, name):
        """

        :param name:
        :return:
        """
        ct = self._class_table
        if name in ct:
            return ct[name]
        else:
            return None

    def remove_link(self, link):
        """
        Removes a link from the graph...
        :param link:
        :return:
        """
        _logger.debug("Attempting to remove link: {}".format(link))
        target = ObjectProperty.search(self.links, link)
        if target:
            _logger.debug("Found true link: {}".format(target))
            try:
                # we need to be careful, because if the nodes
                # are destroyed, then the edge will not exist!
                self._graph.remove_edge(target.src, target.dst)
                _logger.debug("Edge removed successfully {}-{}".format(target.src, target.dst))
            except Exception:
                _logger.debug("Missing edge, nodes already removed. {}".format(link))
                pass

            self._links.remove(target)
            _logger.debug("Removed link: {}".format(target))
        else:
            msg = "Item {} does not exist in the model.".format(link)
            _logger.error(msg)
        return

    def _parent_chain(self, node):
        """
        Returns the chain of parent classes from ClassNode -> all parents
        including the original
        """
        if node is None:
            raise StopIteration
        else:
            yield node
            yield from self._parent_chain(node.parent)

    def _item_chain(self, z):
        """Returns all data nodes in the class chain of z"""
        for cls in self._parent_chain(z):
            for item in cls.nodes:
                yield item.name

    def _iclass_nodes(self):
        """Returns all inferred class nodes"""
        for node in self.class_nodes:
            yield Class(node.name,
                        list(self._item_chain(node)),
                        prefix=node.prefix)

    def _idata_nodes(self):
        """Returns all inferred data nodes"""
        for node in self.iclass_nodes:
            for dn in node.nodes:
                yield dn

    @property
    def iclass_nodes(self):
        return list(set(self._iclass_nodes()))

    @property
    def idata_nodes(self):
        return list(set(self._idata_nodes()))

    def _child_map(self):
        """Builds up a map of the child chains"""
        m = defaultdict(set)
        for c in self.class_nodes:
            for parent in self._parent_chain(c):
                m[parent].add(c)
        return m

    def _child_chain(self, z):
        """Returns the item z with all descendants"""
        m = self._child_map()
        if z in m:
            for y in m[z]:
                yield y
        else:
            yield z

    def _ilinks(self):
        """Returns the interpreted links..."""
        for link in self.links:
            ilinks = it.product(
                self._child_chain(link.src),
                self._child_chain(link.dst)
            )
            for src, dst in ilinks:
                yield ObjectProperty(link.name, src, dst, prefix=link.prefix)

    @property
    def ilinks(self):
        return list(self._ilinks())

    @staticmethod
    def _is_node(value):
        """
        Determines whether the value is a ClassNode or a DataNode

        :param value:
        :return:
        """
        return issubclass(type(value), Class) or issubclass(type(value), DataProperty)

    @staticmethod
    def _is_link(value):
        """
        Determines whether the value is a Link

        :param value:
        :return:
        """
        return issubclass(type(value), ObjectProperty)

    def remove_node(self, node):
        """
        Removes a node from the graph...
        :param node: ClassNode to remove
        :return:
        """
        _logger.debug("remove_node called with {}".format(node))
        if not self._is_node(node):
            msg = "{} is not a ClassNode or DataNode".format(node)
            raise Exception(msg)

        elif issubclass(type(node), Class):
            _logger.debug("Removing ClassNode: {}".format(node))
            # first we find the actual reference...
            true_node = self._class_table[node.name]

            _logger.debug("Actual node found: {}".format(true_node))

            # recursively remove the DataNodes...
            for n in true_node.nodes:
                self.remove_node(n)

            # next we remove the links pointing to that node.
            # we need to be careful when iterating as we are deleting
            # the list as we go!!
            for link in self.links[::-1]:
                # print()
                # print(">>>>>>>>>> {} <<<<<<<<<<<".format(link.name))
                # print()
                # print(">>> Searching link: {}".format(link))
                # print(">>>   which has {}".format(link.dst))
                # print(">>>   which has {}".format(link.src))
                # print(">>>   comparing {}".format(true_node))
                # print(">>>        with {}".format(link.dst))
                # print(">>>         and {}".format(link.src))
                if link.dst == true_node or link.src == true_node:
                    _logger.debug("Link found, removing {}".format(link))
                    self.remove_link(link)

            # remove the ClassNode from the class table
            del self._class_table[true_node.name]
        else:
            _logger.debug("Removing DataNode: {}".format(node))

            # first we find the actual reference...
            true_node = DataProperty.search(self.data_nodes, node)

            _logger.debug("Actual node found: {}".format(true_node))

        self._graph.remove_node(true_node)

        del true_node

        return

    @staticmethod
    def flatten(xs):
        return [x for y in xs for x in y]

    @property
    def class_nodes(self):
        """The class node objects in the graph"""
        return list(self._class_table.values())

    @property
    def data_nodes(self):
        """All of the data node objects in the graph"""
        # we grab the data nodes from the class nodes...
        # make sure we don't return any duplicates...
        data_nodes = set(self.flatten(cn.nodes for cn in self.class_nodes))

        return list(data_nodes)

    @property
    def links(self):
        """Returns all the links in the graph"""
        return self._links

    @property
    def class_links(self):
        """Returns all the links in the graph"""
        # links = nx.get_edge_attributes(self._graph, self._LINK)
        # return list(links.values())
        def is_class(link):
            return issubclass(type(link), Class)

        return [link for link in self._links if is_class(link.src) and is_class(link.dst)]

    @property
    def data_links(self):
        """Returns all the data links in the graph"""
        def is_class(link):
            return issubclass(type(link), Class)

        def is_data(link):
            return issubclass(type(link), DataProperty)

        return [link for link in self._links
                if is_class(link.src) and is_data(link.dst)]

    def get(self, value):
        """
        Returns a reference to a partial object e.g. ontology.get(ClassNode("Person"))
        :param value:
        :return:
        """
        if issubclass(type(value), Class):
            return Class.search(self.class_nodes, value)
        elif issubclass(type(value), DataProperty):
            return DataProperty.search(self.data_nodes, value)
        elif issubclass(type(value), ObjectProperty):
            return ObjectProperty.search(self.links, value)
        else:
            msg = "{} not supported for get method. Use ClassNode, DataNode or Link"
            raise ValueError(msg)

    def summary(self):
        """Prints a summary of the ontology"""
        print("Class Nodes:")
        for cls in self.class_nodes:
            print("\t", cls)
        print()

        print("Data Nodes:")
        for dn in self.data_nodes:
            print("\t", dn)
        print()

        print("Links:")
        for link in self.links:
            print("\t", link)
        print()

    def show(self):
        BaseVisualizer(self).show()
        return

