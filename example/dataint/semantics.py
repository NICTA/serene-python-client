"""
 License...
"""
import logging
import networkx as nx
import random
import string
import tempfile
import os.path
import collections

from .utils import Searchable

_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)


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
        self._graph = nx.DiGraph()
        self._class_table = {}
        self._links = LinkList()
        self._LINK = "relationship"

    # def add(self, item):
    #     """
    #     Adds a ClassNode or a Link to the model
    #
    #     :param item:
    #     :return:
    #     """
    #     if type(item) == ClassNode:
    #         self.class_node(
    #             item.name,
    #             item.nodes,
    #             item.prefix,
    #             item.parent.name if item.parent else None
    #         )
    #     elif type(item) == Link:
    #         self.link(item.src, item.name, item.dst)

    def class_node(self, name, nodes=None, prefix=None, is_a=None):
        """
        This function creates a ClassNode on the semantic object.

        :param name:
        :param nodes:
        :param prefix:
        :param is_a: The parent object
        :return: The ontology object
        """
        _logger.debug("Adding class node name={}, "
                      "nodes={}, prefix={}, is_a={}"
                      .format(name, nodes, prefix, is_a))

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
        node_dict = {}
        if type(nodes) == dict:
            node_dict = nodes
        elif type(nodes) == list:
            node_dict = {k: str for k in nodes}

        # now we create a class node object...
        cn = ClassNode(name, node_dict, prefix, parent_class)

        self.add_class_node(cn)

        return self

    def add_class_node(self, node):
        """
        Adds the ClassNode node into the Semantic Model

        :param node:
        :return:
        """
        # if the name is in the class table, then we
        # need to remove it from the graph...
        #if node.name in self._class_table:
        #    print("Class node '{}' is in the table".format(node.name))
        #    self._graph.remove_node(self._class_table[node.name])

        self._class_table[node.name] = node

        self._graph.add_node(node)

    def link(self, source, link, dest):
        """
        This function adds a Link relationship between two class nodes
        in the SemanticBase object. The ClassNode source and dest must
        already exist in the SemanticBase graph. This method will create
        the Link object.

        :param source: The source ClassNode
        :param link: The name of the Link relationship
        :param dest: The destination ClassNode object
        :return: The ontology object
        """
        _logger.debug("Adding relationship between "
                      "classes {} and {} as '{}'"
                      .format(source, link, dest))

        if source not in self._class_table:
            msg = "Item {} is not in the class nodes".format(source)
            raise Exception(msg)
        elif dest not in self._class_table:
            msg = "Item {} is not in the class nodes".format(dest)
            raise Exception(msg)

        src = self._class_table[source]
        dst = self._class_table[dest]
        link = Link(link, src, dst)

        self.add_link(link)

        return self

    def add_link(self, link):
        """

        :return:
        """
        self._links.append(link)

        # we also add the link to the graph...
        self._graph.add_edge(
            link.src,
            link.dst,
            {self._LINK: link})
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
        locate = [x for x in self.links if link == x]
        if len(locate):
            target = locate[0]
            self._graph.remove_edge(target.src, target.dst)
            self._links.remove(link)
        else:
            msg = "Item {} does not exist in the model.".format(link)
            _logger.error(msg)
        return

    def remove_node(self, node):
        """
        Removes a node from the graph...
        :param node: ClassNode to remove
        :return:
        """
        del self._class_table[node.name]
        self._graph.remove_node(node)
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
        # we extract the datanodes from the class nodes...
        cns = self.class_nodes
        # make sure we don't return any duplicates...
        data_nodes = set(self.flatten(cn.nodes for cn in cns))

        return list(data_nodes)

    @property
    def links(self):
        """Returns all the links in the graph"""
        # links = nx.get_edge_attributes(self._graph, self._LINK)
        # return list(links.values())
        return self._links


class Ontology(BaseSemantic):
    """
        The Ontology object holds an ontolgy. This can be specified from
        an OWL file, or built by hand using the SemanticBase construction
        methods. Note that this is a subset of the OWL specification, and
        only ClassNodes, DataNodes, relationships and subclass properties
        are used. All other features are ignored.

        This class extends the SemanticBase object to include Ontology
        specific elements such as loading from a file, prefixes and a
        URI value.

    """
    def __init__(self, file=None):
        """
        The constructor can take an additional file object, which
        will be read to build links and class nodes. Note that not
        all attributes from an OWL file will be imported.

        :param file: The name of the .owl file.
        """
        super().__init__()

        if file is not None:
            _logger.debug("Importing {} from file.".format(file))
            self.load(file)
            self.filename = file
        else:
            # default prefixes...
            self._prefixes = {
                "xml": "xml:/some/xml/resource",
                "uri": "uri:/some/uri/resource",
                "owl": "owl:/some/owl/resource"
            }
            self.filename = os.path.join(
                tempfile.gettempdir(),
                "{}.owl".format(random.randint(1e7, 1e8-1))
            )

    def load(self, filename):
        """
        Loads an Ontology from a file. The file must be in .owl RDF format.

        WARNING: What do we do about the same prefix values!!! unclear.
                Should they be merged? Should ClassNodes simply be referred to
                by the prefix under the hood?

        :param filename: The name of the .owl file.
        :return:
        """
        def rand_str(n=5):
            chars = string.ascii_uppercase + string.digits
            return ''.join(random.SystemRandom().choice(chars) for _ in range(n))

        _logger.info("Extracting ontology from file {}.".format(filename))

        # REMOVE!!! These are just junk values...
        self._prefixes = {
            "xml": "xml:/some/xml/resource",
            "uri": "uri:/some/uri/resource",
            "owl": "owl:/some/owl/resource"
        }
        self._uri = "junk"

        self.class_node(rand_str(), [rand_str(), rand_str()])

    def prefix(self, prefix, uri):
        """
        Adds a URI prefix to the ontology

        :param prefix:
        :param uri:
        :return: The ontology object
        """
        _logger.debug("Adding prefix with prefix={}, uri={}".format(prefix, uri))
        self._prefixes[prefix] = uri
        return self

    def uri(self, uri_string):
        """
        Adds the URI to the ontology
        :param uri_string:
        :return: The ontology object
        """
        _logger.debug("Adding URI string uri={}".format(uri_string))
        self._uri = uri_string
        return self


class ClassNode(Searchable):
    """
        ClassNode objects hold the 'types' of the ontology or semantic model.
        A ClassNode can have multiple DataNodes. A DataNode corresponds to an
        attribute or a column in a dataset.
        A ClassNode can link to another ClassNode via a Link.
    """
    # the search parameters...
    getters = [
        lambda node: node.name,
        lambda node: node.nodes if len(node.nodes) else None,
        lambda node: node.prefix if node.prefix else None,
        lambda node: node.parent if node.parent else None
    ]

    def __init__(self, name, nodes=None, prefix=None, parent=None):
        """
        A ClassNode is initialized with a name, a list of string nodes
        and optional prefix and/or a parent ClassNode

        :param name: The string name of the ClassNode
        :param nodes: A list of strings to initialze the DataNode objects
        :param prefix: The URI prefix
        :param parent: The parent object if applicable
        """
        self.name = name
        self.prefix = prefix
        self.parent = parent
        self.nodes = [DataNode(self, n) for n in nodes.keys()] if nodes is not None else []

    def ssd_output(self, ident):
        """
        The output function has not
        :return:
        """
        return {
            "id": ident,
            "label": "{}:{}".format(self.prefix, self.name),
            "type": "ClassNode"
        }

    def __repr__(self):
        nodes = [n.name for n in self.nodes]

        if self.parent is None:
            parent = ""
        else:
            parent = ", parent={}".format(self.parent.name)

        return "ClassNode({}, [{}]{})".format(self.name, ", ".join(nodes), parent)

    def __eq__(self, other):
        return (self.name == other.name) \
               and (self.prefix == other.prefix)

    def __hash__(self):
        return id(self)


class DataNode(Searchable):
    """
        A DataNode is an attribute of a ClassNode. This can correspond to a
        column in a dataset.
    """
    # the search parameters...
    getters = [
        lambda node: node.name,
        lambda node: node.parent.name if node.parent else None,
        lambda node: node.parent.prefix if node.parent else None
    ]

    def __init__(self, *names):
        """
        A DataNode is initialized with name and a parent ClassNode object.
        A DataNode can be initialized in the following ways:

        DataNode(ClassNode("Person"), "name")
        DataNode("Person", "name)
        DataNode("name")

        :param names: The name of the parent classnode and the name of the DataNode
        """
        if len(names) == 1:
            # initialized with DataNode("name") - for lookups only...
            self.name = names[0]
            self.parent = None

        elif len(names) == 2:
            # here the first element is now the parent...
            parent = names[0]

            if type(parent) == ClassNode:
                # initialized with DataNode(ClassNode("Person"), "name")
                self.parent = parent
            else:
                # initialized with DataNode("Person", "name")
                self.parent = ClassNode(parent)
            self.name = names[1]

        else:
            msg = "Insufficient args for DataNode construction."
            raise Exception(msg)

        super().__init__()

    def ssd_output(self, ident):
        """
        The output function has not
        :return:
        """
        if self.parent is None:
            label = self.name
        else:
            label = "{}.{}".format(self.parent.name, self.name)

        return {
            "id": ident,
            "label": label,
            "type": "ClassNode"
        }

    def __repr__(self):
        if self.parent:
            return "DataNode({}, {})".format(self.parent.name, self.name)
        else:
            return "DataNode({})".format(self.name)


class Link(Searchable):
    """
        A Link is a relationship between ClassNodes.
    """
    @staticmethod
    def node_match(node):
        if node.src is None:
            return None
        if node.dst is None:
            return None
        return node.src.name, node.dst.name

    # the search parameters...
    getters = [
        lambda node: node.name,
        node_match
    ]

    # special link names...
    _SUBCLASS = "subclass"

    def __init__(self, name, src=None, dst=None):
        """
        The link can be initialized with a name, and also
        holds the source and destination ClassNode references.

        :param name: The link name
        :param src: The source ClassNode
        :param dst: The destination ClassNode
        """
        self.name = name
        self.src = src
        self.dst = dst

    def ssd_output(self, index_map):
        """

        :return:
        """
        if type(self.src) == ClassNode:
            link_type = 'ObjectProperty'
        else:
            link_type = 'DataProperty'

        return {
            "source": index_map[self.src],
            "target": index_map[self.dst],
            "label": self.name,
            "type": link_type
        }

    def __repr__(self):
        if (self.src is None) or (self.dst is None):
            return "Link({})".format(self.name)
        else:
            return "ClassNode({}) -> Link({}) -> ClassNode({})" \
                .format(self.src.name, self.name, self.dst.name)


class LinkList(collections.MutableSequence):
    """
    Container type for Link objects in the Semantic Source Description
    """

    def __init__(self, *args):
        self.list = list()
        self.extend(list(args))

    @staticmethod
    def check(v):
        if not isinstance(v, Link):
            raise TypeError("Only Link types permitted")

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]

    def __delitem__(self, i):
        del self.list[i]

    def __setitem__(self, i, v):
        self.check(v)
        self.list[i] = v

    def insert(self, i, v):
        self.check(v)
        self.list.insert(i, v)

    def __repr__(self):
        return '\n'.join(str(link) for link in self.list)

