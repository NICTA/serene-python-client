"""
 License...
"""
import logging
import networkx as nx
import random
import string


_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)


class SemanticBase(object):
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
        self._LINK = "relationship"

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

        # if the name is in the class table, then we
        # need to remove it from the graph...
        if name in self._class_table:
            self._graph.remove_node(self._class_table[name])

        self._class_table[name] = cn
        self._graph.add_node(cn)

        return self

    def relationship(self, source, link, dest):
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

        if (source not in self._class_table) or (dest not in self._class_table):
            raise Exception("Item {} is not in the class nodes")

        src = self._class_table[source]
        dst = self._class_table[dest]

        self._graph.add_edge(
            src,
            dst,
            {self._LINK: Link(link, src, dst)})

        return self

    @staticmethod
    def flatten(xs):
        return [x for y in xs for x in y]

    @property
    def class_nodes(self):
        """The class node objects in the graph"""
        return self._class_table.values()

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
        links = nx.get_edge_attributes(self._graph, self._LINK)
        return list(links.values())


class Ontology(SemanticBase):
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
        else:
            # default prefixes...
            self._prefixes = {
                "xml": "xml:/some/xml/resource",
                "uri": "uri:/some/uri/resource",
                "owl": "owl:/some/owl/resource"
            }

    def load(self, filename):
        """
        Loads an Ontology from a file. The file must be in .owl RDF format.

        WARNING: What do we do about the same prefix values!!! unclear.
                Should they be merged? Should ClassNodes simply be referred to
                by the prefix under the hood?

        :param filename: The name of the .owl file.
        :return:
        """
        def rand_str(N=5):
            chars = string.ascii_uppercase + string.digits
            return ''.join(random.SystemRandom().choice(chars) for _ in range(N))

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


class ClassNode(object):
    """
        ClassNode objects hold the 'types' of the ontology or semantic model.
        A ClassNode can have multiple DataNodes. A DataNode corresponds to an
        attribute or a column in a dataset.
        A ClassNode can link to another ClassNode via a Link.
    """
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
        self.nodes = [DataNode(n, self) for n in nodes.keys()] if nodes is not None else []

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


class DataNode(object):
    """
        A DataNode is an attribute of a ClassNode. This can correspond to a
        column in a dataset.
    """
    def __init__(self, name, parent):
        """
        A DataNode is initialized with name and a parent ClassNode object...

        :param name: The name of the DataNode or data column
        :param parent: The parent ClassNode object...
        """
        self.name = name
        self.parent = parent
        self.prefix = parent.prefix

    def __repr__(self):
        return "DataNode({}, ClassNode({}))".format(self.name, self.parent.name)

    def __eq__(self, other):
        return (self.name == other.name) and (self.parent == other.parent)

    def __hash__(self):
        return id(self)


class Link(object):
    """
        A Link is a relationship between ClassNodes.
    """
    def __init__(self, name, src, dst):
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

    def __repr__(self):
        return "ClassNode({}) -> Link({}) -> ClassNode({})" \
            .format(self.src.name, self.name, self.dst.name)
