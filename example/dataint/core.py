import logging
import networkx as nx

_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)


class SemanticModeller(object):
    """
        SemanticModeller

    """
    def __init__(self, config=None, ontologies=None):
        """

        :param config: Arguments to configure the Schema Modeller
        :param ontologies: List of ontologies to use for mapping
        """
        # set the config args...
        default_args = {
            "kmeans": 5,
            "threshold": 0.7,
            "search-depth": 100
        }
        self.config = default_args

        # fill in the default args...
        if type(config) == dict:
            for k, v in config.items():
                if k in default_args.keys():
                    self.config[k] = v
                else:
                    _logger.warn("The key {} is not supported for config".format(k))

        # add the ontology list...
        self.ontologies = []
        if ontologies is not None:
            for o in ontologies:
                self.ontologies.append(Ontology(o))

        _logger.debug("Config set to:", self.config)
        _logger.debug("Ontologies set to:", self.ontologies)


class Ontology(object):
    """
        Ontology object

    """
    def __init__(self, file=None):
        """

        :param file:
        """
        if file is not None:
            _logger.debug("Importing {} from file.".format(file))
        else:
            self._prefixes = {
                "xml": "xml:/some/xml/resource",
                "uri": "uri:/some/uri/resource",
                "owl": "owl:/some/owl/resource"
            }

            self._uri = ""
            self._graph = nx.DiGraph()
            self._class_table = {}

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

    def class_node(self, name, nodes=None, prefix=None, parent=None):
        """

        :param name:
        :param nodes:
        :param prefix:
        :param parent:
        :return: The ontology object
        """
        _logger.debug("Adding class node name={}, "
                      "nodes={}, prefix={}, parent={}"
                      .format(name, nodes, prefix, parent))

        cn = ClassNode(name, nodes, prefix, parent)

        # if the name is in the class table, then we
        # need to remove it from the graph...
        if name in self._class_table:
            self._graph.remove_node(self._class_table[name])

        self._class_table[name] = cn
        self._graph.add_node(cn)

        return self

    def relationship(self, source, link, dest):
        """

        :param source:
        :param link:
        :param dest:
        :return: The ontology object
        """
        _logger.debug("Adding relationship between "
                      "classes {} and {} as '{}'"
                      .format(source, link, dest))

        if (source not in self._class_table) or (dest not in self._class_table):
            raise Exception("Item {} is not in the class nodes")

        class_source = self._class_table[source]
        class_dest = self._class_table[dest]

        self._graph.add_edge(
            class_source,
            class_dest,
            {"relationship": Link(link)}
        )

        return self


class ClassNode(object):
    """

    """
    def __init__(self, name, nodes=None, prefix=None, parent=None):
        """

        :param name:
        :param nodes:
        :param prefix:
        :param parent:
        """
        self.name = name
        self.nodes = [DataNode(n) for n in nodes] if nodes is not None else []
        self.prefix = prefix
        self.parent = parent

    def __str__(self):
        nodes = [n.name for n in self.nodes]
        return "ClassNode({}, {})".format(self.name, ", ".join(nodes))


class DataNode(object):
    """

    """
    def __init__(self, name):
        """

        :param name:
        """
        self.name = name

    def __str__(self):
        return "DataNode({})".format(self.name)


class Link(object):
    """

    """
    def __init__(self, name):
        """

        :param name:
        """
        self.name = name

    def __str__(self):
        return "Link({})".format(self.name)
