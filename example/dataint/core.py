"""
License...
"""
import logging
import pandas as pd

from .semantics import Ontology, DataNode, ClassNode, Link

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
        self._ontologies = []
        if ontologies is not None:
            for o in ontologies:
                self.add_ontology(Ontology(o))

        _logger.debug("Config set to:", self.config)
        _logger.debug("Ontologies set to:", self._ontologies)

    def add_ontology(self, ontology):
        """
        Adds an ontology to the list...
        :param ontology:
        :return:
        """
        if type(ontology) == Ontology:
            self._ontologies.append(ontology)
        elif type(ontology) == str:
            # attempt to open from a file...
            self.add_ontology(Ontology(ontology))
        else:
            raise Exception("Failed to add ontology: {}".format(ontology))

        return self

    def to_ssd(self, filename):
        """
        Loads in a datafile to transform to an ssd object
        :param filename:
        :return:
        """
        ssd = SemanticSourceDesc(filename, self)
        return ssd

    def _data_node_lookup(self, data_node):
        """
        Takes a DataNode object, and uses the name property to
        find the true match in the data node mapping. This allows
        users to use the short hand:

        DataNode("name")

        rather than

        DataNode("name", ClassNode("Parent"))

        :param data_node: A DataNode object
        :return:
        """
        dns = self.data_nodes

        matches = [dn for dn in dns if data_node.name == dn.name]

        if len(matches) == 0:
            # if not found throw error...
            msg = "Failed to find DataNode: " + data_node.name
            _logger.error(msg)
            raise Exception(msg)
        elif len(matches) == 1:
            return matches[0]
        else:
            # if ambiguities throw error
            msg = "DataNode name {} is ambiguous: {}".format(data_node.name, matches)
            _logger.error(msg)
            raise Exception(msg)

    @staticmethod
    def flatten(xs):
        return [x for y in xs for x in y]

    @property
    def class_nodes(self):
        return self.flatten(o.class_nodes for o in self._ontologies)

    @property
    def data_nodes(self):
        return self.flatten(o.data_nodes for o in self._ontologies)

    @property
    def links(self):
        return self.flatten(o.links for o in self._ontologies)

    @property
    def ontologies(self):
        return self._ontologies


class SemanticSourceDesc(object):
    """
        Semantic source description is the translator betweeen a data
        file (csv file) and the source description (.ssd file).
    """
    def __init__(self, filename, model):
        """

        """
        self.df = pd.read_csv(filename)
        self.file = filename
        self._mapping = {}
        self._model = model
        # initialize the mapping...
        for i, name in enumerate(self.df.columns):
            column = Column(name, self.df, self.file, i)
            self._mapping[column] = Mapping(column, None)

    def map(self, column, data_node, transform=None):
        """

        :param column:
        :param data_node:
        :return:
        """
        columns = self._mapping.keys()
        data_nodes = self._model.data_nodes

        # find the matching items in the list using these attributes
        col = self.find_match(columns, column)
        dn = self.find_match(data_nodes, data_node)
        self._mapping[col] = Mapping(col, dn, transform)
        return self

    @staticmethod
    def find_match(container, item):
        """
        Takes a named object (object with a .name property, say Column
        or a DataNode), and uses the name property to
        find the true match in the column mapping. This allows
        users to use the short hand:

        Column("name")

        rather than

        Column("name", "/some/file.csv", <SomeDataFrame>, 9)

        :param container: An iterable of named objects
        :param item: The item to find.
        :return:
        """
        matches = [x for x in container if x == item]

        if len(matches) == 0:
            # if not found throw error...
            msg = "Failed to find item: " + item.name
            _logger.error(msg)
            raise Exception(msg)
        elif len(matches) == 1:
            return matches[0]
        else:
            # if ambiguities throw error
            msg = "Item name '{}' is ambiguous: {}".format(item.name, matches)
            _logger.error(msg)
            raise Exception(msg)


    @property
    def mappings(self):
        return list(self._mapping.values())

    def __repr__(self):
        """
        Displays the maps of columns to the data nodes of the Ontology
        :return:
        """
        return str(self.mappings)


class Column(object):
    """
    Holds a reference to the original data column
    """
    def __init__(self, name, df=None, filename=None, index=None):
        self.df = df
        self.filename = filename
        self.name = name
        self.index = index

    def __repr__(self):
        """
        Displays the column and the name
        :return:
        """
        return "Column({})".format(self.name)

    def __eq__(self, other):
        return other.name == self.name

    def __hash__(self):
        return id(self)


class Mapping(object):
    """

    """
    def __init__(self, column, node, transform=None):
        self.column = column
        self.node = node
        self.transform = Transform(transform) if transform is not None else None

    def __repr__(self):
        """

        :return:
        """
        if self.transform is not None:
            return "{} -> {} -> {}".format(self.column, self.transform, self.node)
        else:
            return "{} -> {}".format(self.column, self.node)


class Transform(object):
    """

    """
    def __init__(self, func):
        self.func = func

    def __repr__(self):
        return "Transform(3)"