"""
License...
"""
import logging
import pandas as pd

from .utils import Searchable
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
        Adds a link between the column and data_node for the
        mapping. A transform can also be applied to change
        the column.

        :param column:
        :param data_node:
        :param transform:
        :return:
        """
        columns = self._mapping.keys()
        data_nodes = self._model.data_nodes

        # find the matching items in the list using these attributes
        col = Column.search(columns, column)
        dn = DataNode.search(data_nodes, data_node)

        if col is None:
            msg = "Failed to find column: {}".format(column)
            _logger.error(msg)
            raise Exception(msg)

        if dn is None:
            msg = "Failed to find data node: {}".format(data_node)
            _logger.error(msg)
            raise Exception(msg)

        self._mapping[col] = Mapping(col, dn, transform)

        return self

    @property
    def mappings(self):
        return list(self._mapping.values())

    def __repr__(self):
        """
        Displays the maps of columns to the data nodes of the Ontology
        :return:
        """
        map_str = '\n\t'.join(str(m) for m in self.mappings)
        return "[\n\t{}\n]".format(map_str)


class Column(Searchable):
    """
    Holds a reference to the original data column
    """
    # now we setup the search parameters...
    getters = [
        lambda col: col.name,
        lambda col: col.index,
        lambda col: col.filename
    ]

    def __init__(self, name, df=None, filename=None, index=None):
        self.df = df
        self.filename = filename
        self.name = name
        self.index = index
        super().__init__()

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