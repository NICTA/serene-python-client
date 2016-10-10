"""
License...
"""
import logging
import pandas as pd
import collections
import time
import random
import networkx as nx
import matplotlib.pyplot as plt

from .utils import Searchable
from .semantics import Ontology, DataNode, ClassNode, Link, SemanticBase

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


class SemanticSourceDesc(SemanticBase):
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
        self._transforms = TransformList()
        self._links = LinkList()

        # TODO: Descend properly from the SemanticBase!
        super().__init__()

        # initialize the mapping...
        for i, name in enumerate(self.df.columns):
            column = Column(name, self.df, self.file, i)
            self._mapping[column] = Mapping(column, None)

    def _find_column(self, column):
        """

        :param column:
        :return:
        """
        columns = self._mapping.keys()
        col = Column.search(columns, column)
        if col is None:
            msg = "Failed to find column: {}".format(column)
            _logger.error(msg)
            raise Exception(msg)
        return col

    def _find_data_node(self, data_node):
        """

        :param data_node:
        :return:
        """
        data_nodes = self._model.data_nodes
        dn = DataNode.search(data_nodes, data_node)
        if dn is None:
            msg = "Failed to find DataNode: {}".format(dn)
            _logger.error(msg)
            raise Exception(msg)
        return dn

    def _find_transform(self, transform):
        """

        :param transform:
        :return:
        """
        ts = self._transforms
        t = Transform.search(ts, transform)
        if t is None:
            msg = "Failed to find Transform: {}".format(t)
            _logger.error(msg)
            raise Exception(msg)
        return t

    def _find_class(self, cls):
        """

        :param cls:
        :return:
        """
        classes = set(m.node.parent for m in self._mapping.values() if m.node is not None)

        c = ClassNode.search(classes, cls)
        if c is None:
            msg = "Failed to find Class: {}".format(c)
            _logger.error(msg)
            raise Exception(msg)
        return c

    def _find_link(self, link):
        """

        :param link:
        :return:
        """
        link_ = Link.search(self._links, link)
        if link_ is None:
            msg = "Failed to find Link: {}".format(link)
            _logger.error(msg)
            raise Exception(msg)
        return link_

    def find(self, item):
        """
        Helper function to locate objects using shorthands e.g.

        ssd.find(Transform(1))

        :param item: Transform, Column, ClassNode or DataNode object
        :return:
        """
        if type(item) == Transform:
            return self._find_transform(item)
        elif type(item) == DataNode:
            return self._find_data_node(item)
        elif type(item) == Column:
            return self._find_column(item)
        elif type(item) == ClassNode:
            return self._find_class(item)
        elif type(item) == Link:
            return self._find_link(item)
        else:
            raise TypeError("This type is not supported in find().")

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
        return self._map(column, data_node, transform, predicted=False)

    def _map(self, column, data_node, transform=None, predicted=False):
        """
        [Internal] Adds a link between the column and data_node for the
        mapping. A transform can also be applied to change
        the column.

        :param column:
        :param data_node:
        :param transform:
        :param predicted: Is the mapping predicted or not
        :return:
        """
        col = self._find_column(column)
        dn = self._find_data_node(data_node)

        # intialize the transform (if available)
        t = None
        if transform is not None:
            if callable(transform):
                t = Transform(id=None, func=transform)
            else:
                t = self._find_transform(transform)
            self._transforms.append(t)

        self._mapping[col] = Mapping(col, dn, t, predicted)

        return self

    def link(self, src, dst, relationship):
        """
        Adds a link between ClassNodes. The relationship must
        exist in the ontology

        :param src:
        :param dst:
        :param relationship:
        :return:
        """
        s_class = self._find_class(src)
        d_class = self._find_class(dst)

        # now check that the link is in the ontology...
        parent_links = self._model.links
        target_link = Link(relationship, s_class, d_class)
        link = Link.search(parent_links, target_link)

        if link is None:
            msg = "Link {} does not exist in the ontology.".format(relationship)
            _logger.error(msg)
            raise Exception(msg)
        elif link in self._links:
            msg = "Link {} is already in the links"
            _logger.info(msg)
        else:
            self._links.append(link)

        return self

    def sample(self, column, transform=None, n=10):
        """
        Samples the column

        ssd.sample(Column('name'))

        optionally you can apply the transform to the column...

        ssd.sample(Column('name'), Transform(1))

        :param column: The column object to sample
        :param transform: An optional transform to apply
        :param n: The number of samples to take
        :return: List of samples from the column
        """
        # first we check that this request is valid...
        col = self._find_column(column)

        # next we pull a sample from the Column dataframe...
        samples = col.df[col.name].sample(n).values

        if transform is None:
            return list(samples)
        else:
            # if we have a valid transform, then we can apply it...
            t = self._find_transform(transform)
            return list(map(t.func, samples))

    def remove(self, item):
        """
        Remove the links

        :return:
        """
        if type(item) == Transform:
            elem = self._find_transform(item)
            self._transforms.remove(elem)

        elif type(item) == DataNode:
            elem = self._find_data_node(item)
            key = None
            for k, v in self._mapping.items():
                if v.node == elem:
                    key = k
                    break
            del self._mapping[key]

        elif type(item) == Column:
            elem = self._find_column(item)
            key = None
            for k, v in self._mapping.items():
                if v.col == elem:
                    key = k
                    break
            del self._mapping[key]

        elif type(item) == Link:
            elem = self._find_link(item)
            self._links.remove(elem)

        else:
            raise TypeError("This type is not supported in remove().")

        return self

    def predict(self):
        """
        Attempt to predict the mappings and transforms for the
        mapping.

        :return:
        """
        print("Calculating prediction...")
        time.sleep(1)
        print("Done.")
        for mapping in self._mapping.values():
            if mapping.node is None:
                print("Predicting value for", mapping.column)
                # TODO: make real!
                node = random.choice(self._model.data_nodes)

                print("Value {} predicted for {} with probability 0.882".format(node, mapping.column))

                self._map(mapping.column, node, predicted=True)
            else:
                # these are the user labelled data points...
                pass

    def show(self):
        """
        Draw the semantic model.
        :return:
        """
        self._model.ontologies[2].show()

    @property
    def predictions(self):
        return list(m for m in self._mapping.values() if m.predicted)

    @property
    def mappings(self):
        return list(self._mapping.values())

    @property
    def transforms(self):
        return self._transforms

    @property
    def links(self):
        return self._links

    def __repr__(self):
        """
        Displays the maps of columns to the data nodes of the Ontology
        :return:
        """
        map_str = [str(m) for m in self.mappings]
        link_str = [str(link) for link in self.links]

        items = map_str + link_str
        full_str = '\n\t'.join(items)

        return "[\n\t{}\n]".format(full_str)


class TransformList(collections.MutableSequence):
    """
    Container type for Transform objects in the Semantic Source Description
    """

    def __init__(self, *args):
        self.list = list()
        self.extend(list(args))

    @staticmethod
    def check(v):
        if not isinstance(v, Transform):
            raise TypeError("Only Transform types permitted")

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
        transforms = []
        for v in self.list:
            s = "Transform({}): {}".format(v.id, v)
            transforms.append(s)
        return '\n'.join(transforms)


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
        return '\n'.join(self.list)


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
        A Mapping object that describes the link between a column and
        a data node. An optional transform can be included.
    """
    def __init__(self, column, node=None, transform=None, predicted=False):
        if type(column) != Column:
            raise TypeError("Column type required for 'column' in Mapping object")

        if (transform is not None) and (type(node) != DataNode):
            raise TypeError("DataNode type required for 'node' in Mapping object")

        if (transform is not None) and (type(transform) != Transform):
            raise TypeError("Transform type required for 'transform' in Mapping object")

        self.column = column
        self.node = node
        self.transform = transform
        self.predicted = predicted

    def __repr__(self):
        if self.transform is not None:
            s = "{} -> Transform({}) -> {}".format(self.column, self.transform.id, self.node)
        else:
            s = "{} -> {}".format(self.column, self.node)

        # we add an asterix to denote predicted values...
        if self.predicted:
            s = '{} [*]'.format(s)
        return s


class Transform(Searchable):
    """
        The transform object that describes the transform between
        the raw column and the datanode type.
    """
    _id = 0  # Transform ID value...
    getters = [
        lambda t: t.id,
        lambda t: t.name,
        lambda t: t.sql
    ]

    def __init__(self, id=None, func=None, name='', sql=None):
        if id is None:
            self.id = Transform._id
            Transform._id += 1
        else:
            self.id = id
        self.name = name
        self.func = func
        self.sql = sql if sql is not None else 'SELECT * from {}'.format(name)
        super().__init__()

    def __repr__(self):
        return "{{\n" \
               "    id: {}\n" \
               "    name: {}\n" \
               "    func: {}\n" \
               "    sql: {}\n" \
               "}}".format(self.id, self.name, self.func, self.sql)

    def __hash__(self):
        return id(self)
