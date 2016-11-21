"""
License...
"""
import collections
import logging

from .utils import Searchable
from .semantics import DataNode


# logging functions...
_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)


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

        if (node is not None) and (type(node) != DataNode):
            raise TypeError("DataNode type required for 'node' in Mapping object")

        if (transform is not None) and not issubclass(type(transform), Transform):
            raise TypeError("Transform type required for 'transform' in Mapping object: {}".format(transform))

        self.column = column
        self.node = node
        self.transform = transform if transform is not None else IdentTransform()
        self.predicted = predicted

    def __repr__(self):
        if self.transform is not None:
            if type(self.transform) == IdentTransform:
                s = "{} -> {}".format(self.column, self.node)
            else:
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

    @staticmethod
    def apply(col):
        return 'SELECT {} from {}'.format(col.name, col.filename)

    def __repr__(self):
        return "{{\n" \
               "    id: {}\n" \
               "    name: {}\n" \
               "    func: {}\n" \
               "    sql: {}\n" \
               "}}".format(self.id, self.name, self.func, self.sql)

    def __hash__(self):
        return id(self)


class IdentTransform(Transform):
    """
    Transform that is just the identity...
    """
    def __init__(self, id=None, func=None, name='', sql=None):
        super().__init__(id, func, name, sql)
        self.name = "ident"
        self.func = lambda x: x

    @staticmethod
    def apply(col):
        return 'SELECT {} from {}'.format(col.name, col.filename)


class TransformList(collections.MutableSequence):
    """
    Container type for Transform objects in the Semantic Source Description

    """
    def __init__(self, *args):
        self.list = list()
        self.extend(list(args))

    @staticmethod
    def check(v):
        if not issubclass(type(v), Transform):
            raise TypeError("Only Transform types permitted: {}".format(v))

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

