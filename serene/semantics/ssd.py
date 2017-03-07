"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Code for the SSD dataset description object
"""
import logging
import random
import time
import pandas as pd

from serene.ssd_util import SSDJsonBuilder
from serene.elements import Transform, Mapping, \
    Column, ClassNode, DataNode, Link, \
    TransformList, IdentTransform
from ..visualizers import SSDVisualizer
from .base import BaseSemantic

_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)

#LINK_NAME = "relationship"
#DATA_NODE_LINK_NAME = "property"

class SSD(object):
    """
        Semantic source description is the translator between a data
        file (csv file) and the source description (.ssd file).

        The SemanticSourceDesc contains the data columns, a transform
        layer, a mapping between each column to a DataNode, and a
        semantic model built from the available ontologies.
    """
    def __init__(self, filename, modeller):
        """
        Builds up a SemanticSourceDesc object given a csv filename
        and a parent modeller.

        The object contains a Pandas DataFrame of the csv, as
        well as the mapping, transforms and semantic model.
        """
        self.df = pd.read_csv(filename)
        self.file = filename
        self._VERSION = "0.1"
        self._id = random.randint(1e8, 1e9-1)
        self._mapping = {}
        self._modeller = modeller
        self._transforms = TransformList()

        # semantic model
        self._model = BaseSemantic()

        # initialize the mapping...
        for i, name in enumerate(self.df.columns):
            column = Column(name, self.df, self.file, i)
            self._mapping[column] = Mapping(
                column,
                node=None,
                transform=IdentTransform())

    def _find_column(self, column):
        """
        Searches for a Column in the system given a abbreviated
        Column type. e.g. Column("name")

        would return the full Column object (if it exists in the system):

        Column("name", index=1, file="something.csv")

        :param column: An abbreviated Column type
        :return: The actual Column in the system, or an error if not found or ambiguous
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
        Searches for a Datanode in the system given a abbreviated
        DataNode type. e.g. DataNode("name")

        would return the full DataNode object (if it exists in the system):

        DataNode("name", parent=ClassNode("Person"))

        :param data_node: An abbreviated DataNode type
        :return: The actual DataNode in the system, or an error if not found or ambiguous
        """
        data_nodes = self._modeller.data_nodes
        dn = DataNode.search(data_nodes, data_node)
        if dn is None:
            msg = "Failed to find DataNode: {}".format(dn)
            _logger.error(msg)
            raise Exception(msg)
        return dn

    def _find_transform(self, transform):
        """
        Searches for a Transform in the system given a abbreviated
        Transform type. e.g. Transform(1)

        would return the full Transform object (if it exists in the system):

        Transform(sql="select * from table")

        :param transform: An abbreviated Transform type
        :return: The actual Transform in the system, or an error if not found or ambiguous
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
        Searches for a ClassNode in the system given a abbreviated
        ClassNode type. e.g. ClassNode('Person')

        would return the full Link object (if it exists in the system):

        ClassNode("Person", ["name", "dob", "phone-number"])

        :param cls: An abbreviated ClassNode type
        :return: The actual ClassNode in the system, or an error if not found or ambiguous
        """
        classes = set(self.class_nodes)

        c = ClassNode.search(classes, cls)
        if c is None:
            msg = "Failed to find Class: {}".format(c)
            _logger.error(msg)
            raise Exception(msg)
        return c

    def _find_link(self, link):
        """
        Searches for a Link in the system given a abbreviated
        Link type. e.g. Link('owns')

        would return the full Link object (if it exists in the system):

        Link(src=ClassNode("Person"), dst=ClassNode("Car"), relationship='owns')

        :param link: An abbreviated Link type
        :return: The actual Link in the system, or an error if not found or ambiguous
        """
        link_ = Link.search(self.links, link)
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

        :param column: The source Column
        :param data_node: The destination DataNode
        :param transform: The optional transform to be performed on the Column
        :return:
        """
        return self._map(column, data_node, transform, predicted=False)

    def _map(self, column, data_node, transform=None, predicted=False):
        """
        [Internal] Adds a link between the column and data_node for the
        mapping. A transform can also be applied to change
        the column.

        :param column: The source Column
        :param data_node: The destination DataNode
        :param transform: The optional transform to be performed on the Column
        :param predicted: Is the mapping predicted or not (internal)
        :return:
        """
        assert type(column) == Column
        assert type(data_node) == DataNode

        col = self._find_column(column)
        dn = self._find_data_node(data_node)

        m = self._mapping[col]
        if m.transform in self._transforms:
            # remove the old one...
            m = self._mapping[col]
            self._transforms.remove(m.transform)

        # add the transform...
        t = self._add_transform(transform)

        # add the mapping element to the table...
        self._mapping[col] = Mapping(col, dn, t, predicted)

        # by making this mapping, the class node is now in the SemanticModel...
        self._model.add_class_node(dn.parent)

        return self

    def _add_transform(self, transform):
        """
        Initialize the transform (if available)
        otherwise we use the identity transform...

        :param transform:
        :return:
        """
        if transform is not None:
            if callable(transform):
                t = Transform(id=None, func=transform)
            else:
                t = self._find_transform(transform)
        else:
            t = IdentTransform()

        self._transforms.append(t)

        return t

    def link(self, src, dst, relationship):
        """
        Adds a link between ClassNodes. The relationship must
        exist in the ontology

        :param src: Source ClassNode
        :param dst: Destination ClassNode
        :param relationship: The label for the link between the src and dst nodes
        :return: Updated SSD
        """
        s_class = self._find_class(src)
        d_class = self._find_class(dst)

        # now check that the link is in the ontology...
        parent_links = self._modeller.links
        target_link = Link(relationship, s_class, d_class)
        link = Link.search(parent_links, target_link)

        if link is None:
            msg = "Link {} does not exist in the ontology.".format(relationship)
            _logger.error(msg)
            raise Exception(msg)
        elif link in self.links:
            msg = "Link {} is already in the links"
            _logger.info(msg)
        else:
            # if it is ok, then add the new link to the SemanticModel...
            self._model.add_link(link)

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
        Remove the item. The item can be a transform or
        a data node, column or link.

        :return:
        """
        if type(item) == Transform:
            elem = self._find_transform(item)
            self._transforms.remove(elem)

        elif type(item) == DataNode:
            elem = self._find_data_node(item)
            key = None
            value = None
            for k, v in self._mapping.items():
                if v.node == elem:
                    key = k
                    value = v
                    break

            # we put the default mapping back in...
            self._mapping[key] = Mapping(
                value.column,
                node=None,
                transform=IdentTransform()
            )
            # we also need to remove the old transform
            self._transforms.remove(value.transform)

            # note that we now need to check whether
            # we should remove the classNode from the
            # SemanticModel
            old = set(self._model.class_nodes)
            new = set(self.class_nodes)
            targets = new - old
            for t in targets:
                self._model.remove_node(t)

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

            self._model.remove_link(elem)
        else:
            raise TypeError("This type is not supported in remove().")

        return self

    def predict(self):
        """
        Attempt to predict the mappings and transforms for the
        mapping.

        :return: The updated SSD object
        """
        print("Calculating prediction...")
        time.sleep(1)
        print("Done.")
        for mapping in self.mappings:
            if mapping.node is None:
                print("Predicting value for", mapping.column)

                # TODO: make real!
                node = random.choice(self._modeller.data_nodes)
                print("Value {} predicted for {} with probability 0.882".format(node, mapping.column))

                self._map(mapping.column, node, predicted=True)
            else:
                # these are the user labelled data points...
                pass
        return self

    def show(self):
        """
        Shows the Semantic Modeller in a png. This uses the SSDVisualizer helper
        to launch the graphviz image.

        :return: None
        """
        SSDVisualizer(self).show()
        return

    def save(self, file):
        """
        Saves the file to an ssd file

        :param file: The output file
        :return:
        """
        with open(file, "w+") as f:
            f.write(self.json)
        return

    @property
    def version(self):
        """The SSD version"""
        return self._VERSION

    @property
    def ontologies(self):
        """The read-only list of ontology objects"""
        return self._modeller.ontologies

    @property
    def ssd(self):
        """The SSD as a Python dictionary"""
        builder = SSDJsonBuilder(self)
        return builder.to_dict()

    @property
    def json(self):
        """The SSD as a JSON string"""
        builder = SSDJsonBuilder(self)
        return builder.to_json()

    @property
    def model(self):
        """The read-only semantic model"""
        return self._model

    @property
    def predictions(self):
        """The list of all predicted mappings"""
        return list(m for m in self.mappings if m.predicted)

    @property
    def mappings(self):
        """The list of all current mappings"""
        return list(self._mapping.values())

    @property
    def columns(self):
        """The list of columns currently used"""
        return list(self._mapping.keys())

    @property
    def transforms(self):
        """All available transforms"""
        return list(self._transforms)

    @property
    def data_nodes(self):
        """All available data nodes"""
        return list(m.node for m in self.mappings)

    @property
    def class_nodes(self):
        """All available class nodes"""
        return list(set(n.parent for n in self.data_nodes if n is not None))

    @property
    def links(self):
        """
            The links in the SSD. The links come from 2 sources. The object links come from
            what's defined in the semantic model. The data links come
            from the relevant mappings to the columns. The other links
            are not necessary.
        """
        # grab the object links from the model...
        object_links = [link for link in self._model.links
                        if link.link_type == Link.OBJECT_LINK]

        # grab the data links from the model...
        data_links = [link for link in self._model.links
                      if link.link_type == Link.DATA_LINK]

        # combine the relevant links...
        return object_links + [link for link in data_links if link.dst in self.data_nodes]

    def __repr__(self):
        """
        Displays the maps of columns to the data nodes of the Ontology
        :return:
        """
        map_str = [str(m) for m in self.mappings]
        link_str = [str(link) for link in self.links if link.link_type == Link.OBJECT_LINK]

        items = map_str + link_str
        full_str = '\n\t'.join(items)

        return "[\n\t{}\n]".format(full_str)