"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Code for the SSD dataset description object
"""
import json
import logging
import random
import time
import os
import tempfile
from collections import OrderedDict

from serene.semantics import Ontology
from serene.elements import Transform, Mapping, Column, ClassNode
from serene.elements import DataNode, Link, TransformList, IdentTransform
from serene.visualizers import SSDVisualizer
from .base import BaseSemantic
from ..utils import gen_id
from collections import defaultdict

_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)


class SSDInternal(object):
    """
        Semantic source description is the translator between a DataSet
        and a set of Ontologies

        The SemanticSourceDesc contains the data columns, a transform
        layer, a mapping between each column to a DataNode, and a
        semantic model built from the available ontologies.
    """
    def __init__(self,
                 dataset=None,
                 ontology=None,
                 ontology_endpoint=None,
                 dataset_endpoint=None):
        """
        Builds up a SemanticSourceDesc object given a dataset and
        and a parent ontology (or set of ontologies).
        """
        if not issubclass(type(ontology), Ontology):
            msg = "Required Ontology, not {}".format(type(ontology))
            raise Exception(msg)

        # dataset and ontology must be stored on the server!!!!
        if not dataset.stored:
            msg = "{} must be stored on the server, use <Serene>.datasets.upload"
            raise Exception(msg)

        if not ontology.stored:
            msg = "{} must be stored on the server, use <Serene>.ontologies.upload"
            raise Exception(msg)

        self._name = gen_id()
        self._dataset = dataset
        self._ontology = ontology
        self._VERSION = "0.1"
        self._id = None
        self._mapping = {}  # Column to Mapping mapping
        self._transforms = TransformList()  # to be removed
        self._stored = False  # is stored on the server?
        #self._path = None  #self._get_filename()  # the path to the local file...
        self._date_created = None
        self._date_modified = None
        self._ds_endpoint = dataset_endpoint
        self._on_endpoint = ontology_endpoint

        # semantic model
        self._semantic_model = BaseSemantic()

        # initialize the mapping...
        #for i, column in enumerate(self._dataset.columns):
        #    self._mapping[column] = Mapping(
        #        column,
        #        node=None,
        #        transform=IdentTransform())

    @classmethod
    def from_json(cls, json_string, dataset_endpoint, ontology_endpoint):
        """Create the object from json directly"""
        new = cls(None,
                  None,
                  dataset_endpoint,
                  ontology_endpoint)
        return new.read(json_string)

    def read(self, json):
        """
        Updates parameters from a json string.

        :param json:
        :return:
        """
        self._stored = True
        self._name = json['name']
        self._date_created = json['dateCreated']
        self._date_modified = json['dateModified']
        self._id = int(json['id'])

        ontologies = json['ontology']
        self._ontology = self._on_endpoint.get(ontologies[0])

        # fill out the dataset...
        jsm = json["semanticModel"]
        ds_key = jsm["datasetID"]
        self._dataset = self._ds_endpoint.get(ds_key)

        # now build the more complicated elements
        self._semantic_model = self._build_semantic(json)
        self._build_mapping(json)
        return self

    def _build_mapping(self, json):
        """Builds the mapping from the json string"""
        jsm = json["semanticModel"]
        raw_nodes = jsm["nodes"]
        raw_map = jsm["mappings"]

        def node_split(label):
            items = label.split(".")
            return items[0], items[-1]

        def get(label):
            DataNode.search(
                self._semantic_model.data_nodes,
                DataNode(*node_split(label)))

        data_nodes = {n["id"]: get(n["label"])
                      for n in raw_nodes
                      if n["type"] == "DataNode"}

        column_map = {col.id: col for col in self._dataset.columns}

        for link in raw_map:
            cid = column_map[link["attribute"]]
            node = data_nodes[link["node"]]
            self.map(cid, node)

    def _build_semantic(self, json):
        """Builds the semantic model from a json string"""
        sm = BaseSemantic()
        jsm = json["semanticModel"]
        raw_nodes = jsm["nodes"]
        links = jsm["links"]

        def node_split(label):
            items = label.split(".")
            return items[0], items[-1]

        class_table = {n["id"]: n["label"] for n in raw_nodes
                       if n["type"] == "ClassNode"}

        data_nodes = {n["id"]: node_split(n["label"])[1]
                      for n in raw_nodes
                      if n["type"] == "DataNode"}

        class_links = [(n["source"], n["target"], n["label"])
                       for n in links
                       if n["type"] == "ObjectPropertyLink"]

        data_links = [(n["source"], n["target"], n["label"])
                      for n in links
                      if n["type"] == "DataPropertyLink"]

        # first fill the class links in a lookup table
        lookup = defaultdict(list)
        for link in data_links:
            src, dst, name = link
            lookup[src].append(dst)

        # next we pull the class names out with the data nodes
        for cls, dns in lookup.items():
            class_node = class_table[cls]
            data_nodes = [data_nodes[dn] for dn in dns]
            sm.class_node(class_node, data_nodes)

        # finally we add the class links
        for link in class_links:
            src, dst, name = link
            sm.link(src, name, dst)

        return sm

    # def read_from_file(self, filename):
    #     """Initialize SSD from a file"""
    #     with open(filename) as f:
    #         data = json.load(f)
    #         return self.read(data)
    #
    # def write(self):
    #     """
    #     Writes the SSD out to the path.
    #     :return: None
    #     """
    #     data = self.json
    #     with open(self._path, 'wb') as f:
    #         f.write(data)
    #
    # @property
    # def filename(self):
    #     return self._path
    #
    # def _get_filename(self):
    #     return os.path.join(
    #         tempfile.gettempdir(),
    #         self._name
    #     )

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
        data_nodes = self._ontology.data_nodes
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

        :param item: Transform, Column, ClassNode, Link or DataNode object
        :return:
        """
        if issubclass(type(item), Transform):
            return Transform.search(self.transforms, item) #self._find_transform(item)
        elif issubclass(type(item), DataNode):
            return DataNode.search(self.data_nodes, item) #self._find_data_node(item)
        elif issubclass(type(item), Column):
            return Column.search(self.columns) #self._find_column(item)
        elif issubclass(type(item), ClassNode):
            return  #self._find_class(item)
        elif issubclass(type(item), Link):
            return  #self._find_link(item)
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
        self._semantic_model.add_class_node(dn.parent)

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
        parent_links = self._ontology.links
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
            self._semantic_model.add_link(link)

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
            old = set(self._semantic_model.class_nodes)
            new = set(self.class_nodes)
            targets = new - old
            for t in targets:
                self._semantic_model.remove_node(t)

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

            self._semantic_model.remove_link(elem)
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
        return self._ontologies

    @property
    def ssd(self):
        """The SSD as a Python dictionary"""
        builder = SSDJsonWriter(self)
        return builder.to_dict()

    @property
    def json(self):
        """The SSD as a JSON string"""
        builder = SSDJsonWriter(self)
        return builder.to_json()

    @property
    def model(self):
        """The read-only semantic model"""
        return self._semantic_model

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
        object_links = [link for link in self._semantic_model.links
                        if link.link_type == Link.OBJECT_LINK]

        # grab the data links from the model...
        data_links = [link for link in self._semantic_model.links
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


class SSDJsonWriter(object):
    """
    Helper class to build up the json output for the SSD file.
    """
    def __init__(self, ssd):
        """

        :param ssd:
        """
        self._ssd = ssd

        self._all_nodes = [n for n in
                           self._ssd.class_nodes + self._ssd.data_nodes
                           if n is not None]

        self._node_map = {m: i for i, m in enumerate(self._all_nodes)}
        self._attr_map = {m: i for i, m in enumerate(self._ssd.mappings)}

    def to_dict(self):
        """

        :return:
        """
        d = OrderedDict()
        d["version"] = self._ssd.version
        d["name"] = self._ssd.file
        d["columns"] = self.columns
        d["attributes"] = self.attributes
        d["ontology"] = [o.set_filename for o in self._ssd.ontologies]
        d["semanticModel"] = self.semantic_model
        d["mappings"] = self.mappings

        return d

    def to_json(self):
        """

        :return:
        """
        return json.dumps(self.to_dict(), indent=4)

    @property
    def columns(self):
        """

        :return:
        """
        return [
            {
                "id": c.index,
                "name": c.name
            } for c in self._ssd.columns]

    @property
    def attributes(self):
        """

        :return:
        """
        return [
            {
                "id": self._attr_map[m],
                "name": m.column.name,
                "label": m.transform.name,
                "columnIds": [m.column.index],
                "sql": m.transform.apply(m.column)
            } for m in self._ssd.mappings]

    @property
    def semantic_model(self):
        """

        :return:
        """
        return {
            "nodes": [node.ssd_output(index)
                      for index, node in enumerate(self._all_nodes)],
            "links": [link.ssd_output(self._node_map)
                      for link in self._ssd.links]
        }

    @property
    def mappings(self):
        """

        :return:
        """
        return [
            {
                "attribute": self._attr_map[m],
                "node": self._node_map[m.node]
            } for m in self._ssd.mappings if m.node is not None]