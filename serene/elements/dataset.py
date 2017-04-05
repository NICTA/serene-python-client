"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Serene Python client: DataSet classes

The DataSet holds the dataset object from the server and associated state and
collection objects.

"""
import collections
import pandas as pd
import logging
import os.path
import json

from serene.utils import convert_datetime
from serene.elements import Column
from serene.elements.semantics.ontology import Ontology


class DataSet(object):
    """
    The DataSet object represents the dataset object from the
    server. This is essentially a persisted CSV file with
    metadata. The columns are also extracted to their own
    datatype
    """
    def __init__(self, json):
        """
        Initialize a DataSet object with a json response
        :param json:
        """
        self.id = json['id']
        self.columns = [Column('--').update(c, self) for c in json['columns']]
        self.filename = json['filename']
        self.path = json['path']
        self.type_map = json['typeMap']
        self.description = json['description']
        self.date_created = convert_datetime(json['dateCreated'])
        self.date_modified = convert_datetime(json['dateModified'])
        self.sample = pd.DataFrame({
            c.name: c.sample for c in self.columns
        })
        self._stored = True

    @property
    def stored(self):
        return self._stored

    def __getitem__(self, i):
        return self.columns[i]

    def __iter__(self):
        return self.columns.__iter__()

    def __len__(self):
        return len(self.columns)

    def __eq__(self, other):
        return all([
            self.id == other.id,
            set(self.column_names()) == set(other.column_names()),
            self.filename == other.filename,
            self.path == other.path,
            self.type_map == other.type_map,
            self.description == other.description,
            self.date_created == other.date_created,
            self.date_modified == other.date_modified #,
        ])

    def column_names(self):
        return [z.name for z in self.columns]

    def column(self, name):
        candidates = [c for c in self.columns if c.name == name]
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            raise ValueError("Column name {} is ambiguous.".format(name))
        else:
            raise ValueError("Column name {} does not exist.".format(name) )

    def summary(self):
        """Shows the information about the dataset"""
        print("\n".join([
            "id: {}".format(self.id),
            "columns: {}".format([c.name for c in self.columns]),
            "filename: {}".format(self.filename),
            "path: {}".format(self.path),
            "typeMap: {}".format(self.type_map),
            "description: {}".format(self.description),
            "dateCreated: {}".format(self.date_created),
            "dateModified: {}".format(self.date_modified),
            "sample: {}".format(self.sample)])
        )

    def __repr__(self):
        disp = "DataSet({}, {})".format(self.id, self.filename)
        return disp

    @staticmethod
    def _process_attributes(attributes, column_map, attr_map):
        """
        Helper method for bind_ssd to process attributes
        :param attributes: attributes from the original ssd
        :param column_map: dictionary column_name -> column_id
        :param attr_map: dictionary attribute_id -> attribute_name
        :return: list of attributes which will be in the bound ssd
        """
        new_attributes = []
        # change ids in attributes
        for attr in attributes:
            if attr["name"] not in column_map:
                logging.warning("Attribute name {} is not in column map".format(attr["name"]))
            else:
                attr["id"] = column_map[attr["name"]]
                attr["columnIds"] = [column_map[attr_map[c]] for c in attr["columnIds"]]
                new_attributes.append(attr)
        return new_attributes

    @staticmethod
    def _process_mappings(mappings, column_map, attr_map):
        """
        Helper method for bind_ssd to process mappings
        :param mappings: list of mappings in the original ssd
        :param column_map: dictionary column_name -> column_id
        :param attr_map: dictionary attribute_id -> attribute_name
        :return:    list of mappings which will be in the bound ssd;
                    set of node ids which should be removed from semantic model
        """
        nodes_to_delete = set()
        new_mappings = []
        # change ids in mappings
        for map in mappings:
            if attr_map[map["attribute"]] not in column_map:
                logging.warning("Deleting attribute {} from mappings".format(attr_map[map["attribute"]]))
                nodes_to_delete.add(map["node"])
            else:
                map["attribute"] = column_map[attr_map[map["attribute"]]]
                new_mappings.append(map)

        return new_mappings, nodes_to_delete

    @staticmethod
    def _process_semantic_model(semantic_model, nodes_to_delete, default_ns):
        """
        Helper method for bind_ssd to process nodes and links in the semantic model
        :param semantic_model: semantic model from the original ssd
        :param nodes_to_delete: set of nodes to be removed from nodes and links
        :param default_ns: default namespace for nodes and links
        :return: list of nodes and links for the semantic model which will be used in the bound ssd
        """
        # specify namespaces in nodes and links
        nodes = semantic_model["nodes"]
        links = semantic_model["links"]
        new_nodes = []
        new_links = []
        for node in nodes:
            if "prefix" not in node:
                node["prefix"] = default_ns
            if node["id"] in nodes_to_delete:
                logging.warning("Deleting node {}".format(node["id"]))
                del node
            else:
                new_nodes.append(node)

        for link in links:
            if "prefix" not in link:
                link["prefix"] = default_ns

            if link["source"] in nodes_to_delete or link["target"] in nodes_to_delete:
                logging.warning("Deleting link {}".format(link["id"]))
                del link
            else:
                new_links.append(link)
        return new_nodes, new_links

    def bind_ssd(self, ssd_json, ontologies, default_ns):
        """
        Modifies ssd json to include proper column ids.
        This method binds ssd_json to the dataset through the column names.

        :param ssd_json: location of the file with ssd json
        :param ontologies: list of ontologies to be used by ssd
        :param default_ns: default namespace to be used for nodes and properties
        :return: json dict
        """
        # a stored dataset
        assert self.stored
        assert (len(ontologies) > 0)
        # stored ontologies
        for onto in ontologies:
            assert (issubclass(type(onto), Ontology))
            assert onto.stored

        with open(ssd_json) as f:
            data = json.load(f)

        logging.debug("Binding process begins!")
        attributes = data["attributes"]
        mappings = data["mappings"]

        column_map = {c.name: c.id for c in self.columns}
        attr_map = {attr["id"]: attr["name"] for attr in attributes}

        new_attributes = self._process_attributes(attributes, column_map, attr_map)
        new_mappings, nodes_to_delete = self._process_mappings(mappings, column_map, attr_map)

        # specify namespaces in nodes and links
        new_nodes, new_links = self._process_semantic_model(data["semanticModel"], nodes_to_delete, default_ns)

        # specify proper ids of ontologies
        data["ontologies"] = [onto.id for onto in ontologies]
        data["mappings"] = new_mappings
        data["attributes"] = new_attributes
        if len(new_attributes) < 1 or len(mappings) < 1:
            logging.warning("There are no attributes or mappings in ssd!")
            data["semanticModel"]["nodes"] = []
            data["semanticModel"]["links"] = []
        else:
            data["semanticModel"]["nodes"] = new_nodes
            data["semanticModel"]["links"] = new_links

        # remove id if present
        if "id" in data:
            del data["id"]

        return data


class DataSetList(collections.MutableSequence):
    """
    Container type for DataSet objects in the Serene Python Client
    """

    def __init__(self, *args):
        self.list = list()
        self.extend(list(args))

    @staticmethod
    def check(v):
        if not issubclass(type(v), DataSet):
            raise TypeError("Only DataSet types permitted: {}".format(v))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]

    def __delitem__(self, i):
        msg = "Use Serene.remove_dataset to correctly remove dataset"
        logging.error(msg)
        raise Exception(msg)

    def __setitem__(self, i, v):
        self.check(v)
        self.list[i] = v

    def insert(self, i, v):
        self.check(v)
        self.list.insert(i, v)

    def __repr__(self):
        ds = []
        for v in self.list:
            ds.append(str(v))
        return "[{}]".format('\n'.join(ds))

    @property
    def summary(self):
        df = pd.DataFrame(columns=[
            'id',
            'name',
            'description',
            'created',
            'modified',
            'rows',
            'cols'
        ])
        for elem in self.list:
            df.loc[len(df)] = [
                elem.id,
                elem.filename,
                elem.description,
                elem.date_created,
                elem.date_modified,
                max(c.size for c in elem.columns),
                len(elem.columns)
            ]
        return df

