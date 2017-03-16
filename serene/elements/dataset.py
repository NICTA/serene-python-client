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
        self.columns = [Column('--').update(c) for c in json['columns']]
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

    def bind_ssd(self, ssd_json, ontologies):
        """
        Modifies ssd json to include proper column ids.
        This method binds ssd_json to the dataset through the column names.

        :param ssd_json: location of the file with ssd json
        :param ontologies
        :return: json dict
        """
        # a stored dataset
        assert (self.stored)
        assert (len(ontologies) > 0)
        # stored ontologies
        for onto in ontologies:
            assert (issubclass(type(onto), Ontology))
            assert (onto.stored)

        with open(ssd_json) as f:
            data = json.load(f)

        attributes = data["attributes"]
        mappings = data["mappings"]
        attr_map = {attr["id"]: attr["name"] for attr in attributes}

        column_map = {c.name: c.id for c in self.columns}

        # change ids in attributes
        data["attributes"] = []
        for attr in attributes:
            attr["id"] = column_map[attr["name"]]
            attr["columnIds"] = [column_map[attr_map[c]] for c in attr["columnIds"]]
            data["attributes"].append(attr)

        # change ids in mappings
        data["mappings"] = []
        for map in mappings:
            map["attribute"] = column_map[attr_map[map["attribute"]]]
            data["mappings"].append(map)

        # specify proper ids of ontologies
        data["ontology"] = [onto.id for onto in ontologies]

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
                os.path.basename(elem.set_filename),
                elem.description,
                elem.date_created,
                elem.date_modified,
                max(c.size for c in elem.columns),
                len(elem.columns)
            ]
        return df

