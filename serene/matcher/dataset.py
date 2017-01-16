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

from .utils import convert_datetime


class Column(object):
    """
    The Column object is owned by a DataSet. It reflects the Column
    object from the backend, and contains a sample of the dataset,
    the logicalType, size stats and an id. The id is used to refer
    to columns around the project.
    """
    def __init__(self, json):
        """
        Initialize a Column object using a json table...

        :param json: JSON table from the server
        """
        self.index = json['index']
        self.path = json['path']
        self.name = json['name']
        self.id = json['id']
        self.size = json['size']
        self.datasetID = json['datasetID']
        self.sample = json['sample']
        self.logicalType = json['logicalType']


class DataSet(object):
    """
    The DataSet object represents the dataset object from the
    server. This is essentially a persisted CSV file with
    metadata. The columns are also extracted to their own
    datatype
    """
    def __init__(self, json, parent):
        """
        Initialize a DataSet object with a json response
        :param json:
        """
        self.parent = parent
        self.id = json['id']
        self.columns = [Column(c) for c in json['columns']]
        self.filename = json['filename']
        self.path = json['path']
        self.type_map = json['typeMap']
        self.description = json['description']
        self.date_created = convert_datetime(json['dateCreated'])
        self.date_modified = convert_datetime(json['dateModified'])
        self.sample = pd.DataFrame({
            c.name: c.sample for c in self.columns
        })

    def __getitem__(self, i):
        return self.columns[i]

    def __iter__(self):
        return self.columns.__iter__()

    def __len__(self):
        return len(self.columns)

    def column(self, name):
        candidates = [c for c in self.columns if c.name == name]
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            raise ValueError("Column name {} is ambiguous.".format(name))
        else:
            raise ValueError("Column name {} does not exist.".format(name) )

    @property
    def summary(self):
        """Shows the information about the dataset"""
        return "\n".join([
            "id: {}".format(self.id),
            "columns: {}".format(self.columns),
            "filename: {}".format(self.filename),
            "path: {}".format(self.path),
            "typeMap: {}".format(self.type_map),
            "description: {}".format(self.description),
            "dateCreated: {}".format(self.date_created),
            "dateModified: {}".format(self.date_modified),
            "sample: {}".format(self.sample),
        ])


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
        msg = "Use SchemaMatcher.remove_dataset to correctly remove dataset"
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
            s = "DataSet({})".format(v.id)
            ds.append(s)
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
                os.path.basename(elem.filename),
                elem.description,
                elem.date_created,
                elem.date_modified,
                max(c.size for c in elem.columns),
                len(elem.columns)
            ]
        return df

