"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Serene Python client: Model classes

The Model holds the model object from the server and associated state and
collection objects.

"""
import collections
import logging
import pprint
import time
from enum import Enum, unique
from functools import lru_cache

import pandas as pd

from serene.elements.dataset import DataSet, Column
from serene.utils import convert_datetime

@unique
class SamplingStrategy(Enum):
    UPSAMPLE_TO_MAX = "UpsampleToMax"
    RESAMPLE_TO_MEAN = "ResampleToMean"
    UPSAMPLE_TO_MEAN = "UpsampleToMean"
    BAGGING = "Bagging"
    BAGGING_TO_MAX = "BaggingToMax"
    BAGGING_TO_MEAN = "BaggingToMean"
    NO_RESAMPLING = "NoResampling"

    @classmethod
    def values(cls):
        return [z.value for z in cls]


@unique
class ModelType(Enum):
    RANDOM_FOREST = "randomForest"

    @classmethod
    def values(cls):
        return [z.value for z in cls]


class Status(Enum):
    """Enumerator of possible model states."""
    ERROR = "error"
    UNTRAINED = "untrained"
    BUSY = "busy"
    COMPLETE = "complete"

    @staticmethod
    def to_status(status):
        """Helper function to convert model state
        from a string to Status Enumerator."""
        if status == "error":
            return Status.ERROR
        if status == "untrained":
            return Status.UNTRAINED
        if status == "busy":
            return Status.BUSY
        if status == "complete":
            return Status.COMPLETE
        raise ValueError("Status {} is not supported.".format(status))


class ModelState(object):
    """
    Class to wrap the model state.

    Attributes:
        status
        message
        date_created
        date_modified
    """
    def __init__(self, json):
        """
        Initialize instance of class ModelState.

        Args:
            status : string
            date_created
            date_modified
        """
        self.status = Status.to_status(json['status'])  # convert to Status enum
        self.message = json['message']
        self.date_modified = convert_datetime(json['dateChanged'])

    def __repr__(self):
        if len(self.message):
            return "ModelState({}, modified on {}, msg: {})".format(
                self.status,
                self.date_modified,
                self.message
            )
        else:
            return "ModelState({}, modified on {})".format(
                self.status,
                self.date_modified
            )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        return (self.status == other.status) and \
               (self.message == other.message) and \
               (self.date_modified == other.date_modified)


def decache(func):
    """
    Decorator for clearing the cache. Here we explicitly mark the
    caches that need clearing. There may be a more elegant way to
    do this by having a new lru_cache wrapper that adds the functions
    to a global store, and the cache busters simply clear from this
    list.
    """
    def wrapper(self, *args, **kwargs):
        """
        Wrapper function that busts the cache for each lru_cache file
        """
        if not issubclass(type(self), Model):
            raise ValueError("Can only clear cache of the Model")

        Model.predict.cache_clear()
        Model._full_predict.cache_clear()

        return func(self, *args, **kwargs)
    return wrapper


class Model(object):
    """Holds information about the Model object on the Serene server"""
    def __init__(self, json, session, dataset_endpoint):

        self._pp = pprint.PrettyPrinter(indent=4)

        # self.parent = parent
        #self.parent = parent  # parent.api
        #self.endpoint = self.parent.api.model
        #self._ds_endpoint = dataset_endpoint
        self._session = session
        self._ds_endpoint = dataset_endpoint

        self.PREDICT_KEYS = [
            "column_id",
            "column_name",
            "confidence",
            "dataset_id",
            "model_id",
            "label",
            "user_label"
        ]
        self.PREDICT_SCORE_PRE = "scores"
        self.PREDICT_FEATURE_PRE = "features"

        self.description = json['description']
        self.id = json['id']
        self.model_type = json['modelType']
        self.classes = json['classes']
        self.features = json['features']
        self.cost_matrix = json['costMatrix']
        self.resampling_strategy = json['resamplingStrategy']
        self.label_data = json['labelData']
        self.ref_datasets = json['refDataSets']
        self.model_path = json['modelPath'] if 'modelPath' in json else ''
        self.state = ModelState(json['state'])
        self.date_created = convert_datetime(json['dateCreated'])
        self.date_modified = convert_datetime(json['dateModified'])
        self.num_bags = json['numBags']
        self.bag_size = json['bagSize']

    @decache
    def add_label(self, col, label):
        """Users can add a label to column col. `col` can be a
        Column object or a string id

        Args:
            col: Column object or int id
            label: The class label (this must exist in the model params)

        Returns:
            The updated set of labels
        """
        key, value = self._label_entry(col, label)

        label_table = self.label_data
        label_table[key] = value

        json = self._session.model_api.update(self.id, labels=label_table)

        self._update(json)

        return self.labels

    @decache
    def add_labels(self, table):
        """Users can add a label to column col. `col` can be a
        Column object or a string id

        Args:
            table: key-value dict with keys as Column objects or int ids
                   and the values as the class labels (this must exist in the model params)

        Returns:
            The updated set of labels
        """
        label_table = self.label_data

        for k, v in table.items():
            key, value = self._label_entry(k, v)
            label_table[key] = value

        json = self._session.model_api.update(self.id, labels=label_table)

        self._update(json)

        return self.labels

    @decache
    def train(self):
        """
        Send the training request to the API.

        Args:
            wait : boolean indicator whether to wait for the training to finish.

        Returns: boolean -- True if model is trained, False otherwise

        """
        self._session.model_api.train(self.id)  # launch training

        def state():
            """Query the server for the model state"""
            json = self._session.model_api.item(self.id)
            self._update(json)
            return self.state

        def is_finished():
            """Check if training is finished"""
            return state().status in {Status.COMPLETE, Status.ERROR}

        print("Training model {}...".format(self.id))
        while not is_finished():
            logging.info("Waiting for the training to complete...")
            time.sleep(2)  # wait in polling loop

        print("Training complete for {}".format(self.id))
        logging.info("Training complete for {}.".format(self.id))
        return state().status == Status.COMPLETE

    @lru_cache(maxsize=32)
    def predict(self, dataset, scores=True, features=False):
        """Runs a prediction across the `dataset`"""
        logging.debug("Model prediction start...")
        df = self._full_predict(dataset)
        logging.debug("--> full predict done")

        keys = [k for k in self.PREDICT_KEYS]

        if features:
            keys += [col for col in df.columns if self.PREDICT_FEATURE_PRE in col]

        if scores:
            keys += [col for col in df.columns if self.PREDICT_SCORE_PRE in col]
        logging.debug("-->finished processing df")

        return df[keys]

    @property
    def is_error(self):
        """Returns True if model is in the error state"""
        return self.state.status == Status.ERROR

    @property
    def summary(self):
        """Shows the information about the dataset"""
        df = {
            "id": self.id,
            "description": self.description,
            "modelType": self.model_type,
            "classes": self.classes,
            "features": self.features,
            "cost_matrix": self.cost_matrix,
            "resamplingStrategy": self.resampling_strategy,
            "labelData": self.label_data,
            "refDataSets": self.ref_datasets,
            "modelPath": self.model_path,
            "state": self.state,
            "dateCreated": self.date_created,
            "dateModified": self.date_modified,
            "numBags": self.num_bags,
            "bagSize": self.bag_size
        }
        return self._pp.pformat(df)

    @property
    def labels(self):
        """Returns the label DataFrame, which contains the
        user-specified columns
        """
        keys = [int(k) for k in self.label_data.keys()]
        labels = [lab for lab in self.label_data.values()]

        return pd.DataFrame({
            'user_label': labels,
            'column_name': [self._column_lookup[x].name for x in keys],
            'dataset_id': [self._column_lookup[x].datasetID for x in keys],
            'column_id': keys
        })

    @lru_cache(maxsize=32)
    def _full_predict(self, dataset):
        """
        Predict the column labels for this dataset

        :param dataset: Can be dataset id or a DataSet object

        :return: Pandas Dataframe with prediction data
        """
        if issubclass(type(dataset), DataSet):
            key = dataset.id
        else:
            key = int(dataset)

        json = self._session.model_api.predict(self.id, key)

        logging.debug("converting model predictions to pandas df")
        df = self._predictions(json)
        logging.debug("convertion success")

        return df

    @decache
    def _update(self, json):
        """Re-initializes the model based on an updated json string"""
        self.__init__(json, self._session, self._ds_endpoint)

    def _columns(self):
        # first we grab all the columns out from the datasets
        return [ds.columns for ds in self._ds_endpoint.items]

    @property
    def _column_lookup(self):
        # first we grab all the columns out from the datasets
        return {item.id: item for sublist in self._columns() for item in sublist}

    def _label_entry(self, col, label):
        """Prepares the label entry by ensuring the key is a
           valid string key and the label is also valid"""

        if issubclass(type(col), Column):
            key = str(col.id)
        else:
            key = str(col)

        # ensure that the key is valid...
        assert \
            int(key) in self._column_lookup, \
            "Key '{}' is not in column ids {}".format(key, self._column_lookup.keys())

        # ensure that the label is in the classes...
        assert \
            label in self.classes, \
            "Label '{}' is not in classes {}".format(label, self.classes)

        return str(key), label

    def _predictions(self, json):
        """
        Here we flatten out the nested json returned from Serene
        into a flat table structure.

        Initially we have:
            datasetID: {
                item1 : asdf
                item2 : qwer
                nested-item1: {
                    item1: asdf
                    item2: qwer
                }
                nested-item2: {
                    item1: asdf
                    item2: qwer
                }
            }

        and we want to flatten it to a DataFrame with:

            'datasetID': [...]
            'item1': [...]
            'item2': [...]
            'nested-item1-item1: [...]
            'nested-item1-item2: [...]
            'nested-item2-item1: [...]
            'nested-item2-item2: [...]

        :param json: JSON returned from the backend
        :return: Flattened dictionary of lists for each key
        """
        table = self._flat_predict(json)
        logging.debug("flattening of predictions success")

        df = pd.DataFrame(table)

        # now we add the user labels
        final = pd.merge(
            df,
            self.labels[['column_id', 'user_label']],
            on='column_id',
            how='left'
        )

        logging.debug("merging of predictions success")

        final['column_name'] = final['column_id'].apply(
            lambda col_id: self._column_lookup[col_id].name)
        logging.debug("looking up columns")

        return final

    def _flat_predict(self, json):
        """
         Here we flatten out the nested json returned from Serene
        into a flat table structure.

        Initially we have:
            datasetID: {
                item1 : asdf
                item2 : qwer
                nested-item1: {
                    item1: asdf
                    item2: qwer
                }
                nested-item2: {
                    item1: asdf
                    item2: qwer
                }
            }

        and we want to flatten it to a DataFrame with:
            {
                'datasetID': [...]
                'item1': [...]
                'item2': [...]
                'nested-item1-item1: [...]
                'nested-item1-item2: [...]
                'nested-item2-item1: [...]
                'nested-item2-item2: [...]
            }

        :param json: JSON returned from the backend
        :return: Flattened dictionary of lists for each key

        """
        # first we need to drop the datasetID key down
        def update(d, key, val):
            """Better dictionary update function"""
            d[key] = val
            return d

        # first store the IDs for later...
        datasetID = json['dataSetID']
        modelID = json['modelID']

        # next drop the key into a "columnID":key inside the child dictionary
        dlist = [update(d, "column_id", int(k)) for k, d in json['predictions'].items()]

        # we now want to flatten the nested items
        flat_list = [self._flatten(d) for d in dlist]

        # now we add the IDs as well
        final = [update(d, "dataset_id", int(datasetID)) for d in flat_list]
        final = [update(d, "model_id", int(modelID)) for d in final]

        table = collections.defaultdict(list)

        # WARNING: This is dangerous! If a value is missing it will
        # break! We need to add np.nan as missing values if there
        # are missing elements...
        for d in final:
            for k, v in d.items():
                table[k].append(v)

        return table

    def _flatten(self, d, parent_key='', sep='_'):
        """
        Flattens a nested dictionary by squashing the
        parent keys into the sub-key name with separator
        e.g.
            flatten({'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]})
            >> {'a': 1, 'c_a': 2, 'c_b_x': 5, 'd': [1, 2, 3], 'c_b_y': 10}

        :param d: The nested dictionary
        :param parent_key: parent key prefix
        :param sep: The separator to
        :return:
        """
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(self._flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class ModelList(collections.MutableSequence):
    """
    Container type for Model objects in the Serene Python Client
    """

    def __init__(self, *args):
        self.list = list()
        self.extend(list(args))

    @staticmethod
    def check(v):
        if not issubclass(type(v), Model):
            raise TypeError("Only Model types permitted: {}".format(v))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]

    def __delitem__(self, i):
        msg = "Use SchemaMatcher.remove_model to correctly remove model"
        logging.error(msg)
        raise Exception(msg)

    def __setitem__(self, i, v):
        self.check(v)
        self.list[i] = v

    def insert(self, i, v):
        self.check(v)
        self.list.insert(i, v)

    def __repr__(self):
        ms = []
        for v in self.list:
            s = "Model({})".format(v.id)
            ms.append(s)
        return "[{}]".format('\n'.join(ms))

    @property
    def summary(self):
        df = pd.DataFrame(columns=[
            'model_id',
            'description',
            'created',
            'modified',
            'status',
            'state_modified'
        ])
        for elem in self.list:
            df.loc[len(df)] = [
                elem.id,
                elem.description,
                elem.date_created,
                elem.date_modified,
                elem.state.status.name,
                elem.state.date_modified
            ]
        return df
