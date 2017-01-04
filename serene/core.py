"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Serene Python client: Data Integration Software
"""
import logging
import pandas as pd

from functools import lru_cache


from .api import matcher
from .dataset import DataSetList, DataSet, Column
from .model import ModelList, Model


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
        if not issubclass(type(self), SchemaMatcher):
            raise ValueError("Can only clear cache of the SchemaMatcher")

        SchemaMatcher.models.fget.cache_clear()
        SchemaMatcher.datasets.fget.cache_clear()
        SchemaMatcher.predict_all.cache_clear()

        return func(self, *args, **kwargs)
    return wrapper


class SchemaMatcher(object):
    """
        Attributes:
            api : class instance of SchemaMatcherSession

        Properties:
            datasets : A list of all the datasets available on the server
            models : A list of all the models available on the server

        Functions:
            confusion_matrix : A function to determine the confusion matrix for a multi-class prediction
            create_model :
            create_dataset :
            remove_model :
            remove_dataset :
            train_all :
            predict_all :
    """
    def __init__(self,
                 host="127.0.0.1",
                 port=8080,
                 auth=None,
                 cert=None,
                 trust_env=None):
        """
        Initialize class instance of SchemaMatcher.
        """
        logging.info('Initialising schema matcher class object.')
        self.api = matcher.Session(host, port, auth, cert, trust_env)

    @property
    @lru_cache(maxsize=32)
    def datasets(self):
        """Maintains a list of DataSet objects"""
        keys = self.api.dataset_keys()
        ds = DataSetList()
        for k in keys:
            ds.append(DataSet(self.api.dataset(k), self))
        return ds

    @property
    @lru_cache(maxsize=32)
    def models(self):
        """Maintains a list of Model objects"""
        keys = self.api.model_keys()
        ms = ModelList()
        for k in keys:
            ms.append(Model(self.api.model(k), self))
        return ms

    @staticmethod
    def confusion_matrix(prediction):
        """
        Calculate confusion matrix of the model.
        If all_data is not available for the model, it will return None.

        Args:
            prediction: The prediction dataframe, must have label and user_label

        Returns: Pandas data frame.

        """
        if prediction.empty:
            logging.warning("Model all_data is empty. Confusion matrix cannot be calculated.")
            return None

        # take those rows where user_label is available
        available = prediction[["user_label", "label"]].dropna()

        y_actu = pd.Series(available["user_label"], name='Actual')
        y_pred = pd.Series(available["label"], name='Predicted')

        return pd.crosstab(y_actu, y_pred)

    @decache
    def create_model(self,
                     feature_config,
                     description="",
                     classes=None,
                     model_type="randomForest",
                     labels=None,
                     cost_matrix=None,
                     resampling_strategy="ResampleToMean"):

        """
        Post a new model to the schema matcher server.
        Refresh SchemaMatcher instance to include the new model.

        Args:
             feature_config : dictionary
             description : string which describes the model to be posted
             classes : list of class names
             model_type : string
             labels : dictionary
             cost_matrix :
             resampling_strategy : string

        Returns: MatcherModel

        """
        # the unknown class must be present...
        if classes is None:
            classes = ["unknown"]

        assert 'unknown' in classes

        def column_parse(col):
            """Turns a column into the id"""
            if issubclass(type(col), Column):
                return str(col.id)
            else:
                return str(col)

        def label_parse(ld):
            """Convert the column requests into a string id"""
            return {column_parse(k): v for k, v in ld.items()}

        json = self.api.post_model(feature_config,
                                   description,
                                   classes,
                                   model_type,
                                   label_parse(labels),
                                   cost_matrix,
                                   resampling_strategy)  # send API request
        # create model wrapper...
        return Model(json, self)

    @decache
    def create_dataset(self, file_path, description, type_map):
        """
        Upload a new dataset to the schema matcher server.
        Refresh SchemaMatcher instance to include the new dataset.

        Args:
            description: string which describes the dataset to be posted
            file_path: string which indicates the location of the dataset to be posted
            type_map: dictionary with type map for the dataset

        Returns: newly uploaded MatcherDataset

        """
        json = self.api.post_dataset(description, file_path, type_map)
        return DataSet(json, self)

    @decache
    def remove_dataset(self, key):
        """
        Remove dataset `key` (can be the id, or the DataSet object)

        :param key: The key for the dataset or DataSet object
        :return: None
        """
        if issubclass(type(key), DataSet):
            self.api.delete_dataset(key.id)
        else:
            self.api.delete_dataset(key)

    @decache
    def remove_model(self, key):
        """
        Remove model `key` (can be the id, or the Model object)

        :param key: The key for the model or Model object
        :return: None
        """
        if issubclass(type(key), Model):
            self.api.delete_model(key.id)
        else:
            self.api.delete_model(key)

    @decache
    def train_all(self):
        """
        Launch training for all models in the repository.

        Args: None

        Returns: True if training for all models succeeded.

        """
        return all([model.train() for model in self.models])

    @lru_cache(maxsize=32)
    def predict_all(self, model, scores=True, features=False):
        """
        Launch prediction for all models for all datasets in the repository.
        Please be aware that schema matcher at the moment can handle only one model prediction at a time.
        So, running this method with wait=False is not permitted.

        Args: None

        Returns: Nothing, model attributes get updated.

        """
        results = [model.predict(d, scores, features) for d in self.datasets]
        return pd.concat(results)

    def __repr__(self):
        return "<SchemaMatcher({})>".format(self.api)
