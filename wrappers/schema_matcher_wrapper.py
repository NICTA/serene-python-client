
# Python wrapper for the Schema Matcher API
# external
import logging
import requests
import os
import datetime
from urllib.parse import urljoin
import pandas as pd
import json
from wrappers.exception_spec import BadRequestError, NotFoundError, OtherError, InternalDIError

# project
from config import settings as conf_set

################### helper funcs
def pretty_print_POST(req):
    """
    At this point it is completely built and ready
    to be fired; it is "prepared".

    However pay attention at the formatting used in
    this function because it is programmed to be pretty
    printed and may differ from the actual request.
    """
    print('{}\n{}\n{}\n\n{}'.format(
        '-----------START-----------',
        req.method + ' ' + req.url,
        '\n'.join('{}: {}'.format(k, v) for k, v in req.headers.items()),
        req.body,
    ))
###################


class SchemaMatcherSession(object):
    """
    This is the class which sets up the session for the schema matcher api.
    It provides wrappers for all calls to the API.
        Attributes:
            session: Session instance with configuration parameters for the schema matcher server from config.py
            uri: general uri for the schema matcher API
            uri_ds: uri for the dataset endpoint of the schema matcher API
            uri_model: uri for the model endpoint of the schema matcher API
    """
    def __init__(self):
        """Initialize instance of the SchemaMatcherSession."""
        logging.info('Initialising session to connect to the schema matcher server.')
        self.session = requests.Session()
        self.session.trust_env = conf_set.schema_matcher_server['trust_env']
        self.session.auth = conf_set.schema_matcher_server['auth']
        self.session.cert = conf_set.schema_matcher_server['cert']

        # uri to send requests to
        self.uri = conf_set.schema_matcher_server['uri']
        self.uri_ds = urljoin(self.uri, 'dataset') + '/' # uri for the dataset endpoint
        self.uri_model = urljoin(self.uri, 'model') + '/'  # uri for the model endpoint

    def __repr__(self):
        return "<SchemaMatcherSession at (" + str(self.uri) + ")>"

    def __str__(self):
        return self.__repr__()

    def handle_errors(self, response, expr):
        """
        Raise errors based on response status_code
        Args:
            response -- response object from request
            expr -- expression where the error occurs

        :return: raise errors
        """
        if response.status_code == 200:
            # there are no errors here
            return

        if response.status_code == 400:
            logging.error("BadRequest in " + str(expr) + ": message='" + str(response.json()['message']) + "'")
            raise BadRequestError(expr, response.json()['message'])
        elif response.status_code == 404:
            logging.error("NotFound in " + str(expr) + ": message='" + str(response.json()['message']) + "'")
            raise NotFoundError(expr, response.json()['message'])
        else:
            logging.error("Other error with status_code=" + str(response.status_code) +
                          " in " + str(expr) + ": message='" + str(response.json()['message']) + "'")
            raise OtherError(response.status_code, expr, response.json()['message'])


    def list_alldatasets(self):
        """
        List ids of all datasets in the dataset repository at the Schema Matcher server.
        If the connection fails, empty list is returned and connection error is logged.

        :return: list of dataset keys
        """
        logging.info('Sending request to the schema matcher server to list datasets.')
        try:
            r = self.session.get(self.uri_ds)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("list_alldatasets", e)
        self.handle_errors(r, "GET " + self.uri_ds)
        return r.json()

    def post_dataset(self, description, file_path, type_map):
        """
        Post a new dataset to the schema mather server.
            Args:
                 description: string which describes the dataset to be posted
                 file_path: string which indicates the location of the dataset to be posted
                 type_map: dictionary with type map for the dataset

            :return: Dictionary
            """

        logging.info('Sending request to the schema matcher server to post a dataset.')
        try:
            data = {"description": description, "file": file_path, "typeMap": type_map}
            r = self.session.post(self.uri_ds, json=data)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("post_dataset", e)
        self.handle_errors(r, "POST " + self.uri_ds)
        return r.json()

    def update_dataset(self, dataset_key, description, type_map):
        """
        Update an existing dataset in the repository at the schema matcher server.
            Args:
                 description: string which describes the dataset to be posted
                 dataset_key: integer which is the dataset id
                 type_map: dictionary with type map for the dataset

            :return:
            """

        logging.info('Sending request to the schema matcher server to update dataset %d' % dataset_key)
        uri = urljoin(self.uri_ds, str(dataset_key))
        try:
            data = {"description": description, "typeMap": type_map}
            r = self.session.post(uri, data=data)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("update_dataset", e)
        self.handle_errors(r, "PATCH " + uri)
        return r.json()

    def list_dataset(self, dataset_key):
        """
        Get information on a specific dataset from the repository at the schema matcher server.
        Args:
             dataset_key: integer which is the key of the dataset

        :return: dictionary
        """
        logging.info('Sending request to the schema matcher server to get dataset info.')
        uri = urljoin(self.uri_ds, str(dataset_key))
        try:
            r = self.session.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("list_dataset", e)
        self.handle_errors(r, "GET " + uri)
        return r.json()

    def delete_dataset(self, dataset_key):
        """
        Delete a specific dataset from the repository at the schema matcher server.
            Args:
                 dataset_key: int

            :return:
            """
        logging.info('Sending request to the schema matcher server to delete dataset.')
        uri = urljoin(self.uri_ds, str(dataset_key))
        try:
            r = self.session.delete(uri)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("delete_dataset", e)
        self.handle_errors(r, "DELETE " + uri)
        return r.json()

    def list_allmodels(self):
        """
        List ids of all models in the Model repository at the Schema Matcher server.

        :return: list of model keys
        """
        logging.info('Sending request to the schema matcher server to list models.')
        try:
            r = self.session.get(self.uri_model)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("list_allmodels", e)
        self.handle_errors(r, "GET " + self.uri_model)
        return r.json()

    def process_model_input(self, feature_config,
                   description="",
                   classes=["unknown"],
                   model_type="randomForest",
                   labels=None, cost_matrix=None,
                   resampling_strategy="ResampleToMean"):

        # TODO: more sophisticated check of input...
        data = dict()
        if description:
            data["description"] = description
        if classes and type(classes) == list and len(classes)>0:
            data["classes"] = classes
        if model_type:
            data["modelType"] = model_type
        if labels and type(labels) == dict:
            data["labelData"] = labels
        if cost_matrix:
            data["costMatrix"] = cost_matrix
        if resampling_strategy:
            data["resamplingStrategy"] = resampling_strategy
        if feature_config and type(feature_config) == dict and len(feature_config)>0:
            data["features"] = feature_config
        return data

    def post_model(self, feature_config,
                   description="",
                   classes=["unknown"],
                   model_type="randomForest",
                   labels=None, cost_matrix=None,
                   resampling_strategy="ResampleToMean"):
        """
        Post a new model to the schema matcher server.
            Args:
                 feature_config: dictionary
                 description: string which describes the model to be posted
                 classes: list of class names
                 model_type: string
                 labels: dictionary
                 cost_matrix:
                 resampling_strategy: string

            :return: model dictionary
            """

        logging.info('Sending request to the schema matcher server to post a dataset.')

        try:
            data = self.process_model_input(feature_config, description, classes,
                                            model_type, labels, cost_matrix, resampling_strategy)
            r = self.session.post(self.uri_model, json=data)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("post_model", e)
        self.handle_errors(r, "POST " + self.uri_model)
        return r.json()

    def update_model(self, model_key,
                     feature_config,
                     description="",
                     classes=["unknown"],
                     model_type="randomForest",
                     labels=None, cost_matrix=None,
                     resampling_strategy="ResampleToMean"):
        """
        Update an existing model in the model repository at the schema matcher server.
            Args:
                 model_key: integer which is the key of the model in the repository
                 feature_config: dictionary
                 description: string which describes the model to be posted
                 classes: list of class names
                 model_type: string
                 labels: dictionary
                 cost_matrix:
                 resampling_strategy: string

            :return: model dictionary
            """

        logging.info('Sending request to the schema matcher server to update model %d' % model_key)
        uri = urljoin(self.uri_model, str(model_key))
        try:
            data = self.process_model_input(feature_config, description, classes,
                                            model_type, labels, cost_matrix, resampling_strategy)
            r = self.session.post(uri, data=data)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("update_model", e)
        self.handle_errors(r, "PATCH " + uri)
        return r.json()

    def list_model(self, model_key):
        """
        Get information on a specific model in the model repository at the schema matcher server.
        Args:
             model_key: integer which is the key of the model in the repository

        :return: dictionary
        """
        logging.info('Sending request to the schema matcher server to get dataset info.')
        uri = urljoin(self.uri_model, str(model_key))
        try:
            r = self.session.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("list_model", e)
        self.handle_errors(r, "GET " + uri)
        return r.json()

    def delete_model(self, model_key):
        """
            Args:
                 model_key: integer which is the key of the model in the repository

            :return: dictionary
            """
        logging.info('Sending request to the schema matcher server to delete dataset.')
        uri = urljoin(self.uri_model, str(model_key))
        try:
            r = self.session.delete(uri)
        except Exception as e:
            logging.error(e)
            raise InternalDIError("delete_model", e)
        self.handle_errors(r, "DELETE " + uri)
        return r.json()


class MatcherDataset(object):
    """
        Attributes:

    """
    def __init__(self, resp_dict):
        """
        Args:
            resp_dict: dictionary
        """
        self.filename = resp_dict['filename']
        self.filepath = resp_dict['path']
        self.description = resp_dict['description']
        self.ds_key = resp_dict['id']
        self.date_created = resp_dict['dateCreated']
        self.date_modified = resp_dict['dateModified']
        self.columns = resp_dict['columns']
        self.type_map = resp_dict['typeMap']

        headers = []
        data = []
        self.columnMap = []
        for col in self.columns:
            headers.append(col['name'])
            data.append(col['sample'])
            self.columnMap.append([col['id'], col['name'], col['datasetID']])
        self.sample = pd.DataFrame(data).transpose()  # TODO: define dtypes based on typeMap
        self.sample.columns = headers

    def __str__(self):
        return "<MatcherDataset(" + str(self.ds_key) + ")>"

    def __repr__(self):
        return self.__str__()

class MatcherModel(object):
    """
        Attributes:

    """

    def __init__(self):
        """
        Args:

        """
        pass

    def __str__(self):
        return "<MatcherModel(" + str() + ")>"

    def __repr__(self):
        return self.__str__()

class SchemaMatcher(object):
    """
        Attributes:

    """

    def __init__(self):
        """
        Initialize
        """
        logging.info('Initialising schema matcher class object.')
        self.session = SchemaMatcherSession()

        self.ds_keys = self.get_dataset_keys()  # list of keys of all available datasets
        self.dataset = self.populate_datasets()

    def __str__(self):
        return "<SchemaMatcherAt(" + str(self.session.uri) + ")>"

    def __repr__(self):
        return self.__str__()

    def get_dataset_keys(self):
        """Obtain the list of keys of all available datasets."""
        return self.session.list_alldatasets()

    def populate_datasets(self):
        """
        Construct MacherDataset per each dataset key.

        :return: Dictionary for now
        """
        dataset = dict()
        for ds_key in self.ds_keys:
            dataset[ds_key] = MatcherDataset(self.session.list_dataset(ds_key))
        return dataset


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.abspath(os.curdir))

    # setting up the logging
    log_file = 'schemamatcher_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S0') + '.log'
    logging.basicConfig(filename=os.path.join(proj_dir, 'data', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    sess = SchemaMatcherSession()
    all_ds = sess.list_alldatasets()
    all_models = sess.list_allmodels()

    model = sess.list_model(all_models[0])
    features_conf = model["features"]

    sess.post_model(description="blahblah", feature_config=features_conf)

    dm = SchemaMatcher()

    print(dm)
    print(dm.dataset)