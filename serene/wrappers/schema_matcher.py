"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Schema Matcher wrapper around the Serene API backend schema matcher endpoints
"""
import logging
import requests
# import pandas as pd
# import time
# import numpy as np
#
# from .. import utils
from ..exceptions import BadRequestError, NotFoundError, OtherError, InternalError
from enum import Enum
from urllib.parse import urljoin


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
        raise InternalError("to_status({})", "status not supported.".format(status))


class Session(object):
    """
    This is the class which sets up the session for the schema matcher api.
    It provides wrappers for all calls to the API.
    This is an internal class and should not be explicitly used/called by the user.

        Attributes:
            session: Session instance with configuration parameters for the schema matcher server from config.py
            _uri: general uri for the schema matcher API
            _uri_ds: uri for the dataset endpoint of the schema matcher API
            _uri_model: uri for the model endpoint of the schema matcher API
    """
    def __init__(self, host, port,
                 auth=None, cert=None, trust_env=None):
        """
        Initialize and maintain a session with the Serene backend
        :param host: The address of the Serene backend
        :param port: The port number
        :param version: The version of the API (default defined in const) e.g. v1.0
        :param auth: The authentication token (None if non-secure connection)
        :param cert: The security certificate
        :param trust_env: The trusted environment token
        """
        logging.info('Initialising session to connect to the schema matcher server.')

        self.session = requests.Session()
        self.session.trust_env = trust_env
        self.session.auth = auth
        self.session.cert = cert

        # root URL for the Serene backend
        root = "http://{}:{}".format(host, port)

        # this will throw an error an fail the init if not available...
        self.version = self._test_connection(root)

        self._uri = urljoin(root, self.version + '/')

        self._uri_ds = urljoin(self._uri, 'dataset/')  # + '/'  # uri for the dataset endpoint
        self._uri_model = urljoin(self._uri, 'model/')  # + '/'  # uri for the model endpoint

    def _test_connection(self, root):
        """
        Tests the connection at root. The server response should be a json
        string indicating the correct version endpoint to use e.g.
        {'version': 'v1.0'}

        Returns: The version string of the server

        """
        logging.info('Testing connection to the server...')

        try:
            # test connection to the server
            req = self.session.get(root)

            # attempt to decode the version...
            version = req.json()['version']

            logging.info("Connection to the server has been established.")
            print("Connection to server established: {}".format(root))

            return version

        except requests.exceptions.RequestException as e:
            msg = "Failed to connect to {}".format(root)
            print(msg)
            logging.error(msg)
            logging.error(e)
        except KeyError:
            msg = "Failed to decode server response"
            logging.error(msg)
            raise ConnectionError(msg)

        # here we raise a simpler error to prevent the giant
        # requests error stack...
        raise ConnectionError(msg)

    def __repr__(self):
        return "<Session at (" + str(self._uri) + ")>"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def _handle_errors(response, expr):
        """
        Raise errors based on response status_code

        Args:
            response : response object from request
            expr : expression where the error occurs

        Returns: None or raise errors.

        Raises: BadRequestError, NotFoundError, OtherError.

        """
        def log_msg(msg_type, status_code, expr, response):
            logging.error("{} ({}) in {}: message='{}'".format(msg_type, status_code, expr, response))

        if response.status_code == 200 or response.status_code == 202:
            # there are no errors here
            return

        if response.status_code == 400:
            log_msg("BadRequest", response.status_code, expr, response)
            raise BadRequestError(expr, response.json()['message'])
        elif response.status_code == 404:
            log_msg("NotFound", response.status_code, expr, response)
            raise NotFoundError(expr, response.json()['message'])
        else:
            log_msg("RequestError", response.status_code, expr, response)
            raise OtherError(response.status_code, expr, response.json()['message'])

    def dataset_keys(self):
        """
        List ids of all datasets in the dataset repository at the Schema Matcher server.
        If the connection fails, empty list is returned and connection error is logged.

        Returns: list of dataset keys.
        Raises: InternalDIError on failure.
        """
        logging.info('Sending request to the schema matcher server to list datasets.')
        try:
            r = self.session.get(self._uri_ds)
        except Exception as e:
            logging.error(e)
            raise InternalError("list_alldatasets", e)
        self._handle_errors(r, "GET " + self._uri_ds)
        return r.json()

    def post_dataset(self, description, file_path, type_map):
        """
        Post a new dataset to the schema mather server.
        Args:
             description: string which describes the dataset to be posted
             file_path: string which indicates the location of the dataset to be posted
             type_map: dictionary with type map for the dataset

        Returns: Dictionary.
        """

        logging.info('Sending request to the schema matcher server to post a dataset.')
        try:
            f = {"file": open(file_path, "rb")}
            data = {"description": str(description), "typeMap": type_map}
            r = self.session.post(self._uri_ds, data=data, files=f)
        except Exception as e:
            logging.error(e)
            raise InternalError("post_dataset", e)
        self._handle_errors(r, "POST " + self._uri_ds)
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
        uri = self._dataset_endpoint(dataset_key)
        try:
            data = {"description": description, "typeMap": type_map}
            r = self.session.post(uri, data=data)
        except Exception as e:
            logging.error(e)
            raise InternalError("update_dataset", e)
        self._handle_errors(r, "PATCH " + uri)
        return r.json()

    def dataset(self, dataset_key):
        """
        Get information on a specific dataset from the repository at the schema matcher server.
        Args:
             dataset_key: integer which is the key of the dataset.

        Returns: dictionary.
        """
        logging.info('Sending request to the schema matcher server to get dataset info.')
        uri = self._dataset_endpoint(dataset_key)
        try:
            r = self.session.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("list_dataset", e)
        self._handle_errors(r, "GET " + uri)
        return r.json()

    def delete_dataset(self, dataset_key):
        """
        Delete a specific dataset from the repository at the schema matcher server.
        Args:
             dataset_key: int

        Returns:
        """
        logging.info('Sending request to the schema matcher server to delete dataset.')
        uri = self._dataset_endpoint(dataset_key)
        try:
            r = self.session.delete(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("delete_dataset", e)
        self._handle_errors(r, "DELETE " + uri)
        return r.json()

    def _dataset_endpoint(self, key):
        """The URI endpoint for the dataset at `key`"""
        return urljoin(self._uri_ds, str(key))

    def model_keys(self):
        """
        List ids of all models in the Model repository at the Schema Matcher server.

        Returns: list of model keys
        """
        logging.info('Sending request to the schema matcher server to list models.')
        try:
            r = self.session.get(self._uri_model)
        except Exception as e:
            logging.error(e)
            raise InternalError("model_keys", e)
        self._handle_errors(r, "GET " + self._uri_model)
        return r.json()

    def process_model_input(self,
                            feature_config=None,
                            description=None,
                            classes=None,
                            model_type=None,
                            labels=None,
                            cost_matrix=None,
                            resampling_strategy=None):

        # TODO: more sophisticated check of input...
        data = dict()
        if description:
            data["description"] = description
        if classes:
            data["classes"] = classes
        if model_type:
            data["modelType"] = model_type
        if labels:
            lab_str = dict([(str(key), str(val)) for key,val in labels.items()])
            data["labelData"] = lab_str
        if cost_matrix:
            data["costMatrix"] = cost_matrix
        if resampling_strategy:
            data["resamplingStrategy"] = resampling_strategy
        if feature_config:
            data["features"] = feature_config
        return data

    def post_model(self,
                   feature_config,
                   description="",
                   classes=None,
                   model_type="randomForest",
                   labels=None,
                   cost_matrix=None,
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

        Returns: model dictionary
        """

        if classes is None:
            classes = ["unknown"]

        logging.info('Sending request to the schema matcher server to post a model.')

        try:
            data = self.process_model_input(feature_config,
                                            description,
                                            classes,
                                            model_type,
                                            labels,
                                            cost_matrix,
                                            resampling_strategy)
            r = self.session.post(self._uri_model, json=data)
        except Exception as e:
            logging.error(e)
            raise InternalError("post_model", e)
        self._handle_errors(r, "POST " + self._uri_model)
        return r.json()

    def update_model(self,
                     model_key,
                     feature_config,
                     description,
                     classes,
                     model_type,
                     labels,
                     cost_matrix,
                     resampling_strategy):
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

        Returns: model dictionary
        """

        logging.info('Sending request to the schema matcher server to update model %d' % model_key)
        uri = self._model_endpoint(model_key)
        try:
            data = self.process_model_input(feature_config,
                                            description,
                                            classes,
                                            model_type,
                                            labels,
                                            cost_matrix,
                                            resampling_strategy)
            r = self.session.post(uri, json=data)
        except Exception as e:
            logging.error(e)
            raise InternalError("update_model", e)
        self._handle_errors(r, "PATCH " + uri)
        return r.json()

    def list_model(self, model_key):
        """
        Get information on a specific model in the model repository at the schema matcher server.
        Args:
             model_key: integer which is the key of the model in the repository

        Returns: dictionary
        """
        logging.info('Sending request to the schema matcher server to get model info.')
        uri = self._model_endpoint(model_key)
        try:
            r = self.session.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("list_model", e)
        self._handle_errors(r, "GET " + uri)
        return r.json()

    def delete_model(self, model_key):
        """
        Args:
             model_key: integer which is the key of the model in the repository

        Returns: dictionary
        """
        logging.info('Sending request to the schema matcher server to delete model.')
        uri = self._model_endpoint(model_key)
        try:
            r = self.session.delete(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("delete_model", e)
        self._handle_errors(r, "DELETE " + uri)
        return r.json()

    def train_model(self, model_key):
        """
        Args:
             model_key: integer which is the key of the model in the repository

        Returns: True

        """
        logging.info('Sending request to the schema matcher server to train the model.')
        uri = urljoin(self._model_endpoint(model_key), "train")
        try:
            r = self.session.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("train_model", e)
        self._handle_errors(r, "GET " + uri)
        return True

    def predict_model(self, model_key):
        """
        Post request to perform prediction based on the model.

        Args:
             model_key: integer which is the key of the model in the repository

        Returns: True

        """
        logging.info('Sending request to the schema matcher server to preform prediction based on the model.')
        uri = urljoin(self._model_endpoint(model_key), "predict")
        try:
            r = self.session.post(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("predict_model", e)
        self._handle_errors(r, "POST " + uri)
        return True

    def _model_endpoint(self, id):
        """Returns the endpoint url for the model `id`"""
        return urljoin(self._uri_model, str(id) + "/")

    # def get_model_predict(self, model_key):
    #     """
    #     Get predictions based on the model.
    #
    #     Args:
    #          model_key: integer which is the key of the model in the repository
    #
    #     Returns: dictionary
    #
    #     """
    #     logging.info('Sending request to the schema matcher server to get predictions based on the model.')
    #     uri = urljoin(urljoin(self.uri_model, str(model_key) + "/"), "predict")
    #     try:
    #         resp = self.session.get(uri)
    #     except Exception as e:
    #         logging.error(e)
    #         raise InternalError("get_model_predict", e)
    #     self.handle_errors(resp, "GET " + uri)
    #     return resp.json()


# class DataSet(object):
#     """
#         Attributes:
#             id: dataset id by which it is referred on the server
#             filename: name of the file
#             filepath: string which indicates the location of the file on the server
#             description: description of the file
#             date_created: date when the file was created
#             date_modified: last date when the dataset was modified
#             type_map: optional type map
#             sample: Pandas dataframe with data sample
#
#             _matcher
#             _column_map
#     """
#     def __init__(self, resp_dict, api):
#         """
#         Initialize instance of class MatcherDataset.
#
#         Args:
#             resp_dict: dictionary which is returned by Schema Matcher API
#             api: instance of class SchemaMatcher, it contains session as attribute
#
#         """
#         try:
#             self.id = int(resp_dict["id"])
#         except Exception as e:
#             logging.error("Failed to initialize MatcherDataset: dataset key could not be converted to integer.")
#             raise InternalError("MatcherDataset initialization", e)
#
#         self.filename = resp_dict['filename']
#         self.filepath = resp_dict['path']
#         self.description = resp_dict['description']
#         self.date_created = resp_dict['dateCreated']
#         self.date_modified = resp_dict['dateModified']
#         self.type_map = resp_dict['typeMap']
#
#         self._api = api
#
#         headers = []
#         data = []
#         self._column_map = dict() # mapping column ids to column names
#         for col in resp_dict['columns']:
#             headers.append(col['name'])
#             data.append(col['sample'])
#             self._column_map[int(col['id'])] = col['name']
#         self.sample = pd.DataFrame(data).transpose()  # TODO: define dtypes based on typeMap
#         self.sample.columns = headers
#
#     def __eq__(self,other):
#         if isinstance(other, DataSet):
#             return (self.id == other.id) and (self.filepath == other.filepath) \
#                    and (self.filename == other.filename) and (self.description == other.description) \
#                    and (self.date_modified == other.date_modified) \
#                    and (self.date_created == other.date_created) and (self.sample.equals(other.sample)) \
#                    and (self._column_map == other._column_map)
#         return False
#
#     def __ne__(self, other):
#         return not self.__eq__(other)
#
#     def _refresh(self, resp_dict):
#         """
#         Refresh instance attributes.
#
#         Args:
#             resp_dict: dictionary which is returned by Schema Matcher API
#
#         Returns: Nothing.
#         """
#         try:
#             self.id = int(resp_dict["id"])
#         except Exception as e:
#             logging.error("Failed to initialize MatcherDataset: dataset key could not be converted to integer.")
#             raise InternalError("MatcherDataset initialization", e)
#
#         self.filename = resp_dict['filename']
#         self.filepath = resp_dict['path']
#         self.description = resp_dict['description']
#         self.date_created = resp_dict['dateCreated']
#         self.date_modified = resp_dict['dateModified']
#         self.type_map = resp_dict['typeMap']
#
#         headers = []
#         data = []
#         self._column_map = dict() # mapping column ids to column names
#         for col in resp_dict['columns']:
#             headers.append(col['name'])
#             data.append(col['sample'])
#             self._column_map[int(col['id'])] = col['name']
#         self.sample = pd.DataFrame(data).transpose()  # TODO: define dtypes based on typeMap
#         self.sample.columns = headers
#
#     def __str__(self):
#         return "<MatcherDataset(" + str(self.id) + ")>"
#
#     def __repr__(self):
#         return self.__str__()
#
#     def delete(self):
#         """
#         Delete dataset in the dataset repository at the schema matcher server.
#         This method also updates attributes in the matcher instance.
#         It also destroys this instance.
#
#         Returns: Nothing.
#
#         """
#         logging.info("Deleting MatcherDataset %d" % self.id)
#         self._api.delete_dataset(self.id)
#         # delete references in self._matcher
#         # self._matcher._ds_keys
#         # self._matcher.datasets
#         # self._matcher.dataset_summary
#         # self._matcher._column_map
#         self._matcher._remove_dataset(self)
#
#         del self
#
#     def update(self, description=None, type_map=None):
#         """
#         Update the dataset with new description/type_map on the server.
#         If any of the parameters are empty, the current values of the dataset are used.
#
#         Args:
#             description: new description for the dataset
#             type_map: new type_map for the dataset
#
#         Returns: Nothing.
#
#         """
#         if description is None:
#             description = self.description
#         if type_map is None:
#             type_map = self.type_map
#         new_dict = self._api.update_dataset(self.id, description, type_map)
#         self._refresh(new_dict)
#         self._matcher._refresh_dataset(self)
#
#     def construct_labelData(self, filepath):
#         """
#         We want to construct a dictionary {column_id:class_label} for the dataset based on a .csv file.
#         This method reads in .csv as a Pandas data frame, selects columns "column_name" and "class",
#         drops NaN and coverts these two columns into dictionary.
#         We obtain a lookup dictionary
#         where the key is the column name and the value is the class label.
#         Then by using column_map the method builds the required dictionary.
#
#         Args:
#             filepath: string where .csv file is located.
#
#         Returns: dictionary
#
#         """
#         label_data = {}  # we need this dictionary (column_id, class_label)
#         try:
#             frame = pd.read_csv(filepath)
#             # dictionary (column_name, class_label)
#             name_labels = frame[["column_name", "class"]].dropna().set_index('column_name')['class'].to_dict()
#             for col_id, col_name in self._column_map.items():
#                 if col_name in name_labels:
#                     label_data[int(col_id)] = name_labels[col_name]
#         except Exception as e:
#             raise InternalError("construct_labelData", e)
#
#         return label_data
#
#
# class ModelState(object):
#     """
#     Class to wrap the model state.
#
#     Attributes:
#         status
#         message
#         date_created
#         date_modified
#     """
#     def __init__(self, status, message, date_modified):
#         """
#         Initialize instance of class ModelState.
#
#         Args:
#             status : string
#             date_created
#             date_modified
#         """
#         self.status = Status.to_status(status)  # convert to Status enum
#         self.message = message
#         self.date_modified = convert_datetime(date_modified)
#
#     def __repr__(self):
#         return "ModelState(" + repr(self.status)\
#                + ", modified on " + str(self.date_modified)\
#                + ", message " + repr(self.message) + ")"
#
#     def __eq__(self, other):
#         if isinstance(other, ModelState):
#             return self.status == other.status \
#                    and self.date_modified == other.date_modified
#         return False
#
#     def __ne__(self, other):
#         return not self.__eq__(other)
#
#     def __str__(self):
#         return self.__repr__()
#
#
# class Model(object):
#     """
#     Class to wrap the model.
#
#     Attributes:
#         id: model key by which the model is refered on the server
#         model_type: type of the model
#         description: description of the model
#         features_config: model configuration of the features
#         cost_matrix: cost matrix
#         resampling_strategy: resampling strategy of the model
#         ref_datasets: list of dataset ids which are referenced by labeled data
#         classes: list of semantic types
#         date_created: date when model was created
#         date_modified: last date when model was modified
#         model_state: information about model state
#
#         _label_data: internal
#         _matcher: internal
#         _column_map: internal
#
#         all_data: Pandas dataframe
#         features: Pandas dataframe
#         scores: Pandas dataframe
#         user_labels: Pandas dataframe
#
#     """
#
#     def __init__(self, resp_dict, api, column_map=None):
#         """
#         Args:
#             resp_dict : dictionary which is returned by Schema Matcher API
#             api: instance of class Session
#             column_map : optional dictionary for mapping columns fom ids to names
#         """
#         print(resp_dict)
#
#         try:
#             self.id = int(resp_dict["id"])
#         except Exception as e:
#             logging.error("Failed to initialize MatcherModel: model key could not be converted to integer.")
#             raise InternalError("MatcherModel initialization", e)
#
#         self.model_type = resp_dict["modelType"]
#         self.description = resp_dict["description"]
#         self.features_config = resp_dict["features"]
#         self.cost_matrix = resp_dict["costMatrix"]
#         self.resampling_strategy = resp_dict["resamplingStrategy"]
#         self._label_data = {} # this attribute is internal and should not be viewed/modified by the user
#
#         try:
#             # print(resp_dict["labelData"])
#             # print(type(resp_dict["labelData"]))
#             for col_id, lab in resp_dict["labelData"].items():
#                 self._label_data[int(col_id)] = lab
#         except Exception as e:
#             raise InternalError("failed to convert labelData", e)
#
#         self.ref_datasets = set(resp_dict["refDataSets"])
#         self.classes = resp_dict["classes"]
#         self.date_created = utils.convert_datetime(resp_dict["dateCreated"])
#         self.date_modified = utils.convert_datetime(resp_dict["dateModified"])
#         self.model_state = ModelState(resp_dict["state"]["status"],
#                                       resp_dict["state"]["message"],
#                                       resp_dict["state"]["dateChanged"])  #,
#                                       #resp_dict["state"]["dateModified"])
#
#         self._api = api  # this attribute is internal and should not be viewed/modified by the user
#         self._column_map = column_map  # this attribute is internal and should not be viewed/modified by the user
#
#         # create dataframe with user defined labels and predictions
#         self.all_data = self._label_data()  # column_map?
#         self.features = pd.DataFrame()  # empty data frame
#         self.scores = pd.DataFrame()  # empty data frame
#         self.user_labels = self.all_data.copy()
#
#     def __eq__(self, other):
#         if isinstance(other, Model):
#             return (self.id == other.id) and (self.model_type == other.model_type) \
#                    and (self.model_state == other.model_state) and (self.date_created == other.date_created) \
#                    and (self.date_modified == other.date_modified) and (self.classes == other.classes) \
#                    and (self.ref_datasets == other.ref_datasets) \
#                    and (self.features_config == other.features_config) and (self.description == other.description) \
#                    and (self.all_data.equals(other.all_data)) and (self.scores.equals(other.scores)) \
#                    and (self.features.equals(other.features)) and (self.user_labels.equals(other.user_labels))
#         return False
#
#     def __ne__(self, other):
#         return not self.__eq__(other)
#
#     def _session_update(self):
#         """
#         Download model updates from the API.
#         """
#         cur_model = self._api.list_model(self.id)
#         self._refresh(cur_model)
#
#     def train(self):
#         """
#         Send the training request to the API.
#
#         Args:
#             wait : boolean indicator whether to wait for the training to finish.
#
#         Returns: boolean -- True if model is trained, False otherwise
#
#         """
#         self._api.train_model(self.id) # launch training
#
#         # training is done if finished
#         def is_finished():
#             self._session_update()
#             return self.model_state.status in {Status.COMPLETE, Status.ERROR}
#
#         while not is_finished():
#             logging.info("Waiting for the training...")
#             time.sleep(5)  # wait for some time
#             self._session_update()
#
#         logging.info("Training complete.")
#         return self.model_state.status == Status.COMPLETE
#
#     def predict(self, wait=True, dataset_id=None):  # TODO: additional parameter for dataset_id
#         """
#         Get predictions based on the model.
#         Predictions are done for all datasets in the repository if dataset_id=None.
#         If a dataset id is passed to the method, prediction will be done only for this dataset.
#
#         Args:
#             wait : boolean indicator whether to wait for the training to finish, default is True.
#             dataset_id: dataset id for which the prediction should be done;
#                         if None, prediction is done for all datasets (this is default).
#
#         Returns: Pandas data framework.
#
#         """
#         train_status = self.train()  # do training
#         self._api.predict_model(self.id)  # launch prediction
#
#         self._session_update()
#         # predictions are available if finished
#         finished = self.model_state.status == Status.COMPLETE \
#                    or self.model_state.status == Status.ERROR
#
#         while wait and not finished:
#             time.sleep(3)  # wait for some time
#             # The sleep function is an OS call that differs from busy wait in that it doesn't block the thread.
#             # TODO: should we switch to threading.Event?
#             self._session_update()  # update model
#             if self.model_state.status == Status.ERROR or \
#                             self.model_state.status == Status.COMPLETE:
#                 finished = True  # finish waiting if prediction failed or got complete
#
#         if finished and self.model_state.status == Status.COMPLETE:
#             # prediction has successfully finished
#             resp_dict = self._api.get_model_predict(self.id)
#         elif self.model_state.status == Status.ERROR:
#             # either training or prediction failed
#             raise InternalError("Prediction/training failed", self.model_state.message)
#         elif self.model_state.status == Status.BUSY:
#             # either training or prediction is still ongoing
#             raise InternalError("Prediction/training still in progress", self.model_state.message)
#         else:
#             raise InternalError("Training has not been launched", self.model_state.message)
#
#         self.all_data = self._process_predictions(resp_dict)
#         self.scores = self._scores()
#         self.features = self._get_features()
#         return self.all_data
#
#     def _process_predictions(self, response):
#         """
#         Process column predictions into Pandas data framework.
#
#         Args:
#             response: list of column predictions (dictionaries) which is returned by the schema matcher API
#
#         Returns: Pandas data framework.
#
#         """
#         headers = ["model_id", "column_id", "column_name", "dataset_id", "user_label", "predicted_label", "confidence"]
#         classes = []
#         feature_names = []
#         # additionally scores for all classes and features
#         data = []
#
#         for col_pred in response:
#             # if column_map is available, lookup the name of the column
#             if self._column_map:
#                 column_name = self._column_map.get(col_pred["columnID"], np.nan)
#             else:
#                 column_name = np.nan
#             # if user provided label for this column, get the label, otherwise NaN
#             actual_lab = self._label_data.get(str(col_pred["columnID"]), np.nan)
#             conf = col_pred["confidence"]
#             if conf > 0:
#                 label = col_pred["label"]
#             else:
#                 label = np.nan # training actually failed...
#             new_row = (col_pred["id"], col_pred["columnID"], column_name, col_pred["datasetID"],
#                        actual_lab, label, conf)
#             if len(classes) == 0:  # get names of classes
#                 classes = col_pred["scores"].keys()
#             if len(feature_names) == 0: # get names of features
#                 feature_names = col_pred["features"].keys()
#
#             new_row += tuple(col_pred["scores"].values()) # get scores for all classes
#             new_row += tuple(col_pred["features"].values())  # get features which were used for prediction
#             data.append(new_row)
#
#         if not len(data):
#             logging.warning("Predictions are not available!")
#             return self.all_data
#         all_data = pd.DataFrame(data)
#         # TODO: what should be done if the data is empty?
#         all_data.columns = headers + list(classes) + list(feature_names)
#         return all_data
#
#     def confusion_matrix(self):
#         """
#         Calculate confusion matrix of the model.
#         If all_data is not available for the model, it will return None.
#
#         Returns: Pandas data frame.
#
#         """
#         if self.all_data.empty:
#             logging.warning("Model all_data is empty. Confusion matrix cannot be calculated.")
#             return None
#
#         # take those rows where user_label is available
#         try:
#             available = self.all_data[["user_label", "predicted_label"]].dropna()
#         except Exception as e:
#             raise InternalError("confusion_matrix", e)
#         y_actu = pd.Series(available["user_label"], name='Actual')
#         y_pred = pd.Series(available["predicted_label"], name='Predicted')
#         return pd.crosstab(y_actu, y_pred)
#
#     def _label_data(self):
#         """
#         Creates a Pandas dataframe which contains user specified labels for columns.
#
#         Returns: Pandas dataframe with columns
#                 "model_id", "column_id", "column_name", "user_label".
#
#         """
#         headers = ["model_id", "column_id", "column_name", "user_label"]
#         # TODO: add dataset id ???
#         data = []
#         for columnID, label in self._label_data.items():
#             if self._column_map:
#                 column_name = self._column_map.get(columnID, np.nan)
#             else:
#                 column_name = np.nan
#             data.append((self.id, columnID, column_name, label))
#
#         sample = pd.DataFrame(data).dropna()
#         sample.columns = headers
#         return sample
#
#     def _scores(self):
#         """
#         Get a slice of all_data which has scores for classes.
#
#         Returns: Pandas data framework
#
#         """
#         # just get a slice of all_data
#         # all_data has columns:
#         headers = ["model_id", "column_id", "column_name", "dataset_id", "user_label", "predicted_label",
#                    "confidence"] + self.classes
#         if len(set(headers).intersection(set(self.all_data.columns))) == len(headers):
#             return self.all_data[headers]
#         logging.warning("Scores are not available for model: " + str(self.id))
#         return pd.DataFrame()
#
#     def _get_features(self):
#         """
#         Get a slice of all_data which has features calculated for columns in the dataset repo.
#
#         Returns: Pandas data framework
#
#         """
#         # just get a slice of all_data
#         headers = list(set(self.all_data.columns).difference(self.classes))
#         return self.all_data[headers]
#
#     def _refresh(self, resp_dict, column_map=None):
#         """
#         Update the model according to the newly provided parameters.
#
#         Args:
#             resp_dict: dictionary with new parameters
#             column_map: dictionary of column ids
#
#         Returns: Nothing.
#
#         """
#         # TODO: check resp_dict
#         try:
#             self.id = int(resp_dict["id"])
#         except Exception as e:
#             logging.error("Failed to initialize MatcherModel: model key could not be converted to integer.")
#             raise InternalError("MatcherModel initialization", e)
#         self.model_type = resp_dict["modelType"]
#         self.features_config = resp_dict["features"]
#         self.cost_matrix = resp_dict["costMatrix"]
#         self.description = resp_dict["description"]
#         self.resampling_strategy = resp_dict["resamplingStrategy"]
#         self._label_data = resp_dict["labelData"]
#         self.ref_datasets = set(resp_dict["refDataSets"])
#         self.classes = resp_dict["classes"]
#         self.date_created = utils.convert_datetime(resp_dict["dateCreated"])
#         self.date_modified = utils.convert_datetime(resp_dict["dateModified"])
#         self.model_state = ModelState(resp_dict["state"]["status"],
#                                       resp_dict["state"]["message"],
#                                       resp_dict["state"]["dateModified"])
#         if column_map: # update column_map if it's provided
#             self._column_map = column_map
#
#         # refresh _matcher
#         self._matcher._refresh_model(self)
#
#         # reset all_data, scores and features
#         self.all_data = self._label_data()
#         self.scores = pd.DataFrame()
#         self.features = pd.DataFrame()
#         self.user_labels = self.all_data.copy()
#
#     def add_labels(self, additional_labels):
#         """
#         Add label data to the model and upload the changes to the server.
#
#         Args:
#             additional_labels : dictionary with additional labels for columns
#
#         Returns: Nothing.
#
#         """
#         # labels in additional_labels extend (in case of conflict override) the ones which are currently defined
#         self._label_data.update(additional_labels)
#         self.update(labels=self._label_data)
#
#     def update(self,
#                feature_config=None,
#                description=None,
#                classes=None,
#                model_type=None,
#                labels=None,
#                cost_matrix=None,
#                resampling_strategy=None):
#         """
#         Update model in the model repository at the schema matcher server.
#         If any of the parameters is None, this method will take the current model value for it.
#
#         Args:
#             feature_config : dictionary
#             description : string which describes the model to be posted
#             classes : list of class names
#             model_type : string
#             labels : dictionary
#             cost_matrix : cost matrix
#             resampling_strategy : string
#
#         Returns:
#         """
#         # if any of the parameters is None, then take the current one
#         if feature_config is None:
#             feature_config = self.features_config
#         if description is None:
#             description = self.description
#         if classes is None:
#             classes = self.classes
#         if model_type is None:
#             model_type = self.model_type
#         if labels is None:
#             labels = self._label_data
#         if cost_matrix is None:
#             cost_matrix = self.cost_matrix
#         if resampling_strategy is None:
#             resampling_strategy = self.resampling_strategy
#         # send model patch request to the schema matcher API
#         new_dict = self._api.update_model(
#             self.id,
#             feature_config,
#             description,
#             classes,
#             model_type,
#             labels,
#             cost_matrix,
#             resampling_strategy
#         )
#         # refresh model
#         self._refresh(new_dict)
#
#     def delete(self):
#         """
#         Delete model in the model repository at the schema matcher server.
#         This method also updates attributes in the matcher instance.
#         It also destroys this instance.
#         """
#         logging.info("Deleting MatcherModel %d" % self.id)
#         self._api.delete_model(self.id)
#         #self._matcher._remove_model(self)
#
#         del self
#
#     def info(self):
#         """Construct a string which summarizes all model parameters."""
#         return "\n".join([
#             "model_key: {}".format(self.id),
#             "model_type: {}".format(self.model_type),
#             "features_config: {}".format(self.features_config),
#             "cost_matrix: {}".format(self.cost_matrix),
#             "resampling_strategy: {}".format(self.resampling_strategy),
#             "label_data: {}".format(self._label_data),
#             "ref_datasets: {}".format(self.ref_datasets),
#             "classes: {}".format(self.classes),
#             "date_created: {}".format(self.date_created),
#             "date_modified: {}".format(self.date_modified),
#             "model_state: {}".format(self.model_state)
#         ])
#
#     def __str__(self):
#         """Show summary of the model"""
#         return self.info()
#
#     def __repr__(self):
#         """Different from the original docs"""
#         return "<MatcherModel({})>".format(self.id)
#
#
# if __name__ == "__main__":
#
#     sess = Session("127.0.0.1", 8080)
#     all_ds = sess.dataset_keys()
#     all_models = sess.model_keys()
