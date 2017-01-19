"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Schema Matcher wrapper around the Serene API backend schema matcher endpoints
"""
import logging
import requests

from ..exceptions import BadRequestError, NotFoundError, OtherError, InternalError
from urllib.parse import urljoin


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

        self._uri_ds = urljoin(self._uri, 'dataset/')  # uri for the dataset endpoint
        self._uri_model = urljoin(self._uri, 'model/')  # uri for the model endpoint

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
            raise InternalError("dataset_keys", e)
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

        assert "unknown" in classes

        logging.info('Sending request to the schema matcher server to post a model.')

        try:
            data = self._process_model_input(feature_config,
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
                     feature_config=None,
                     description=None,
                     classes=None,
                     model_type=None,
                     labels=None,
                     cost_matrix=None,
                     resampling_strategy=None):
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
            data = self._process_model_input(feature_config,
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
        self._handle_errors(r, "POST " + uri)
        return r.json()

    def model(self, model_key):
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
            r = self.session.post(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("train_model", e)
        self._handle_errors(r, "POST " + uri)
        return True

    def predict_model(self, model_key, dataset_key=None):
        """
        Post request to perform prediction based on the model.

        Args:
            model_key: integer which is the key of the model in the repository
            dataset_key: integer key for the dataset to predict

        Returns: True

        """
        logging.info('Sending request to the schema matcher server to preform prediction based on the model.')
        uri = urljoin(self._model_endpoint(model_key), "predict/")
        try:
            if dataset_key is None:
                r = self.session.post(uri)
            else:
                r = self.session.post(urljoin(uri, str(dataset_key)))
        except Exception as e:
            logging.error(e)
            raise InternalError("predict_model", e)
        self._handle_errors(r, "POST " + uri)
        return r.json()

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

    def _dataset_endpoint(self, key):
        """The URI endpoint for the dataset at `key`"""
        return urljoin(self._uri_ds, str(key))

    def _process_model_input(self,
                             feature_config=None,
                             description=None,
                             classes=None,
                             model_type=None,
                             labels=None,
                             cost_matrix=None,
                             resampling_strategy=None):
        """Prepares a json update string for the model update request"""
        data = {}
        if description:
            data["description"] = description
        if classes:
            data["classes"] = classes
        if model_type:
            data["modelType"] = model_type
        if labels:
            lab_str = {str(key): str(val) for key, val in labels.items()}
            data["labelData"] = lab_str
        if cost_matrix:
            data["costMatrix"] = cost_matrix
        if resampling_strategy:
            data["resamplingStrategy"] = resampling_strategy
        if feature_config:
            data["features"] = feature_config
        return data

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

    def _model_endpoint(self, id):
        """Returns the endpoint url for the model `id`"""
        return urljoin(self._uri_model, str(id) + "/")

    def __repr__(self):
        return "<Session at (" + str(self._uri) + ")>"

    def __str__(self):
        return self.__repr__()
