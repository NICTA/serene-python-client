"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Schema Matcher wrapper around the Serene API backend schema matcher endpoints
"""
import logging
import requests
from enum import Enum, unique

from urllib.parse import urljoin
from .exceptions import BadRequestError, NotFoundError, OtherError, InternalError


class HTTPObject(object):
    """
    Basic HTTP helper methods
    """
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

        msg = response.json()['message']

        if response.status_code == 400:
            log_msg("BadRequest", response.status_code, expr, response)
            raise BadRequestError(expr, msg)
        elif response.status_code == 404:
            log_msg("NotFound", response.status_code, expr, response)
            raise NotFoundError(expr, msg)
        else:
            log_msg("RequestError", response.status_code, expr, response)
            raise OtherError(response.status_code, expr, msg)

    @staticmethod
    def join_urls(*args):
        """Crude url joiner"""
        return '/'.join(args)

class Session(HTTPObject):
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
        logging.info('Initialising session to connect to Serene.')

        self.session = requests.Session()
        self.session.trust_env = trust_env
        self.session.auth = auth
        self.session.cert = cert

        # root URL for the Serene backend
        root = "http://{}:{}".format(host, port)

        # this will throw an error an fail the init if not available...
        self.version = self._test_connection(root)

        self.host = host
        self.port = port

        self._uri = urljoin(root, self.version + '/')

        #self._uri_ds = urljoin(self._uri, 'dataset/')  # uri for the dataset endpoint
        #self._uri_model = urljoin(self._uri, 'model/')  # uri for the model endpoint

        self.ontology = OntologyAPI(self._uri, self.session)
        self.dataset = DataSetAPI(self._uri, self.session)
        self.model = ModelAPI(self._uri, self.session)

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

    @property
    def uri(self):
        return self._uri

    def __repr__(self):
        return "<Session at (" + str(self._uri) + ")>"

    def __str__(self):
        return self.__repr__()


class DataSetAPI(HTTPObject):
    """
    Handles the DataSet endpoint requests
    """
    def __init__(self, uri, conn):
        """
        Requires a valid session object

        :param uri: The base URI for the Serene server
        :param conn: The live connection object
        """
        self.connection = conn
        self._uri = urljoin(uri, 'dataset/')

    def keys(self):
        """
        List ids of all datasets in the dataset repository at the Schema Matcher server.
        If the connection fails, empty list is returned and connection error is logged.

        Returns: list of dataset keys.
        Raises: InternalDIError on failure.
        """
        logging.debug('Sending request to the schema matcher server to list datasets.')

        uri = self._uri

        try:
            r = self.connection.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to fetch dataset keys", e)
        self._handle_errors(r, "GET " + self._uri)
        return r.json()

    def post(self, description, file_path, type_map):
        """
        Post a new dataset to the Serene server.
        Args:
             description: string which describes the dataset to be posted
             file_path: string which indicates the location of the dataset to be posted
             type_map: dictionary with type map for the dataset

        Returns: Dictionary.
        """

        logging.debug('Sending request to the schema matcher server to post a dataset.')

        uri = self._uri

        try:
            f = {"file": open(file_path, "rb")}
            data = {
                "description": str(description),
                "typeMap": type_map
            }
            r = self.connection.post(uri, data=data, files=f)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to post dataset", e)

        self._handle_errors(r, "POST " + uri)

        return r.json()

    def update(self, key, description, type_map):
        """
        Update an existing dataset in the repository on the Serene server.
        Args:
             description: string which describes the dataset to be posted
             key: integer dataset id
             type_map: dictionary with type map for the dataset

        :return:
        """
        logging.debug('Sending request to the schema matcher server to update dataset %d' % key)
        uri = urljoin(self._uri, str(key))

        try:
            data = {"description": description, "typeMap": type_map}
            r = self.connection.post(uri, data=data)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to update dataset", e)

        self._handle_errors(r, "PATCH " + uri)

        return r.json()

    def item(self, key):
        """
        Get information on a specific dataset from the repository on the Serene server.
        Args:
             key: integer which is the key of the dataset.

        Returns: dictionary.
        """
        logging.debug('Sending request to the Serene server to get dataset info.')
        uri = urljoin(self._uri, str(key))
        try:
            r = self.connection.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to get dataset", e)
        self._handle_errors(r, "GET " + uri)
        return r.json()

    def delete(self, key):
        """
        Delete a specific dataset from the repository on the Serene server.
        Args:
             key: int

        Returns:
        """
        logging.debug('Sending request to the Serene server to delete dataset.')
        uri = urljoin(self._uri, str(key))
        try:
            r = self.connection.delete(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to delete dataset", e)
        self._handle_errors(r, "DELETE " + uri)
        return r.json()


class ModelAPI(HTTPObject):
    """
    Handles requests for the Model Endpoint
    """
    def __init__(self, uri, conn):
        """
        Requires a valid session object

        :param uri: The base URI for the Serene server
        :param conn: The live connection object
        """
        self.connection = conn
        self._uri = urljoin(uri, 'model/')

    def post(self,
             feature_config,
             description="",
             classes=None,
             model_type="randomForest",
             labels=None,
             cost_matrix=None,
             resampling_strategy="ResampleToMean"):
        """
        Post a new model to the schema matcher server.

        :param feature_config: dictionary
        :param description: string which describes the model to be posted
        :param classes: list of class names
        :param model_type: string
        :param labels: dictionary
        :param cost_matrix:
        :param resampling_strategy: string

        :return model dictionary
        """
        if classes is None:
            classes = ["unknown"]

        assert "unknown" in classes

        logging.debug('Sending request to the schema matcher server to post a model.')
        uri = self._uri

        try:
            data = self._process_model_input(feature_config,
                                             description,
                                             classes,
                                             model_type,
                                             labels,
                                             cost_matrix,
                                             resampling_strategy)
            r = self.connection.post(uri, json=data)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to create model", e)
        self._handle_errors(r, "POST " + uri)
        return r.json()

    def update(self,
               key,
               feature_config=None,
               description=None,
               classes=None,
               model_type=None,
               labels=None,
               cost_matrix=None,
               resampling_strategy=None):
        """
        Update an existing model in the model repository at the schema matcher server.

        :param key: integer which is the key of the model in the repository
        :param feature_config: dictionary
        :param description: string which describes the model to be posted
        :param classes: list of class names
        :param model_type: string
        :param labels: dictionary
        :param cost_matrix:
        :param resampling_strategy: string

        :return model dictionary
        """

        logging.debug('Sending request to the schema matcher server to update model %d' % key)
        uri = urljoin(self._uri, str(key))
        try:
            data = self._process_model_input(feature_config,
                                             description,
                                             classes,
                                             model_type,
                                             labels,
                                             cost_matrix,
                                             resampling_strategy)

            r = self.connection.post(uri, json=data)
        except Exception as e:
            logging.error(e)
            raise InternalError("update_model", e)
        self._handle_errors(r, "POST " + uri)
        return r.json()

    def item(self, key):
        """
        Get information on a specific model in the model repository at the schema matcher server.

        :param key: integer which is the key of the model in the repository

        :return: dictionary
        """
        logging.debug('Sending request to the schema matcher server to get model info.')
        uri = urljoin(self._uri, str(key))
        try:
            r = self.connection.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to get model", e)
        self._handle_errors(r, "GET " + uri)
        return r.json()

    def delete(self, key):
        """
        :param key: integer which is the key of the model in the repository

        :returns: dictionary
        """
        logging.debug('Sending request to the Serene server to delete model.')
        uri = urljoin(self._uri, str(key))

        try:
            r = self.connection.delete(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to delete model", e)

        self._handle_errors(r, "DELETE " + uri)
        return r.json()

    def train(self, key):
        """
        :param key: integer which is the key of the model in the repository

        :return: True, this function is asynchronous
        """
        logging.debug('Sending request to the Serene server to train the model.')
        uri = urljoin(self._uri, self.join_urls(str(key), "train"))

        try:
            r = self.connection.post(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to train model", e)

        self._handle_errors(r, "POST " + uri)
        return True

    def predict(self, key, dataset_key):
        """
        Post request to perform prediction based on the model, using the dataset `dataset_key`
        as input.

        Args:
            key: integer which is the key of the model in the repository
            dataset_key: integer key for the dataset to predict

        Returns: True

        """
        logging.debug('Sending request to the Serene server to preform '
                      'a prediction based on the model.')
        uri = urljoin(self._uri, self.join_urls(str(key), "predict", str(dataset_key)))

        try:
            r = self.connection.post(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to predict Model", e)

        self._handle_errors(r, "POST " + uri)

        return r.json()

    def keys(self):
        """
        List ids of all models in the Model repository at the Schema Matcher server.

        Returns: list of model keys
        """
        logging.debug('Sending request to the schema matcher server to list models.')
        uri = self._uri

        try:
            r = self.connection.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to get Model keys", e)

        self._handle_errors(r, "GET " + uri)

        return r.json()

    @staticmethod
    def _process_model_input(feature_config=None,
                             description=None,
                             classes=None,
                             model_type=None,
                             labels=None,
                             cost_matrix=None,
                             resampling_strategy=None):
        """Prepares a json update string for the model update request"""
        data = {
            "description": description,
            "classes": classes,
            "modelType": model_type,
            "labelData": {str(key): str(val) for key, val in labels.items()},
            "costMatrix": cost_matrix,
            "resamplingStrategy": resampling_strategy,
            "features": feature_config
        }
        return {k: v for k, v in data.items() if v is not None}


@unique
class OwlFormat(Enum):
    TURTLE = 'turtle'
    JSON = 'jsonld'
    RDF = 'rdfxml'
    OWL = 'owl'


class OntologyAPI(HTTPObject):
    """
    Handles the Ontology endpoint requests
    """
    def __init__(self, uri, conn):
        """
        Requires a valid session object

        :param uri: The base URI of the Serene server
        :param conn: The live connection object
        """
        self.connection = conn
        self._uri = urljoin(uri, 'owl/')
        self.OWL_FORMATS = {x.value for x in OwlFormat}

    def keys(self):
        """
        List ids of all ontologies in the repository at the Serene server.
        If the connection fails, empty list is returned and connection error is logged.

        Returns: list of ontology keys.
        Raises: InternalDIError on failure.
        """
        logging.debug('Sending request to the Serene server to list ontologies.')
        try:
            r = self.connection.get(self._uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("ontology keys", e)
        self._handle_errors(r, "GET " + self._uri)
        return r.json()

    def item(self, key):
        """
        Get information on a specific ontology from the repository on the Serene server.
        Args:
             key: integer which is the key of the Ontology.

        Returns: dictionary.
        """
        logging.debug('Sending request to Serene server to get the ontology info.')
        uri = urljoin(self._uri, str(key))
        try:
            r = self.connection.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to list ontology", e)

        self._handle_errors(r, "GET " + uri)

        return r.json()

    def post(self, description, file_path, owl_format):
        """
        Post a new ontology to the Serene server.
        Args:
             description: string which describes the ontology to be posted
             file_path: string which indicates the location of the OWL file
             owl_format: type of ontology format
                     e.g. 'turtle', 'jsonld', 'rdfxml', 'owl'
        Returns: Dictionary.
        """

        logging.debug('Sending request to the schema matcher server to post a dataset.')
        uri = self._uri

        if owl_format not in self.OWL_FORMATS:
            msg = "Ontology format value {} is not supported. " \
                  "Use one of: {}".format(owl_format, self.OWL_FORMATS)
            raise ValueError(msg)
        try:
            f = {"file": open(file_path, "rb")}
            data = {
                "description": str(description),
                "format": owl_format
            }
            r = self.connection.post(uri, data=data, files=f)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to create ontology", e)

        self._handle_errors(r, "POST " + uri)

        return r.json()

    def update(self, key, description=None, file_path=None, owl_format=None):
        """
        Update an existing Ontology in the repository on the Serene server.
        Args:
             key: integer ontology id
             description: string which describes the dataset to be posted
             type_map: dictionary with type map for the dataset

        :return:
        """
        logging.debug('Sending request to the schema matcher server to post a dataset.')
        uri = urljoin(self._uri, str(key))

        if owl_format is not None and owl_format not in self.OWL_FORMATS:
            msg = "Ontology format value {} is not supported. " \
                  "Use one of: {}".format(owl_format, self.OWL_FORMATS)
            raise ValueError(msg)
        try:
            if file_path is None:
                f = None
            else:
                f = {"file": open(file_path, "rb")}

            data = {
                "description": str(description) if description is not None else None,
                "format": owl_format
            }
            data = {k: v for k, v in data.items() if v is not None}

            r = self.connection.post(uri, data=data, files=f)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to update ontology", e)

        self._handle_errors(r, "POST " + uri)

        return r.json()

    def delete(self, key):
        """
        Delete a specific ontology from the repository at the Serene server.
        Args:
             key: int

        Returns:
        """
        logging.debug('Sending request to the schema matcher server to delete dataset.')
        uri = urljoin(self._uri, str(key))

        try:
            r = self.connection.delete(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("delete_dataset", e)

        self._handle_errors(r, "DELETE " + uri)

        return r.json()
