"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Schema Matcher wrapper around the Serene API backend schema matcher endpoints
"""
import logging
from urllib.parse import urljoin

import requests

from .http import HTTPObject
from .data_api import DataSetAPI
from .model_api import ModelAPI
from .ontology_api import OntologyAPI
from .ssd_api import SsdAPI
from .octopus_api import OctopusAPI
from .exceptions import InternalError


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

        self.ontology_api = OntologyAPI(self._uri, self.session)
        self.dataset_api = DataSetAPI(self._uri, self.session)
        self.model_api = ModelAPI(self._uri, self.session)
        self.ssd_api = SsdAPI(self._uri, self.session)
        self.octopus_api = OctopusAPI(self._uri, self.session)

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

            logging.info("Connection to the Serene server has been established.")
            print("Connection to the Serene server established: {}".format(root))

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

    def compare(self, compare_json):
        """
        Compares two SSDs.
        This method can be used to evaluate the performance of the semantic modeler.
        :param compare_json: json dict
        :return: precision, recall, jaccard metrics of comparison for two sets of RDF triplets constructed from x and y
        """
        logging.debug('Sending request to the Serene server to compare two SSDs')
        logging.debug('request: ' + str(compare_json))
        uri = urljoin(self._uri, 'evaluate')

        try:
            r = self.session.post(uri, data=compare_json)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to perform evaluation", e)

        self._handle_errors(r, "POST " + uri)

        return r.json()


