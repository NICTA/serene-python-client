"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Wrapper around the Karma DSL (schema matcher) backend
"""
import logging
import requests
import os
import datetime

from serene.exceptions import BadRequestError, NotFoundError, OtherError, InternalError
from urllib.parse import urljoin


class KarmaSession(object):
    """
    This is the class which sets up the session for the Karma DSL api.
    It provides wrappers for all calls to the API.
    This is an internal class and should not be explicitly used/called by the user.

        Attributes:
            session: Session instance with configuration parameters for the schema matcher server from config.py
            _uri: general uri for the Karma DSL api
            _uri_ds: uri for the dataset endpoint of the schema matcher API
            _uri_model: uri for the model endpoint of the schema matcher API
    """
    def __init__(self, host, port=8000,
                 auth=None, cert=None, trust_env=None):
        """
        Initialize and maintain a session with the Serene backend
        :param host: The address of the Serene backend
        :param port: The port number; default is 8000
        :param auth: The authentication token (None if non-secure connection)
        :param cert: The security certificate
        :param trust_env: The trusted environment token
        """
        logging.info('Initialising session to connect to the Karma DSL server.')

        self.session = requests.Session()
        self.session.trust_env = trust_env
        self.session.auth = auth
        self.session.cert = cert

        # root URL for the Karma DSL backend
        root = "http://{}:{}/".format(host, port)

        # this will throw an error an fail the init if not available...
        logging.info(self._test_connection(root))
        self._uri = root

    def list_indexed_folders(self):
        """

        Returns: list of folder names.
        Raises: InternalDIError on failure.
        """
        logging.info('Sending request to KarmaDSL to list folders.')
        uri = urljoin(self._uri, 'folder')
        try:
            r = self.session.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("indexed_folders", e)
        self._handle_errors(r, "GET " + uri)
        return r.json()['folder_names']

    def reset_semantic_labeler(self):
        """
        Resets semantic labeler on Karma DSL server.
        Returns: surprise.
        Raises: InternalDIError on failure.
        """
        logging.info('Sending request to KarmaDSL to reset.')
        uri = urljoin(self._uri, 'reset')
        try:
            r = self.session.post(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("indexed_folders", e)
        self._handle_errors(r, "POST " + uri)
        return r.json()

    def post_folder(self, folder_name, file_list):
        """
        Create a new folder on the server with the specified list of files.
        Args:
             folder_name: name of the new folder
             file_list: list of file names to be included in the folder;
                        these files must be already on the server; they will be only copied.
        Returns: Dictionary.
        """

        logging.info('Sending request to KarmaDSL server to post folder.')
        copy_uri = urljoin(self._uri, 'copy')
        try:
            data = {"folder": folder_name, "files": file_list}
            r = self.session.post(copy_uri, json=data)
        except Exception as e:
            logging.error(e)
            raise InternalError("post_folder", e)
        self._handle_errors(r, "POST " + copy_uri)
        return r.json()

    def train_model(self, folder_names, train_sizes):
        """
        Args:
             folder_names: names of folders where training data is lying
             train_sizes: how to do training

        Returns: True

        """
        logging.info('Sending request to KarmaDSL to train.')
        # first we need to train semantic types
        uri = urljoin(self._uri, "semantic_type")
        try:
            data = {"folder": folder_names}
            r = self.session.get(uri, json=data)
            logging.info("Train semantic types resp: {}".format(r))
        except Exception as e:
            logging.error(e)
            raise InternalError("train semantic types for KarmaDSL", e)
        # now we train the model
        uri = urljoin(self._uri, "train")
        try:
            data = {"folder": folder_names, "size": train_sizes}
            r = self.session.post(uri, json=data)
        except Exception as e:
            logging.error(e)
            raise InternalError("train KarmaDSL", e)
        self._handle_errors(r, "POST " + uri)
        return True

    def predict_folder(self, folder_name):
        """
        Predict semantic types for all sources in the specified folder.
        This folder should already exist on the server.

        Args:
            folder_name: name of the folder where sources for prediction are lying.
        Returns:
            dict with predictions

        """
        logging.info('Sending request to the KarmaDSL to preform prediction for the folder.')
        uri = urljoin(self._uri, "predict")
        try:
            data = {"folder": folder_name}
            r = self.session.get(uri, json=data)
        except Exception as e:
            logging.error(e)
            raise InternalError("predict_model", e)
        self._handle_errors(r, "GET " + uri)
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
            logging.info("Server {}".format(root))
            logging.info(type(req))

            # attempt to decode the version...
            version = req.json()

            logging.info("Connection to server established: {}".format(root))
            print("Connection to server established: {}".format(root))

            return version

        except requests.exceptions.RequestException as e:
            msg = "Failed to connect to Karma DSL {}".format(root)
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

    def __repr__(self):
        return "<KarmaDSL session at (" + str(self._uri) + ")>"

    def __str__(self):
        return self.__repr__()

if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.abspath(os.curdir))

    # setting up the logging
    log_file = 'karmadsl_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S0') + '.log'
    logging.basicConfig(filename=os.path.join(proj_dir, 'data', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    dsl = KarmaSession(host="localhost", port=8000)
    print(dsl.list_indexed_folders())
    print(dsl.reset_semantic_labeler())

    print(dsl.post_folder("train_data", ["s1.txt", "s2.txt", "s3.txt", "s4.txt"]))

    dsl.train_model(["train_data"], [1]) # weird stuff happening

