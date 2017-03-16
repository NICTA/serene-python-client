"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Defines the DataSet API
"""
import logging
from urllib.parse import urljoin

from .exceptions import InternalError
from .http import HTTPObject
from ..utils import flatten


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

