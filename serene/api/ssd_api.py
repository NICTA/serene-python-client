import logging
from urllib.parse import urljoin

from .exceptions import InternalError
from .http import HTTPObject


class SsdAPI(HTTPObject):
    """
    Handles the SSD endpoint requests
    """
    def __init__(self, uri, conn):
        """
        Requires a valid session object

        :param uri: The base URI for the Serene server
        :param conn: The live connection object
        """
        self.connection = conn
        self._uri = urljoin(uri, 'ssd/')

    def keys(self):
        """
        List ids of all SSDs in the dataset repository at the Schema Matcher server.
        If the connection fails, empty list is returned and connection error is logged.

        Returns: list of SSD keys.
        Raises: InternalDIError on failure.
        """
        logging.debug('Sending request to the schema matcher server to list SSDs.')

        uri = self._uri

        try:
            r = self.connection.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to fetch SSD keys", e)
        self._handle_errors(r, "GET " + self._uri)
        return r.json()

    def post(self, json):
        """
        Post a new SSD to the Serene server.
        Args:
             json: parsed json from the SSD file
        Returns: Dictionary.
        """

        logging.debug('Sending request to the schema matcher server to post an SSD.')

        uri = self._uri

        try:
            r = self.connection.post(uri, data=json)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to post SSD", e)

        self._handle_errors(r, "POST " + uri)

        return r.json()

    def update(self, key, json):
        """
        Update an existing SSD in the repository on the Serene server.
        Args:
            key: The key ID of the SSD on the server
            json: The partial json update body

        :return:
        """
        logging.debug('Sending request to the schema matcher server to update SSD %d' % key)
        uri = urljoin(self._uri, str(key))

        try:
            r = self.connection.post(uri, data=json)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to update SSD", e)

        self._handle_errors(r, "PATCH " + uri)

        return r.json()

    def item(self, key):
        """
        Get information on a specific SSD from the repository on the Serene server.
        Args:
             key: integer which is the key of the SSD.

        Returns: dictionary.
        """
        logging.debug('Sending request to the Serene server to get SSD info.')
        uri = urljoin(self._uri, str(key))
        try:
            r = self.connection.get(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to get SSD", e)
        self._handle_errors(r, "GET " + uri)
        return r.json()

    def delete(self, key):
        """
        Delete a specific SSD from the repository on the Serene server.
        Args:
             key: int

        Returns:
        """
        logging.debug('Sending request to the Serene server to delete SSD.')
        uri = urljoin(self._uri, str(key))
        try:
            r = self.connection.delete(uri)
        except Exception as e:
            logging.error(e)
            raise InternalError("Failed to delete SSD", e)
        self._handle_errors(r, "DELETE " + uri)
        return r.json()