# Python wrapper for the Schema Matcher API

# external
import requests
import logging
import os
import datetime
from urllib.parse import urljoin

# project
from config import settings as conf_set

class SchemaMatcherSession(object):
    """
        Attributes:
    """
    def __init__(self):
        logging.info('Initialising session to connect to the schema matcher server.')
        self.session = requests.Session()
        self.session.trust_env = conf_set.schema_matcher_server['trust_env']
        self.session.auth = conf_set.schema_matcher_server['auth']
        self.session.cert = conf_set.schema_matcher_server['cert']

        # uri to send requests to
        self.uri = conf_set.schema_matcher_server['uri']
        self.uri_ds = urljoin(self.uri, 'dataset')

    def list_alldatasets(self):
        """

        :return: list of dataset keys
        """
        logging.info('Sending request to the schema matcher server to list datasets.')
        try:
            r = self.session.get(self.uri_ds)
        except Exception as e:
            logging.error(e)
            return
        if r.status_code != 200:
            logging.error('Error occurred during request: status_code=' + str(r.status_code) +
                          ', uri=' + self.uri_ds)
            return
        return r.json()

    def post_dataset(self):
        """
            Args:
                 dataset_key: int

            :return:
            """
        pass

    def list_dataset(self, dataset_key):
        """
        Args:
             dataset_key: int

        :return: dictionary
        """
        logging.info('Sending request to the schema matcher server to get dataset info.')
        try:
            r = self.session.get(urljoin(self.uri_ds, str(dataset_key)))
        except Exception as e:
            logging.error(e)
            return
        if r.status_code != 200:
            logging.error('Error occurred during request: status_code=' + str(r.status_code) +
                          ', uri=' + urljoin(self.uri_ds, str(dataset_key)))
            return
        return r.json()

class SchemaMatcher(object):
    """
        Attributes:
    """

    def __init__(self):
        logging.info('Initialising schema matcher class object.')
        self.session = SchemaMatcherSession()

        self.ds_keys = self.session.list_alldatasets()  # list of keys of all available datasets

    def populate(self):
        pass


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.abspath(os.curdir))

    # setting up the logging
    log_file = 'schemamatcher_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S0') + '.log'
    logging.basicConfig(filename=os.path.join(proj_dir, 'data', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    sess = SchemaMatcherSession()
    all_ds = sess.list_alldatasets()

    for ds_key in all_ds:
        print('type', type(ds_key))
        info = sess.list_dataset(ds_key)
        print('info_type', type(info))
        print(info)

