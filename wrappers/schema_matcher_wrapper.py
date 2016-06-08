# Python wrapper for the Schema Matcher API

# external
import requests
import logging
import os
import datetime
from urllib.parse import urljoin
import pandas as pd

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
        self.uri_ds = urljoin(self.uri, 'dataset') + '/'

    def __str__(self):
        return "<SchemaMatcherSession at (" + str(self.uri) + ")>"

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
        uri = urljoin(self.uri_ds, str(dataset_key))
        try:
            r = self.session.get(uri)
        except Exception as e:
            logging.error(e)
            return
        if r.status_code != 200:
            logging.error('Error occurred during request: status_code=' + str(r.status_code) +
                          ', uri=' + uri)
            return
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
        for col in self.columns:
            headers.append(col['name'])
            data.append(col['sample'])
        self.data_sample = pd.DataFrame(data).transpose()  # TODO: define dtypes based on typeMap
        self.data_sample.columns = headers

    def __str__(self):
        return "<MatcherDataset(" + str(self.ds_key) + ")>"

    def __repr__(self):
        return self.__str__()

class SchemaMatcher(object):
    """
        Attributes:

    """

    def __init__(self):
        """

        """
        logging.info('Initialising schema matcher class object.')
        self.session = SchemaMatcherSession()

        self.ds_keys = self.session.list_alldatasets()  # list of keys of all available datasets
        self.populate_datasets()

    def __str__(self):
        return "<SchemaMatcherAt(" + str(self.session.uri) + ")>"

    def __repr__(self):
        return self.__str__()

    def populate_datasets(self):
        """
        Args:

        :return:
        """
        self.dataset = dict()
        for ds_key in self.ds_keys:
            self.dataset[ds_key] = MatcherDataset(self.session.list_dataset(ds_key))


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.abspath(os.curdir))

    # setting up the logging
    log_file = 'schemamatcher_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S0') + '.log'
    logging.basicConfig(filename=os.path.join(proj_dir, 'data', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    sess = SchemaMatcherSession()
    all_ds = sess.list_alldatasets()

    dm = SchemaMatcher()

    print(dm)
    print(dm.dataset)

