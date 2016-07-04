# data integration python client

# external
import logging
import os
import datetime

# project
from wrappers import data_quality_wrapper
from wrappers import schema_matcher_wrapper as SMW


class SchemaMatcher(object):
    """
        Attributes:

    """

    def __init__(self):
        """
        Initialize
        """
        logging.info('Initialising schema matcher class object.')
        self.session = SMW.SchemaMatcherSession()

        self.ds_keys = self.get_dataset_keys()   # list of keys of all available datasets
        self.model_keys = self.get_model_keys()  # list of keys of all available models
        # self.dataset = self.populate_datasets()

    def __str__(self):
        return "<SchemaMatcherAt(" + str(self.session.uri) + ")>"

    def __repr__(self):
        return self.__str__()

    def get_dataset_keys(self):
        """Obtain the list of keys of all available datasets."""
        return self.session.list_alldatasets()

    def get_model_keys(self):
        """Obtain the list of keys of all available models."""
        return self.session.list_allmodels()

    def populate_datasets(self):
        """
        Construct MatcherDataset per each dataset key.

        :return: Dictionary for now
        """
        dataset = dict()
        for ds_key in self.ds_keys:
            dataset[ds_key] = SMW.MatcherDataset(self.session.list_dataset(ds_key))
        return dataset


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.abspath(os.curdir))

    # setting up the logging
    log_file = 'schemamatcher_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S0') + '.log'
    logging.basicConfig(filename=os.path.join(proj_dir, 'data', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    sess = SMW.SchemaMatcherSession()
    all_ds = sess.list_alldatasets()
    all_models = sess.list_allmodels()

    model = sess.list_model(all_models[0])
    features_conf = model["features"]

    sess.post_model(feature_config="")

    dm = SchemaMatcher()

    print(dm)
    print(dm.dataset)