"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Abstract model for semantic labelling/typing.
Evaluation will be based on this abstract model.
"""
import logging
import os

from serene.core import SchemaMatcher
from serene.exceptions import InternalError
from serene.utils import construct_labelData
from karmaDSL.api import KarmaSession
import time


domains = ["soccer", "dbpedia", "museum", "weather"]
benchmark = {
    "soccer": ['bundesliga-2015-2016-rosters.csv', 'world_cup_2010_squads.csv',
               'fifa-soccer-12-ultimate-team-data-player-database.csv', 'world_cup_2014_squads.csv',
               'all_world_cup_players.csv', 'mls_players_2015.csv', 'world_cup_squads.csv',
               'players.csv', '2014 WC french.csv', 'uefa_players.csv', 'world_cup_player_ages.csv',
               'WM 2014 Alle Spieler - players fifa.csv'],
    "dbpedia": ['s5.txt', 's2.txt', 's6.txt',
                's3.txt', 's4.txt', 's9.txt',
                's8.txt', 's7.txt', 's10.txt', 's1.txt'],
    "museum": ['s02-dma.csv', 's26-s-san-francisco-moma.json', 's27-s-the-huntington.json',
               's16-s-hammer.xml', 's20-s-lacma.xml', 's15-s-detroit-institute-of-art.json',
               's28-wildlife-art.csv', 's04-ima-artworks.xml', 's25-s-oakland-museum-paintings.json',
               's29-gilcrease.csv', 's05-met.json', 's13-s-art-institute-of-chicago.xml',
               's14-s-california-african-american.json', 's07-s-13.json', 's21-s-met.json',
               's12-s-19-artworks.json', 's08-s-17-edited.xml', 's19-s-indianapolis-artworks.xml',
               's11-s-19-artists.json', 's22-s-moca.xml', 's17-s-houston-museum-of-fine-arts.json',
               's18-s-indianapolis-artists.xml', 's23-s-national-portrait-gallery.json',
               's03-ima-artists.xml', 's24-s-norton-simon.json', 's06-npg.json',
               's09-s-18-artists.json', 's01-cb.csv'],
    "weather": ['w1.txt', 'w3.txt', 'w2.txt', 'w4.txt']
}

class SemanticTyper(object):
    """
    Fixed for 4 domains for now
    """

    allowed_sources = []
    for sources in benchmark.values():
        # only these sources can be used for training or testing by different classifiers of SemanticTyper
        allowed_sources += sources

    def __init__(self, model_type, description=""):
        self.model_type = model_type
        self.description = description

    def reset(self):
        pass

    def define_training_data(self, train_sources, train_labels):
        pass

    def train(self):
        pass

    def predict(self, source):
        pass

    def __str__(self):
        return "<SemanticTyper: model_type={}, description={}>".format(self.model_type, self.description)


class DINTModel(SemanticTyper):
    """
    Wrapper for DINT schema matcher.
    """
    def __init__(self, schema_matcher, feature_config, resampling_strategy, description):
        """
        Initializes DINT model with the specified feature configuration and resampling strategy.
        The model gets created at schema_matcher server.
        :param schema_matcher: SchemaMatcherSession instance
        :param feature_config: Dictionary with feature configuration for DINT
        :param resampling_strategy: Resampling strategy
        :param description: Description
        """
        logging.info("Initializing DINT model.")
        if not(type(schema_matcher) is SchemaMatcher):
            logging.error("DINTModel init: SchemaMatcher instance required.")
            raise InternalError("DINTModel init", "SchemaMatcher instance required")

        super().__init__("DINTModel", description=description)

        self.server = schema_matcher
        self.feature_config = feature_config
        self.resampling_strategy = resampling_strategy
        self.classifier = None

    def reset(self):
        """
        Reset the model to the blank untrained state.
        We accomplish this by first deleting the existing model from the schema matcher server and then creating the
        fresh model at the server.
        :return:
        """
        logging.info("Resetting DINTModel.")
        if self.classifier:
            self.server.remove_model(self.classifier)
        # TODO: remove datasets?
        self.classifier = None

    def define_training_data(self, train_sources, train_labels=None):
        """

        :param train_sources: List of sources. For now it's hard coded for the allowed_sources.
        :param train_labels: List of files with labels, optional. If not present, the default labels will be used.
        :return:
        """
        logging.info("Defining training data for DINTModel.")
        label_dict = {}
        for idx, source in enumerate(train_sources):
            if source not in self.allowed_sources:
                logging.warning("Source '{}' not in allowed_sources. Skipping it.".format(source))
                continue
            # upload source to the schema matcher server
            try:
                matcher_dataset = self.server.create_dataset(file_path=os.path.join("data", "sources", source+".csv"),
                                                             description="traindata",
                                                             type_map={})
            except:
                logging.warning("Ugly fix for tiny dataset upload...")
                # FIXME: InMemoryFileUpload fails!!!
                # ugly fix for this problem: we add empty rows to the file
                filepath = os.path.join("data", "sources", source+".csv")
                with open(filepath) as f:
                    headers = f.readline()
                    num = len(headers.split(","))
                empty_line = ','.join(["" for _ in range(num)])
                empty_lines = '\n'.join([empty_line for _ in range(10000)])
                with open(filepath, 'a') as f:
                    f.write(empty_lines)
                matcher_dataset = self.server.create_dataset(file_path=os.path.join("data", "sources", source + ".csv"),
                                                             description="traindata",
                                                             type_map={})

            # construct dictionary of labels for the uploaded dataset
            try:
                label_dict.update(construct_labelData(matcher_dataset, train_labels[idx]))
            except:
                # in case train_labels are not provided, we take default labels from the benchmark
                label_dict.update(construct_labelData(matcher_dataset,
                                                      filepath=os.path.join("data", "labels", source+".columnmap.txt"),
                                                      header_column="column_name",
                                                      header_label="semantic_type"))
            logging.debug("DINT model label_dict updated: {}".format(label_dict))

        # create model on the server with the labels specified
        logging.debug("Creating model on the DINT server with proper config.")
        classes = list(set(label_dict.values())) + ["unknown"]
        self.classifier = self.server.create_model(self.feature_config,
                                                   classes=classes,
                                                   description=self.description,
                                                   labels=label_dict,
                                                   resampling_strategy=self.resampling_strategy)
        return True

    def train(self):
        """
        Train DINTModel and return True.
        :return:
        """
        logging.info("Training DINTModel.")
        # TODO: track run time
        return self.classifier.train()

    def predict(self, source):
        """
        Prediction with DINTModel for the source.
        :param source:
        :return:
        """
        # TODO: track run time
        logging.info("Predicting with DINTModel for source {}".format(source))
        if source not in self.allowed_sources:
            logging.warning("Source '{}' not in allowed_sources. Skipping it.".format(source))
            return None
        # upload source to the schema matcher server
        matcher_dataset = self.server.create_dataset(file_path=os.path.join("data", "sources", source + ".csv"),
                                                     description="testdata",
                                                     type_map={})
        return self.classifier.predict(matcher_dataset)


class KarmaDSLModel(SemanticTyper):
    """
    Wrapper for Karma DSL semantic labeller.
    KarmaDSL server can hold only one model.
    """
    def __init__(self, karma_session, description):
        logging.info("Initializing KarmaDSL model.")
        if not (type(karma_session) is KarmaSession):
            logging.error("KarmaDSLModel init: KarmaSession instance required.")
            raise InternalError("KarmaDSLModel init", "KarmaSession instance required")

        super().__init__("KarmaDSL", description=description)
        self.karma_session = karma_session
        self.karma_session.reset_semantic_labeler() # we immediately reset their semantic labeller
        self.folder_names = None
        self.train_sizes = None

    def reset(self):
        self.karma_session.reset_semantic_labeler()
        self.folder_names = None
        self.train_sizes = None

    def define_training_data(self, train_sources, train_labels=None):
        """

        :param train_sources:
        :param train_labels: is not used here...
        :return:
        """
        # copy train_sources
        logging.info("Defining train data for KarmaDSL: {}".format(
            self.karma_session.post_folder("train_data", train_sources)))
        self.folder_names = ["train_data"]
        self.train_sizes = [len(train_sources)-1]  # TODO: check that it should be exactly this
        return True

    def train(self):
        # TODO: track run time
        if self.folder_names and self.train_sizes:
            logging.info("Training KarmaDSL...")
            self.karma_session.train_model(self.folder_names, self.train_sizes)
            return True
        logging.error("KarmaDSL cannot be trained since training data is not specified.")
        raise InternalError("KarmaDSL train", "training data absent")

    def predict(self, source):
        """

        :param source:
        :return:
        """
        # TODO: track run time
        resp = self.karma_session.post_folder("test_data", [source])
        logging.info("Posting source {} to karma dsl server: {}".format(source,resp))
        return self.karma_session.predict_folder("test_data")


class NNetModel(SemanticTyper):
    def __init__(self, description):
        super().__init__("NNetModel", description=description)

    def reset(self):
        pass

    def define_training_data(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass


if __name__ == "__main__":

    # setting up the logging
    log_file = 'benchmark.log'
    logging.basicConfig(filename=os.path.join('data', log_file),
                        level=logging.DEBUG, filemode='w+',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    #******* setting up DINTModel
    dm = SchemaMatcher(host="localhost", port=8080)
    # dictionary with features
    feature_config = {"activeFeatures": ["num-unique-vals", "prop-unique-vals", "prop-missing-vals"],
                "activeFeatureGroups": ["stats-of-text-length", "prop-instances-per-class-in-knearestneighbours"],
                "featureExtractorParams": [
                    {"name": "prop-instances-per-class-in-knearestneighbours", "num-neighbours": 5}]
                }
    # resampling strategy
    resampling_strategy = "ResampleToMean"
    dint_model = DINTModel(dm, feature_config, resampling_strategy, "DINTModel with ResampleToMean")

    #******** setting up KarmaDSL model
    # dsl = KarmaSession(host="localhost", port=8000)
    # dsl_model = KarmaDSLModel(dsl, "default KarmaDSL model")

    # train/test
    train_sources = benchmark["soccer"][:-1]
    test_source = [benchmark["soccer"][-1]]
    print("# sources in train: %d" % len(train_sources))
    print("# sources in test: %d" % len(test_source))

    print("Define training data DINT %r" % dint_model.define_training_data(train_sources))
    # print("Define training data KarmaDSL %r" % dsl_model.define_training_data(train_sources))
    # print("Train dsl %r" % dsl_model.train())

    # print(dsl_model.predict(test_source[0]))

    print("Train dint %r" % dint_model.train())
    print(dint_model.predict(test_source[0]))



