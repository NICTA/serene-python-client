"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the museum benchmark
"""
import os
import logging
from pprint import pprint
import json

from serene.elements import SSD, Ontology, DataSet, Octopus
from serene.endpoints import DataSetEndpoint, OntologyEndpoint, SSDEndpoint, OctopusEndpoint, ModelEndpoint
from ..utils import TestWithServer
from serene.elements.semantics.base import KARMA_DEFAULT_NS

# logging functions...
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
ch.setFormatter(formatter)

_logger.addHandler(ch)



class TestMuseum(TestWithServer):
    """
    Tests the JsonReader/JsonWriter for SSD
    """
    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._datasets = None
        self._ontologies = None

        benchmark_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "museum_benchmark")
        self._museum_owl_dir = os.path.join(benchmark_path, "owl")
        self._museum_data = os.path.join(benchmark_path, 'dataset')
        self._museum_ssd = os.path.join(benchmark_path, 'ssd')

    @staticmethod
    def bind_karma_ssd(dataset, ssd_json, default_ns, new_file):
        """
        Modifies ssd json to include proper column ids.
        This method binds ssd_json to the dataset through the column names.

        :param ssd_json: location of the file with ssd json
        :param ontologies: list of ontologies to be used by ssd
        :param default_ns: default namespace to be used for nodes and properties
        :return: json dict
        """
        with open(ssd_json) as f:
            data = json.load(f)

        def find_name(column_names, attr_name):
            for c in column_names:
                if attr_name in c:
                    return c
            return attr_name

        attributes = data["attributes"]
        mappings = data["mappings"]

        column_map = {c.name: c.id for c in dataset.columns}
        attr_name_map = {attr["name"]: find_name(column_map.keys(), attr["name"]) for attr in attributes}
        attr_map = {attr["id"]: attr_name_map[attr["name"]] for attr in attributes}

        new_attributes = []
        # change ids in attributes
        for attr in attributes:
            attr_name = attr_name_map[attr["name"]]
            if attr_name not in column_map:
                new_attributes.append(attr)
            else:
                attr["id"] = column_map[attr_name]
                attr["name"] = attr_name
                attr["columnIds"] = [column_map[attr_map[c]] for c in attr["columnIds"]]
                new_attributes.append(attr)

        new_mappings = []
        # change ids in mappings
        for map in mappings:
            if attr_map[map["attribute"]] not in column_map:
                new_mappings.append(map)
            else:
                map["attribute"] = column_map[attr_map[map["attribute"]]]
                new_mappings.append(map)

        # specify namespaces in nodes and links
        nodes = data["semanticModel"]["nodes"]
        links = data["semanticModel"]["links"]
        new_nodes = []
        new_links = []
        for node in nodes:
            if "prefix" not in node:
                node["prefix"] = default_ns
            new_nodes.append(node)

        for link in links:
            if "prefix" not in link:
                link["prefix"] = default_ns
            new_links.append(link)

        # specify proper ids of ontologies
        data["mappings"] = new_mappings
        data["attributes"] = new_attributes
        if len(new_attributes) < 1 or len(mappings) < 1:
            logging.warning("There are no attributes or mappings in ssd!")
            data["semanticModel"]["nodes"] = []
            data["semanticModel"]["links"] = []
        else:
            data["semanticModel"]["nodes"] = new_nodes
            data["semanticModel"]["links"] = new_links

        _logger.info("Writing to file {}".format(new_file))
        with open(new_file, "w+") as f:
            f.write(json.dumps(data, indent=4, sort_keys=True))

        return data


    def _add_museum(self):
        """
        Helper method to add museum benchmark files to the server!
        :return:
        """
        # add ontologies
        ontologies = []
        for path in os.listdir(self._museum_owl_dir):
            f = os.path.join(self._museum_owl_dir, path)
            ontologies.append(self._ontologies.upload(f))

        # add datasets with their ssds
        for ds in os.listdir(self._museum_data):
            ds_f = os.path.join(self._museum_data, ds)
            ds_name = os.path.splitext(ds)[0]
            ssd_f = os.path.join(self._museum_ssd, ds_name + ".ssd")

            _logger.info("Adding dataset: " + str(ds_f))
            dataset = self._datasets.upload(ds_f, description="museum_benchmark")

            _logger.info("Adding ssd: " + str(ssd_f))
            new_json = dataset.bind_ssd(ssd_f, ontologies, KARMA_DEFAULT_NS)
            # _logger.info("+++++++++++ obtained ssd json")
            # _logger.info(new_json)
            if len(new_json["mappings"]) < 1:
                _logger.info("Skipping this file!")
                continue
            else:
                empty_ssd = SSD(dataset, ontologies)
                # _logger.debug("empty created")
                ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)
                self._ssds.upload(ssd)

    def setUp(self):
        self._datasets = DataSetEndpoint(self._session)
        self._ontologies = OntologyEndpoint(self._session)
        self._ssds = SSDEndpoint(self._session, self._datasets, self._ontologies)
        self._lobsters = ModelEndpoint(session=self._session,
                                       dataset_endpoint=self._datasets)
        self._octopii = OctopusEndpoint(self._session,
                                        model_endpoint=self._lobsters,
                                        ontology_endpoint=self._ontologies,
                                        ssd_endpoint=self._ssds)
        self._clear_storage()
        self._add_museum()

        _logger.info("SetUp complete")


    def _clear_storage(self):
        """Removes all server elements"""
        for ssd in self._ssds.items:
            self._ssds.remove(ssd)

        for ds in self._datasets.items:
            self._datasets.remove(ds)

        for on in self._ontologies.items:
            self._ontologies.remove(on)


    def tearDown(self):
        """
        Be sure to remove all storage elements once a test is finished...
        :return:
        """
        pass
        # self._clear_storage()


    # def test_set_up(self):
    #     """
    #     Just check setUp
    #     :return:
    #     """
    #     self.assertEqual(len(self._ssds.items), 29)
    #     self.assertEqual(len(self._datasets.items), 29)
    #     self.assertEqual(len(self._ontologies.items), 11)


    def test_predict_s16(self):
        """
        Predict for s16
        :return:
        """
        # predict for s16, use the rest as training
        dataset = [ds for ds in self._datasets.items if "s16" in ds.filename]
        print("******************")
        print(dataset)

        octo_local = Octopus(
            ssds=self._ssds.items,
            ontologies=self._ontologies.items,
            name='octopus-test',
            description='Testing example for s16 using museum benchmark',
            resampling_strategy="NoResampling",  # optional
            num_bags=100,  # optional
            bag_size=10,  # optional
            model_type="randomForest",
            feature_config={
                "activeFeatures": [
                    "num-unique-vals",
                    "prop-unique-vals",
                    "prop-missing-vals",
                    "ratio-alpha-chars",
                    "prop-numerical-chars",
                    "prop-whitespace-chars",
                    "prop-entries-with-at-sign",
                    "prop-entries-with-hyphen",
                    "prop-range-format",
                    "is-discrete",
                    "entropy-for-discrete-values"
                ],
                "activeFeatureGroups": [
                    "inferred-data-type",
                    "stats-of-text-length",
                    "stats-of-numeric-type",
                    "prop-instances-per-class-in-knearestneighbours",
                    "mean-character-cosine-similarity-from-class-examples",
                    "min-editdistance-from-class-examples",
                    "min-wordnet-jcn-distance-from-class-examples",
                    "min-wordnet-lin-distance-from-class-examples"
                ],
                "featureExtractorParams": [
                    {
                        "name": "prop-instances-per-class-in-knearestneighbours",
                        "num-neighbours": 3
                    }, {
                        "name": "min-editdistance-from-class-examples",
                        "max-comparisons-per-class": 3
                    }, {
                        "name": "min-wordnet-jcn-distance-from-class-examples",
                        "max-comparisons-per-class": 3
                    }, {
                        "name": "min-wordnet-lin-distance-from-class-examples",
                        "max-comparisons-per-class": 3
                    }
                ]
            }
        )

        octo_server = self._octopii.upload(octo_local)
        print("*******************")
        print(octo_server)
        print("*******************")

        self.fail()
