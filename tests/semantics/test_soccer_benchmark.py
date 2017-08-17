"""
Copyright (C) 2017 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the soccer benchmark
"""
import os
import logging
import tarfile
import csv

from serene.elements import SSD
from serene.endpoints import DataSetEndpoint, OntologyEndpoint, SSDEndpoint
from ..utils import TestWithServer
from serene.elements.semantics.base import KARMA_DEFAULT_NS

# logging functions...
_logger = logging.getLogger()
_logger.setLevel(logging.DEBUG)


class TestSoccer(TestWithServer):
    """
    Tests the JsonReader/JsonWriter for SSD
    """

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._datasets = None
        self._ontologies = None

        self._benchmark_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "soccer")
        self._soccer_owl_dir = os.path.join(self._benchmark_path, "owl")
        self._soccer_data = os.path.join(self._benchmark_path, 'dataset')
        self._soccer_ssd = os.path.join(self._benchmark_path, 'ssd')

    def setUp(self):
        self._datasets = DataSetEndpoint(self._session)
        self._ontologies = OntologyEndpoint(self._session)
        self._ssds = SSDEndpoint(self._session, self._datasets, self._ontologies)
        self._clear_storage()

        _logger.debug("SetUp complete")

    def tearDown(self):
        """
        Be sure to remove all storage elements once a test is finished...
        :return:
        """
        # self._clear_storage()
        pass

    def _add_owls(self):
        # add ontologies
        ontologies = []
        for path in os.listdir(self._soccer_owl_dir):
            f = os.path.join(self._soccer_owl_dir, path)
            ontologies.append(self._ontologies.upload(f))
        return ontologies

    def _add_datasets(self, ontologies):


        # add datasets with their ssds
        for ds in os.listdir(self._soccer_data):
            if ds.endswith(".gz"):
                continue  # skip archive
            ds_f = os.path.join(self._soccer_data, ds)
            # ds_name = os.path.splitext(ds)[0]
            ds_name = ds
            ssd_f = os.path.join(self._soccer_ssd, ds_name + ".ssd")

            _logger.debug("Adding dataset: {}".format(ds_f))
            dataset = self._datasets.upload(ds_f, description="soccer_benchmark")

            _logger.debug("Adding ssd: {}".format(ssd_f))
            new_json = dataset.bind_ssd(ssd_f, ontologies, KARMA_DEFAULT_NS)
            empty_ssd = SSD(dataset, ontologies)
            ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)
            self._ssds.upload(ssd)
            # we remove the csv dataset
            # os.remove(ds_f)


    def _clear_storage(self):
        """Removes all server elements"""
        for ssd in self._ssds.items:
            self._ssds.remove(ssd)

        for ds in self._datasets.items:
            self._datasets.remove(ds)

        for on in self._ontologies.items:
            self._ontologies.remove(on)

    def test_evaluate_soccer(self):
        """
        Test that soccer benchmark can be uploaded!
        :return:
        """
        ontologies = self._add_owls()
        self._add_datasets(ontologies)
        self.assertEqual(len(self._ssds.items), 12)
        self.assertEqual(len(self._datasets.items), 12)
        self.assertEqual(len(self._ontologies.items), 1)

    def test_mappings(self):
        """
        Get label data for soccer benchmark files
        :return:
        """
        ontologies = self._add_owls()
        self._add_datasets(ontologies)
        self.assertEqual(len(self._ssds.items), 12)
        self.assertEqual(len(self._datasets.items), 12)
        self.assertEqual(len(self._ontologies.items), 1)

        for ssd in self._ssds.items:
            print("name: ", ssd.name)
            print("mappings: ", ssd.mappings)
            csvF = ssd.name + ".columnmap.txt"
            with open(csvF, "w+") as csvfile:
                csvwriter = csv.writer(csvfile, dialect="excel")
                csvwriter.writerow(["key", "column_name", "source_name", "semantic_type"])
                for key, value in ssd.mappings.items():
                    label = key.class_node.label + "---" + key.label
                    column = value.name
                    print("column: ", column, ", label:", label)
                    csvwriter.writerow([column, column, ssd.name, label])

        self.fail()

    def test_s27(self):
        """

        :return:
        """
        ontologies = self._add_owls()
        self.assertEqual(len(self._ontologies.items), 1)
        self._add_datasets(ontologies)
        self.assertEqual(len(self._ssds.items), 12)
        self.assertEqual(len(self._datasets.items), 12)

        # first grab the JSON object from the dataset...
        datasets = self._datasets.items
        s27_ds = ""
        for ds in datasets:
            if "s27-s-the-huntington.json" in ds.filename:
                s27_ds = ds

        # secondly grab the reference SSD
        s27_ssd = os.path.join(self._benchmark_path, "ssd", "s27-s-the-huntington.json.ssd")

        # bind the dataset and ssd together...
        new_json = s27_ds.bind_ssd(s27_ssd, ontologies, KARMA_DEFAULT_NS)

        # create a new ssd and update with the reference JSON
        empty_ssd = SSD(s27_ds, ontologies)
        ssd = empty_ssd.update(new_json, self._datasets, self._ontologies)

        # upload to the server and ensure it is parsed correctly...
        f = self._ssds.upload(ssd)
        self.assertEqual(set(ssd.mappings), set(f.mappings))
