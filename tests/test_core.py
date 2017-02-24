"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Tests the core module
"""
import unittest2 as unittest
import serene


class TestSemanticModeller(unittest.TestCase):
    """
    Tests the SemanticModeller class
    """
    def test_junk(self):
        sm = serene.Serene(
            config={
                "kmeans": 5,
                "threshold": 0.7,
                "search-depth": 100
            },
            ontologies=[
                "/some/file/something.owl",
                "/some/other/file/something.owl"
            ]
        )
        self.assertEqual('a', 'a')

    def test_init(self):
        pass

    def test_add_ontology(self):
        pass

    def test_to_ssd(self):
        pass

    def test_flatten(self):
        pass

    def test_class_nodes(self):
        pass

    def test_data_nodes(self):
        pass

    def test_links(self):
        pass

    def test_ontologies(self):
        pass


class TestSemanticSourceDesc(unittest.TestCase):
    """
    Testing the SemanticSourceDesc object...
    """

    def test_init__(self):
        pass

    def test_find(self):
        pass

    def test_map(self):
        pass

    def test_link(self):
        pass

    def test_sample(self):
        pass

    def test_remove(self):
        pass

    def test_predict(self):
        pass

    def test_show(self):
        pass

    def test_save(self):
        pass

    def test_version(self):
        pass

    def test_ontologies(self):
        pass

    def test_ssd(self):
        pass

    def test_json(self):
        pass

    def test_model(self):
        pass

    def test_predictions(self):
        pass

    def test_mappings(self):
        pass

    def test_columns(self):
        pass

    def test_transforms(self):
        pass

    def test_data_nodes(self):
        pass

    def test_class_nodes(self):
        pass

    def test_links(self):
        pass

    def test_repr(self):
        pass
