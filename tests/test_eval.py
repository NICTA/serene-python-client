"""This is a test script of model evaluation module."""

import unittest2 as unittest
from collections import namedtuple
from serene.matcher.core import SchemaMatcher
from serene.matcher.eval import CrossDatasetEvaluation
from serene.matcher.eval import CrossColumnEvaluation
from serene.matcher.eval import load_specs_from_dir


class TestEval(unittest.TestCase):
    """This class tests the model evaluation module."""

    DATA_PATH = "tests/resources/realestate"

    def test_eval(self):
        """Test cross-columns evaluation."""
        Model = namedtuple('Model', 'description features resampling_strategy')

        model = Model(
            description='evaluation model',
            features={
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
                        "name":
                            "prop-instances-per-class-in-knearestneighbours",
                        "num-neighbours": 3
                    }, {
                        "name": "min-editdistance-from-class-examples",
                        "max-comparisons-per-class": 20
                    }, {
                        "name": "min-wordnet-jcn-distance-from-class-examples",
                        "max-comparisons-per-class": 20
                    }, {
                        "name": "min-wordnet-lin-distance-from-class-examples",
                        "max-comparisons-per-class": 20
                    }
                ]
            },
            resampling_strategy="ResampleToMean"
        )

        specs = load_specs_from_dir(self.DATA_PATH)

        dm = SchemaMatcher(
            host="localhost",
            port=8080)

        results = {}

        evaluation = CrossDatasetEvaluation(model, *specs, dm)
        results['cross-dataset'] = evaluation.evaluate()

        evaluation = CrossColumnEvaluation(model, *specs, dm)
        results['cross-column'] = evaluation.evaluate(k=5)

        print(results)

        self.assertEqual(1, 1)