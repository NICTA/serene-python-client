from functools import partial
from io import TextIOBase

from mock import MagicMock, Mock, call, patch
from requests import Response, Session
from unittest2 import TestCase

from serene.api.data_api import DataSetAPI
from serene.api.exceptions import InternalError
from serene.api.model_api import ModelAPI
from serene.api.octopus_api import OctopusAPI
from serene.api.ontology_api import OntologyAPI, OwlFormat
from serene.api.ssd_api import SsdAPI
from serene.elements import DataSet, Ontology, SSD


class TestDataSetAPI(TestCase):
    def setUp(self):
        self.connection = Mock(Session())
        self.root_uri = "http://localhost/"
        self.dataset_path = "dataset/"
        self.uri = self.root_uri + self.dataset_path
        self.api = DataSetAPI(self.root_uri, self.connection)
        self.response = Mock(Response())

        self.description = "this python file"
        self.file_path = __file__
        self.type_map = {"a": "int"}

        self.update_description = "another python file"
        self.update_type_map = {"b": "string"}

    def test_keys(self):
        keys = [1, 2]
        self.response.status_code = 200
        self.response.json = Mock(return_value=keys)
        self.connection.get = Mock(return_value=self.response)
        result = self.api.keys()

        self.assertEqual(result, keys)
        self.connection.get.assert_called_with(self.uri)

    def test_keys_with_connection_exception(self):
        self.connection.get = Mock(side_effect=Exception)

        self.assertRaises(InternalError, self.api.keys)

    def test_post(self):
        message = "Created"
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.post = Mock(return_value=self.response)
        result = self.api.post(self.description, self.file_path, self.type_map)

        args = self.connection.post.call_args
        self.assertEqual(args[0][0], self.uri)
        self.assertEqual(
            args[1]["data"],
            {"description": self.description, "typeMap": self.type_map})
        self.assertIsNotNone(args[1]["files"])
        self.assertEqual(result, message)

    def test_post_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)
        api_post = partial(
            self.api.post, self.description, self.file_path, self.type_map)

        self.assertRaises(InternalError, api_post)

    def test_update(self):
        message = "Updated"
        key = 1
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.post = Mock(return_value=self.response)
        result = self.api.update(
            key, self.update_description, self.update_type_map)

        args = self.connection.post.call_args
        self.assertEqual(args[0][0], self.uri + str(key))
        self.assertEqual(
            args[1]["data"],
            {
                "description": self.update_description,
                "typeMap": self.update_type_map
            })
        self.assertEqual(result, message)

    def test_update_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)
        api_update = partial(
            self.api.update, 1, self.update_description, self.update_type_map)

        self.assertRaises(InternalError, api_update)

    def test_item(self):
        key = 1
        item = {
            "description": self.description,
            "typeMap": self.type_map
        }
        self.response.status_code = 200
        self.response.json = Mock(return_value=item)
        self.connection.get = Mock(return_value=self.response)
        result = self.api.item(key)

        self.assertEqual(result, item)
        self.connection.get.assert_called_with(self.uri + str(key))

    def test_item_with_connection_exception(self):
        self.connection.get = Mock(side_effect=Exception)
        api_item = partial(self.api.item, 1)

        self.assertRaises(InternalError, api_item)

    def test_delete(self):
        message = "Deleted"
        key = 1
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.delete = Mock(return_value=self.response)
        result = self.api.delete(key)

        self.assertEqual(result, message)
        self.connection.delete.assert_called_with(self.uri + str(key))

    def test_delete_with_connection_exception(self):
        self.connection.delete = Mock(side_effect=Exception)
        api_delete = partial(self.api.delete, 1)

        self.assertRaises(InternalError, api_delete)


class TestModelAPI(TestCase):
    def setUp(self):
        self.connection = Mock(Session())
        self.root_uri = "http://localhost/"
        self.model_path = "model/"
        self.uri = self.root_uri + self.model_path
        self.api = ModelAPI(self.root_uri, self.connection)
        self.response = Mock(Response())

        self.feature_config = {
            "activeFeatures": [
                "num-unique-vals"
            ]
        }
        self.description = "test model"
        self.classes = ["name", "age"]
        self.model_type = "randomForest"
        self.labels = {
            "123": "name",
            "666": "age"
        }
        self.cost_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.resampling_strategy = "ResampleToMean"
        self.num_bags = 60
        self.bag_size = 90

        self.data = {
            "features": self.feature_config,
            "description": self.description,
            "classes": ["unknown"],
            "modelType": self.model_type,
            "labelData": self.labels,
            "costMatrix": self.cost_matrix,
            "resamplingStrategy": self.resampling_strategy,
            "numBags": self.num_bags,
            "bagSize": self.bag_size
        }

    def test_post(self):
        message = "Created"
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.post = Mock(return_value=self.response)
        result = self.api.post(
            feature_config=self.feature_config,
            description=self.description,
            classes=None,
            model_type=self.model_type,
            labels=self.labels,
            cost_matrix=self.cost_matrix,
            resampling_strategy=self.resampling_strategy,
            num_bags=self.num_bags,
            bag_size=self.bag_size)

        self.assertEqual(result, message)
        self.connection.post.assert_called_with(self.uri, json=self.data)

    def test_post_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)
        api_post = partial(
            self.api.post,
            feature_config=self.feature_config,
            description=self.description,
            classes=None,
            model_type=self.model_type,
            labels=self.labels,
            cost_matrix=self.cost_matrix,
            resampling_strategy=self.resampling_strategy,
            num_bags=self.num_bags)

        self.assertRaises(InternalError, api_post)

    def test_update(self):
        message = "Updated"
        key = 1
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.post = Mock(return_value=self.response)
        result = self.api.update(
            key,
            feature_config=self.feature_config,
            description=self.description,
            classes=["unknown"],
            model_type=self.model_type,
            labels=self.labels,
            cost_matrix=self.cost_matrix,
            resampling_strategy=self.resampling_strategy,
            num_bags=self.num_bags,
            bag_size=self.bag_size)

        self.assertEqual(result, message)
        self.connection.post.assert_called_with(
            self.uri + str(key),
            json=self.data)

    def test_update_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)
        api_update = partial(
            self.api.update,
            1,
            feature_config=self.feature_config,
            description=self.description,
            classes=["unknown"],
            model_type=self.model_type,
            labels=self.labels,
            cost_matrix=self.cost_matrix,
            resampling_strategy=self.resampling_strategy,
            num_bags=self.num_bags)

        self.assertRaises(InternalError, api_update)

    def test_item(self):
        key = 1
        self.response.status_code = 200
        self.response.json = Mock(return_value=self.data)
        self.connection.get = Mock(return_value=self.response)
        result = self.api.item(key)

        self.assertEqual(result, self.data)
        self.connection.get.assert_called_with(self.uri + str(key))

    def test_item_with_connection_exception(self):
        self.connection.get = Mock(side_effect=Exception)
        api_item = partial(self.api.item, 1)

        self.assertRaises(InternalError, api_item)

    def test_delete(self):
        message = "Deleted"
        key = 1
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.delete = Mock(return_value=self.response)
        result = self.api.delete(key)

        self.assertEqual(result, message)
        self.connection.delete.assert_called_with(self.uri + str(key))

    def test_delete_with_connection_exception(self):
        self.connection.delete = Mock(side_effect=Exception)
        api_delete = partial(self.api.delete, 1)

        self.assertRaises(InternalError, api_delete)

    def test_keys(self):
        keys = [1, 2]
        self.response.status_code = 200
        self.response.json = Mock(return_value=keys)
        self.connection.get = Mock(return_value=self.response)
        result = self.api.keys()

        self.assertEqual(result, keys)
        self.connection.get.assert_called_with(self.uri)

    def test_keys_with_connection_exception(self):
        self.connection.get = Mock(side_effect=Exception)

        self.assertRaises(InternalError, self.api.keys)

    def test_train(self):
        key = 1
        self.response.status_code = 200
        self.connection.post = Mock(return_value=self.response)
        result = self.api.train(key)

        self.assertEqual(result, True)
        self.connection.post.assert_called_with(
            self.uri + str(key) + "/train")

    def test_train_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)

        self.assertRaises(InternalError, partial(self.api.train, 1))

    def test_predict(self):
        modelKey = 1
        dataSetKey = 2
        predictions = {
            "modelID": modelKey,
            "dataSetID": dataSetKey,
            "predictions": {
                "label": "name",
                "confidence": 0.6,
                "scores": {"name": 0.6},
                "features": {}
            }
        }

        self.response.status_code = 200
        self.response.json = Mock(return_value=predictions)
        self.connection.post = Mock(return_value=self.response)
        result = self.api.predict(modelKey, dataSetKey)

        self.assertEqual(result, predictions)
        self.connection.post.assert_called_with(
            self.uri + str(modelKey) + "/predict/" + str(dataSetKey))

    def test_predict_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)

        self.assertRaises(InternalError, partial(self.api.predict, 1, 2))


class TestOntologyAPI(TestCase):
    def setUp(self):
        self.connection = Mock(Session())
        self.root_uri = "http://localhost/"
        self.model_path = "owl/"
        self.uri = self.root_uri + self.model_path
        self.api = OntologyAPI(self.root_uri, self.connection)
        self.response = Mock(Response())

        self.description = "test ontology"
        self.file_path = __file__
        self.owl_format = OwlFormat.TURTLE.value

    def test_post(self):
        message = "Created"
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.post = Mock(return_value=self.response)
        result = self.api.post(
            self.description, self.file_path, self.owl_format)

        args = self.connection.post.call_args
        self.assertEqual(args[0][0], self.uri)
        self.assertEqual(
            args[1]["data"],
            {"description": self.description, "format": self.owl_format})
        self.assertIsNotNone(args[1]["files"]["file"])
        self.assertEqual(result, message)

    def test_post_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)
        api_post = partial(
            self.api.post, self.description, self.file_path, self.owl_format)

        self.assertRaises(InternalError, api_post)

    def test_post_with_unsupported_format(self):
        api_post = partial(
            self.api.post, self.description, self.file_path, "unknown")

        self.assertRaises(ValueError, api_post)

    def test_update(self):
        message = "Updated"
        key = 1
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.post = Mock(return_value=self.response)
        result = self.api.update(
            key, self.description, self.file_path, self.owl_format)

        args = self.connection.post.call_args
        self.assertEqual(args[0][0], self.uri + str(key))
        self.assertEqual(
            args[1]["data"],
            {"description": self.description, "format": self.owl_format})
        self.assertIsNotNone(args[1]["files"]["file"])
        self.assertEqual(result, message)

    def test_update_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)
        api_update = partial(
            self.api.update,
            1,
            self.description,
            self.file_path,
            self.owl_format)

        self.assertRaises(InternalError, api_update)

    def test_update_with_unsupported_format(self):
        api_update = partial(
            self.api.update, 1, self.description, self.file_path, "unknown")

        self.assertRaises(ValueError, api_update)

    def test_delete(self):
        message = "Deleted"
        key = 1
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.delete = Mock(return_value=self.response)
        result = self.api.delete(key)

        self.assertEqual(result, message)
        self.connection.delete.assert_called_with(self.uri + str(key))

    def test_delete_with_connection_exception(self):
        self.connection.delete = Mock(side_effect=Exception)
        api_delete = partial(self.api.delete, 1)

        self.assertRaises(InternalError, api_delete)

    def test_keys(self):
        keys = [1, 2]
        self.response.status_code = 200
        self.response.json = Mock(return_value=keys)
        self.connection.get = Mock(return_value=self.response)
        result = self.api.keys()

        self.assertEqual(result, keys)
        self.connection.get.assert_called_with(self.uri)

    def test_keys_with_connection_exception(self):
        self.connection.get = Mock(side_effect=Exception)

        self.assertRaises(InternalError, self.api.keys)

    def test_item(self):
        key = 1
        item = {
            "description": self.description,
            "format": self.owl_format
        }
        self.response.status_code = 200
        self.response.json = Mock(return_value=item)
        self.connection.get = Mock(return_value=self.response)
        result = self.api.item(key)

        self.assertEqual(result, item)
        self.connection.get.assert_called_with(self.uri + str(key))

    def test_item_with_connection_exception(self):
        self.connection.get = Mock(side_effect=Exception)
        api_item = partial(self.api.item, 1)

        self.assertRaises(InternalError, api_item)

    @patch("requests.get")
    def test_owl_file(self, get):
        key = 3
        self.api.item = Mock(return_value={"name": "test.ttl"})

        self.response.status_code = 200
        chunks = ["1", "2"]
        self.response.iter_content = Mock(return_value=chunks)
        get.configure_mock(return_value=self.response)

        f = MagicMock(spec=TextIOBase)
        f.__enter__ = Mock(return_value=f)
        self.api._create_local_owl_file = Mock(return_value=f)

        result = self.api.owl_file(key)

        get.assert_called_with(self.uri + str(key) + "/file", stream=True)
        self.api._create_local_owl_file.assert_called_with(result)
        f.write.assert_has_calls([call(chunk) for chunk in chunks])

    @patch("requests.get")
    def test_owl_file_with_connection_exception(self, get):
        self.api.item = Mock(return_value={"name": "test.ttl"})
        get.configure_mock(side_effect=Exception)
        api_owl_file = partial(self.api.owl_file, 1)

        self.assertRaises(InternalError, api_owl_file)


def init_ssd(target):
    target.dataset_json = {
        "dateCreated": "2017-03-16T15:29:03.388",
        "dateModified": "2017-03-16T15:29:03.388",
        "description": "",
        "filename": "businessInfo.csv",
        "id": 2035625835,
        "path": "/Users/li151/Dev/serene/./storage/datasets/2035625835/businessinfo.csv",
        "typeMap": {},
        "columns": [
            {
                "datasetID": 2035625835,
                "id": 1246005714,
                "index": 0,
                "logicalType": "string",
                "name": "company",
                "path": "/Users/li151/Dev/serene/./storage/datasets/2035625835/businessinfo.csv",
                "sample": ["Data61"],
                "size": 59
            },
            {
                "datasetID": 2035625835,
                "id": 281689915,
                "index": 1,
                "logicalType": "string",
                "name": "ceo",
                "path": "/Users/li151/Dev/serene/./storage/datasets/2035625835/businessinfo.csv",
                "sample": ["Garv Mcowen"],
                "size": 59
            }
        ]
    }

    target.dataset = DataSet(target.dataset_json)
    target.ontology = Ontology().update({
        "name": __file__,
        "id": 123,
        "description": "test ontology",
        "dateCreated": "2017-03-16T15:29:03.388",
        "dateModified": "2017-03-16T15:29:03.388"
    })

    target.ssd = SSD(target.dataset, target.ontology, "test ssd")
    target.ssd_json = target.ssd.json


class TestSsdAPI(TestCase):
    def setUp(self):
        self.connection = Mock(Session())
        self.root_uri = "http://localhost/"
        self.ssd_path = "ssd/"
        self.uri = self.root_uri + self.ssd_path
        self.api = SsdAPI(self.root_uri, self.connection)
        self.response = Mock(Response())

        init_ssd(self)

    def test_keys(self):
        keys = [1, 2]
        self.response.status_code = 200
        self.response.json = Mock(return_value=keys)
        self.connection.get = Mock(return_value=self.response)
        result = self.api.keys()

        self.assertEqual(result, keys)
        self.connection.get.assert_called_with(self.uri)

    def test_keys_with_connection_exception(self):
        self.connection.get = Mock(side_effect=Exception)

        self.assertRaises(InternalError, self.api.keys)

    def test_post(self):
        message = "Created"
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.post = Mock(return_value=self.response)
        result = self.api.post(self.ssd_json)

        args = self.connection.post.call_args
        self.assertEqual(args[0][0], self.uri)
        self.assertEqual(
            args[1]["data"],
            self.ssd_json)
        self.assertEqual(result, message)

    def test_post_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)
        api_post = partial(
            self.api.post, self.ssd_json)

        self.assertRaises(InternalError, api_post)

    def test_update(self):
        message = "Updated"
        key = 1
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.post = Mock(return_value=self.response)
        result = self.api.update(key, self.ssd_json)

        args = self.connection.post.call_args
        self.assertEqual(args[0][0], self.uri + str(key))
        self.assertEqual(args[1]["data"], self.ssd_json)
        self.assertEqual(result, message)

    def test_update_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)
        api_update = partial(self.api.update, 1, self.ssd_json)

        self.assertRaises(InternalError, api_update)

    def test_item(self):
        key = 1
        self.response.status_code = 200
        self.response.json = Mock(return_value=self.ssd_json)
        self.connection.get = Mock(return_value=self.response)
        result = self.api.item(key)

        self.assertEqual(result, self.ssd_json)
        self.connection.get.assert_called_with(self.uri + str(key))

    def test_item_with_connection_exception(self):
        self.connection.get = Mock(side_effect=Exception)
        api_item = partial(self.api.item, 1)

        self.assertRaises(InternalError, api_item)

    def test_delete(self):
        message = "Deleted"
        key = 1
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.delete = Mock(return_value=self.response)
        result = self.api.delete(key)

        self.assertEqual(result, message)
        self.connection.delete.assert_called_with(self.uri + str(key))

    def test_delete_with_connection_exception(self):
        self.connection.delete = Mock(side_effect=Exception)
        api_delete = partial(self.api.delete, 1)

        self.assertRaises(InternalError, api_delete)


class TestOctopusAPI(TestCase):
    def setUp(self):
        self.connection = Mock(Session())
        self.root_uri = "http://localhost/"
        self.octopus_path = "octopus/"
        self.uri = self.root_uri + self.octopus_path
        self.api = OctopusAPI(self.root_uri, self.connection)
        self.response = Mock(Response())

        init_ssd(self)

        self.name = "test octopus"
        self.feature_config = {
            "activeFeatures": [
                "num-unique-vals"
            ]
        }
        self.description = "test model"
        self.model_type = "randomForest"
        self.resampling_strategy = "ResampleToMean"
        self.num_bags = 60
        self.bag_size = 90
        self.modeling_props = {"topkSteinerTrees": 1}

        self.data = {
            "ssds": [self.ssd],
            "name": self.name,
            "description": self.description,
            "modelType": self.model_type,
            "resamplingStrategy": self.resampling_strategy,
            "features": self.feature_config,
            "numBags": self.num_bags,
            "bagSize": self.bag_size,
            "ontologies": [self.ontology],
            "modelingProps": self.modeling_props
        }

    def test_post(self):
        message = "Created"
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.post = Mock(return_value=self.response)
        result = self.api.post(
            ssds=[self.ssd],
            name=self.name,
            description=self.description,
            feature_config=self.feature_config,
            model_type=self.model_type,
            resampling_strategy=self.resampling_strategy,
            num_bags=self.num_bags,
            bag_size=self.bag_size,
            ontologies=[self.ontology],
            modeling_props=self.modeling_props)

        self.assertEqual(result, message)
        self.connection.post.assert_called_with(self.uri, json=self.data)

    def test_post_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)
        api_post = partial(
            self.api.post,
            ssds=[self.ssd],
            name=self.name,
            description=self.description,
            feature_config=self.feature_config,
            model_type=self.model_type,
            resampling_strategy=self.resampling_strategy,
            num_bags=self.num_bags,
            bag_size=self.bag_size,
            ontologies=[self.ontology],
            modeling_props=self.modeling_props)

        self.assertRaises(InternalError, api_post)

    def test_update(self):
        message = "Updated"
        key = 1
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.post = Mock(return_value=self.response)
        result = self.api.update(
            key,
            ssds=[self.ssd],
            name=self.name,
            description=self.description,
            feature_config=self.feature_config,
            model_type=self.model_type,
            resampling_strategy=self.resampling_strategy,
            num_bags=self.num_bags,
            bag_size=self.bag_size,
            ontologies=[self.ontology],
            modeling_props=self.modeling_props)

        self.assertEqual(result, message)
        self.connection.post.assert_called_with(
            self.uri + str(key),
            json=self.data)

    def test_update_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)
        api_update = partial(
            self.api.update,
            1,
            ssds=[self.ssd],
            name=self.name,
            description=self.description,
            feature_config=self.feature_config,
            model_type=self.model_type,
            resampling_strategy=self.resampling_strategy,
            num_bags=self.num_bags,
            bag_size=self.bag_size,
            ontologies=[self.ontology],
            modeling_props=self.modeling_props)

        self.assertRaises(InternalError, api_update)

    def test_item(self):
        key = 1
        self.response.status_code = 200
        self.response.json = Mock(return_value=self.data)
        self.connection.get = Mock(return_value=self.response)
        result = self.api.item(key)

        self.assertEqual(result, self.data)
        self.connection.get.assert_called_with(self.uri + str(key))

    def test_item_with_connection_exception(self):
        self.connection.get = Mock(side_effect=Exception)
        api_item = partial(self.api.item, 1)

        self.assertRaises(InternalError, api_item)

    def test_delete(self):
        message = "Deleted"
        key = 1
        self.response.status_code = 200
        self.response.json = Mock(return_value=message)
        self.connection.delete = Mock(return_value=self.response)
        result = self.api.delete(key)

        self.assertEqual(result, message)
        self.connection.delete.assert_called_with(self.uri + str(key))

    def test_delete_with_connection_exception(self):
        self.connection.delete = Mock(side_effect=Exception)
        api_delete = partial(self.api.delete, 1)

        self.assertRaises(InternalError, api_delete)

    def test_keys(self):
        keys = [1, 2]
        self.response.status_code = 200
        self.response.json = Mock(return_value=keys)
        self.connection.get = Mock(return_value=self.response)
        result = self.api.keys()

        self.assertEqual(result, keys)
        self.connection.get.assert_called_with(self.uri)

    def test_keys_with_connection_exception(self):
        self.connection.get = Mock(side_effect=Exception)

        self.assertRaises(InternalError, self.api.keys)

    def test_train(self):
        key = 1
        self.response.status_code = 200
        self.connection.post = Mock(return_value=self.response)
        result = self.api.train(key)

        self.assertEqual(result, True)
        self.connection.post.assert_called_with(
            self.uri + str(key) + "/train")

    def test_train_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)

        self.assertRaises(InternalError, partial(self.api.train, 1))

    def test_predict(self):
        octopusKey = 1
        dataSetKey = 2
        predictions = {
            "predictions": [{
                "ssd": {
                    "name": "test ssd",
                },
                "score": {
                    "linkCost": 0.5
                }
            }]
        }

        self.response.status_code = 200
        self.response.json = Mock(return_value=predictions)
        self.connection.post = Mock(return_value=self.response)
        result = self.api.predict(octopusKey, dataSetKey)

        self.assertEqual(result, predictions)
        self.connection.post.assert_called_with(
            self.uri + str(octopusKey) + "/predict/" + str(dataSetKey))

    def test_predict_with_connection_exception(self):
        self.connection.post = Mock(side_effect=Exception)

        self.assertRaises(InternalError, partial(self.api.predict, 1, 2))
