import pandas as pd

class SourceColumn(object):
    """
    Attributes:
        id:
        name:
        sample:
        semantic_types:
    """
    def __init__(self, col_dict, samples=None):
        """
        Initialize class SourceColumn.
        :param col_dict: dictionary which contains info about the source column.
        :param samples: not implemented
        """
        self.id = int(col_dict["id"])  # id of the column
        self.name = col_dict["name"]  # name of the column
        self.sample = []  # sample of the column values -- not implemented
        self.semantic_types = []

    def score(self, model):
        """
        Calculate scores per each semantic type.
        :param model: model.rf -- file with random forest classifier
        :return: None, updates self.semantic_types
        """
        pass


class ExtendedColumn(object):
    """
    Attributes:
        id:
        name:
        source_columns:
        label:
        sql:
        func:
        sample:
        semantic_types:
    """
    def __init__(self, ecol_dict, samples=None):
        """
        Initialize extended column.
        :param ecol_dict: dictionary with info.
        """
        self.id = int(ecol_dict["id"])
        self.source_columns = [int(e) for e in ecol_dict["columnIds"]]
        self.name = ecol_dict["name"]
        self.label = ecol_dict["label"]
        self.sql = ecol_dict["sql"]
        self.func = ""  # we need to convert SQL to python function???
        self.sample = []  # sample of the column values -- not implemented
        self.semantic_types = []  # tuples (SemanticType, score)

    def score(self, model):
        """
        Calculate scores per each semantic type.
        :param model: model.rf -- file with random forest classifier
        :return: None, updates self.semantic_types
        """
        pass


class SemanticType(object):
    """
    This class represents the semantic type which is defined via ontology.
    Each semantic type corresponds either to the class node or the data node in the ontology.
    Attributes:
        type: ClassNode or DataNode
        class_node: uri of the class node in the ontology
        data_node: uri of the data node in the ontology, it's None in case type is ClassNode
    """
    def __init__(self, class_uri, data_uri):
        self.class_node = class_uri  # uri of the class node from the ontology
        self.data_node = data_uri  # uri of the data node from the ontology
        # type corresponds to ClassNode or DataNode
        if data_uri is None:
            self.type = 'ClassNode'
        else:
            self.type = 'DataNode'
