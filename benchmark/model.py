"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Abstract model for semantic labelling/typing.
Evaluation will be based on this abstract model.
"""
from neural_nets import CNN, Column_Labeler

class AbstractModel(object):
    def __init__(self):
        pass

    def reset(self):
        pass

    def define_training_data(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass


class DINTModel(AbstractModel):
    """

    """
    def __init__(self):
        pass

    def reset(self):
        pass

    def define_training_data(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass


class KarmaDSLModel(AbstractModel):
    def __init__(self):
        pass

    def reset(self):
        pass

    def define_training_data(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

class NNetModel(AbstractModel):
    def __init__(self, *args):
        assert "model_type" in args

        self.column_labeler = Column_Labeler('cnn@charseq')

        if isinstance(args["model_type"], list):
            # ensemble
            self.model = ensemble()
        else:
            if args['model_type'] == "CNN":
                self.model = self.column_labeler.classifier
                self.description = "CNN with "


    def reset(self):
        pass

    def define_training_data(self, train_files, train_labels):
        self.column_labeler = Column_Labeler([self.model], train_files, train_labels)

    def train(self):
        pass

    def predict(self, test_file):
        pass

    def evaluate(self):
        pass