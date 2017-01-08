"""
Copyright (C) 2016 Data61 CSIRO
Licensed under http://www.apache.org/licenses/LICENSE-2.0 <see LICENSE file>

Abstract model for semantic labelling/typing.
Evaluation will be based on this abstract model.
"""


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