# ----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
import os
import tensorflow.keras as _keras
from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph


class Keras2GraphNode(GraphNode):

    def __init__(self, layer):
        super(Keras2GraphNode, self).__init__(layer)

    @property
    def name(self):
        return self.layer.name

    @property
    def type(self):
        return self.layer.__class__.__name__

    @property
    def keras_layer(self):
        return self.layer


class Keras2Graph(Graph):

    def __init__(self, model):
        # sanity check.
        print(model.summary())
        # if not (type(model) == _keras.models.Sequential or type(model) == _keras.models.Model):
        #     raise TypeError("Keras layer of type %s is not supported." % type(model))
        super(Keras2Graph, self).__init__(model)
        self.model = model

    def build(self):
        self.input_layers = list()
        for i, layer in enumerate(self.model.layers):
            self.layer_map[layer.name] = Keras2GraphNode(layer)
            self.layer_name_map[layer.name] = layer.name
            print(layer)
            for node in layer._inbound_nodes:
                # for pred in node.inbound_layers:
                if node.name not in self.layer_map:
                    self.layer_map[node.name] = Keras2GraphNode(node)
                    self.layer_name_map[node.name] = node.name
                self._make_connection(node.name, layer.name)

        # Kit: TODO
        # Duplicate models for weight sharing
        # Expand the sub-models
        super(Keras2Graph, self).build()
