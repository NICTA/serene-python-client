# Benchmark for STP

Folder "resources" contains instances for the problem.

Different approaches were used to construct the alignment graph as well as to identify matches, that's why there are numbered folders which correspond to each particular initialization.
Each numbered folder contains only one alignment graph and one set of matches, but there are several instances of the integration graph (each instance corresponds to a different dataset).

Contents of the folders "#id" (e.g., folder with name "2"):

1. File "modeling.properties" contains meta info of how the alignment graph and matches were constructed
2. Files "graph.json" and "graph.dot" are internal representations of the alignment graph in Karma project.
3. File "alignment.graphml" is the converted version of the alignment graph which is used for the construction of the integration graph.
4. Folder "matches" contains ".csv" files with predicted matches per each dataset.
5. File "{#id}.integration.graphml" contains the graph for which STP needs to be solved.
    Nodes of type "Attribute" need to be leaves.
    Degree of the nodes of type "DataNode" has to be 2 in the solution.
6. File "{#id}.ssd.graphml" contains the desired solution.

[GraphML](http://graphml.graphdrawing.org/) format is used throughout this benchmark to serialize graphs.
For STP the most important attributes are "id", "type" for nodes and "weight", "source", "target" for links.