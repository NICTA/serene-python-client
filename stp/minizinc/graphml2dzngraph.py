import sys
import os
from math import *
from .utils import *

#############################################################
##                                                         ##
##  Given an graph in graphml format,                      ##
##  this script will convert it into MiniZinz format (the  ##
##  result will be outputed in stdout).                    ##
##                                                         ##
#############################################################


simplify_graph = False #Remove edges that are heavier than some path between its endpoints.
parse_patterns = False
##Check arguments
if len(sys.argv) < 2 or len(sys.argv) > 4:
    raise RuntimeError("Wrong number of arguments\n\t"
    "Usage: "+os.path.basename(__file__)+" alignment.graphml [patterns.csv] [-s]\n"
    "\t flag -s indicates to implement simplification of the graph")
graphml_file = sys.argv[1]
if not os.path.isfile(graphml_file):
    raise RuntimeError("Wrong argument:\n\t"+graphml_file+" is not a file")
    
if (len(sys.argv) == 3) and sys.argv[2] == '-s':
    simplify_graph = True
if (len(sys.argv) == 4) and sys.argv[3] == '-s':
    simplify_graph = True
if (len(sys.argv) == 3 and sys.argv[2] != '-s') or len(sys.argv) == 4:
    parse_patterns = True
content = ""
with open(sys.argv[1]) as f:
    content = f.read()

g = Graph.from_graphml(content, simplify_graph)

patterns = []
patterns_w = []
if parse_patterns:
    content = ""
    with open(sys.argv[2]) as f:
        content = f.readlines()
    for i in range(1,len(content)):
        l = content[i].strip().replace(' ','').replace('"','').replace('[','').replace(']','')
        l = l.split(',')
        # patter,support,num_edges,edge_keys
        w = int(weight_conversion(float(l[1])))
        edgs = map(lambda x: g.edge_ids[x] if x in g.edge_ids else None,l[3:len(l)])
        edgs = [e for e in edgs if e is not None]
        if len(edgs) > 0:
            patterns_w.append(w)
            patterns.append(edgs)


incr = lambda x: x+1
print(g.to_dzn())
# in python3 we need to get lists explicitly instead of iterators/generators
print("cnodes = " + list2dznset(list(map(incr, [n for n in g.node_types['ClassNode']]))))
print("dnodes = " + list2dznset(list(map(incr, [n for n in g.node_types['DataNode']]))))
print("anodes = " + list2dznset(list(map(incr, [n for n in g.node_types['Attribute']]))))


print("nb_unknown_nodes = "+str(len(g.unk_info.data_nodes))+";")
print("all_unk_nodes = "+list2dznlist(list(map(incr,g.unk_info.as_list()))))

print("nb_patterns = " + str(len(patterns)) + ";")
print("patterns_w = " + str(patterns_w) + ";")
print("patterns = " + list2D2dznsetlist(patterns))
