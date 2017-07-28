import sys
import os
from math import *
from .utils import *

#############################################################
##                                                         ##
##  Given an graph in graphml format,                      ##
##  this script will convert the "attribute" part of the   ##
##  graph (that is, all but the alignment) into a matching ##
##  problem.                                               ##
##                                                         ##
#############################################################

from math import *

simplify_graph = False #Remove edges that are heavier than some path between its endpoints.

##Check arguments
if len(sys.argv) < 2 or len(sys.argv) > 3:
    raise RuntimeError("Wrong number of arguments\n\t"
    "Usage: "+os.path.basename(__file__)+" alignment.graphml [-s]\n"
    "\t flag -s indicates to implement simplification of the graph")
graphml_file = sys.argv[1]
if not os.path.isfile(graphml_file):
    raise RuntimeError("Wrong argument:\n\t"+graphml_file+" is not a file")
    
if (len(sys.argv) == 3) and sys.argv[2] == '-s':
    simplify_graph = True

content = ""
with open(sys.argv[1]) as f:
    content = f.read()

g = Graph.from_graphml(content, simplify_graph)

attributes = g.node_types['Attribute']
print("nbA =", len(attributes), ";")

max_val = 0
min_att = min(attributes)
doms = []
for i,n in enumerate(attributes):
    # in python3 we need to explicitly get list
    d = list(map(lambda x: x+1, [g.otherNode(e,n) for e in g.inc[n]]))
    max_val = max(max_val, max(d))
    doms.append(d)
print("attribute_domains =", list2D2dznsetlist(list(doms)))


mc = [[0 for j in range(0,max_val+1)] for i in attributes]
for i in range(0,len(g.edges)):
    e = g.edges[i]
    if e[0] in attributes:
        mc[e[0] - min_att][1 + e[1]] = e[2] 
    elif e[1] in attributes:
        mc[e[1] - min_att][1 + e[0]] = e[2] 

mcs = "match_costs = [|"
for i,r in enumerate(mc):
    c = str(r[1:]).replace("[", "").replace("]","|") + ("];" if i == len(mc) - 1 else "\n")
    mcs += c
print(mcs)

mcs = "match_costs_sorted = [|"
for i, r in enumerate(mc):
    s = r[1:]
    x = sorted(range(len(s)), key=lambda k: s[k])
    x = list(map(lambda x: x+1, x))
    c = str(x).replace("[","").replace("]","|") + ("];" if i == len(mc) - 1 else "\n")
    mcs += c
print(mcs)

# att_names =
print("attribute_names = ", list2dznlist([int(g.node_names[n]) for n in attributes]))

