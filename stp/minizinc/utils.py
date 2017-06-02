from math import *
import queue as Q
import xml.etree.ElementTree as ET
from functools import reduce

nsfy = lambda x : '{http://graphml.graphdrawing.org/xmlns}'+x #to be able to read namespace tags

list2dznset  = lambda l: str(l).replace("[","{").replace("]","}")+";"
list2dznlist = lambda l: str(l)+";"
_list2D2dznsetlist = lambda l: reduce(lambda x, y: x+","+(str(y).replace("[","{").replace("]","}")),[""]+l)[1:]
list2D2dznsetlist = lambda l: "["+_list2D2dznsetlist(l)+"];"


#This needs to be done better
def weight_conversion(v):
    """If Chuffed is used to solve this STP, it does NOT support floats.
    Therefore a conversion is made to get a higher precision by multiplying the 
    floats by 10000.
    Nonetheless, if some edge is way too heavy, this multiplication may overflow ints.
    Therefore we do a pretty mediocre conversion...."""
    fac = 10000 if v < 100 else 100
    return v*fac


def find_tags(graphml_root):
    """Get the name of the key that will indicate the type of a node (attr, data, class nodes...)
    Same for the weight of an edge."""
    type_label = graphml_root.findall(".//"+nsfy("key")+"[@attr.name='type'][@for='node']")
    if len(type_label) != 1:
        raise RuntimeError("Wrong format in graphml:\n\tToo many or no key indicating"
        " type of nodes")
    type_label = type_label[0]

    weight_label = graphml_root.findall(".//"+nsfy("key")+"[@attr.name='weight'][@for='edge']")
    if len(weight_label) != 1:
        raise RuntimeError("Wrong format in graphml:\n\t: Too many or no key indicating"
        " weight of edges")
    weight_label = weight_label[0]
    return (type_label.attrib['id'],weight_label.attrib['id'])


class Graph:
    def __init__(self,nb_nodes = 0,nb_edges = 0):
        self.nb_nodes = nb_nodes
        self.nb_edges = nb_edges
        self.edges = []
        self.inc = [[] for i in self.nodes()]
        self.out = [[] for i in self.nodes()]
        self.node_types = {}
        self.node_names = []

    def nodes(self):
        return range(0,self.nb_nodes)

    def isNode(self, n):
        return n >=0  and n < self.nb_nodes

    def addEdge(self, s, d, w = 0):
        if self.isNode(s) and self.isNode(d):
            self.edges.append((s,d,w))
            self.inc[d].append(len(self.edges)-1)
            self.out[s].append(len(self.edges)-1)
            self.nb_edges += 1

    def remEdge(self,e):
        s = self.edges[e][0]
        d = self.edges[e][1]
        del self.edges[e]
        self.inc[s] = [x for x in self.inc[s] if x != e]
        self.out[s] = [x for x in self.out[s] if x != e]
        self.inc[d] = [x for x in self.inc[d] if x != e]
        self.out[d] = [x for x in self.out[d] if x != e]
        for n in self.nodes():
            self.inc[n] = map(lambda x: x if x < e else x - 1 ,self.inc[n])
            self.out[n] = map(lambda x: x if x < e else x - 1 ,self.out[n])
        self.nb_edges -= 1

    def otherNode(self, edge, node):
        if node != self.edges[edge][0] and node != self.edges[edge][1]:
            raise RuntimeError("requestion other node"+str(node) +
                               " of the wrong edge "+str(edge) +
                               " "+str(self.edges[edge]))
        return self.edges[edge][1] if self.edges[edge][0] == node else self.edges[edge][0]

    def addNode(self,name = "None"):
        self.nb_nodes += 1
        self.inc.append([])
        self.out.append([])
        self.node_names.append(name)

    def inDeg(self,n):
        return len(self.inc[n])

    def outDeg(self,n):
        return len(self.out[n])

    def deg(self, n):
        return self.inDeg(n) + self.outDeg(n)

    def averageDeg(self):
        s = 0.0
        for i in self.nodes():
            s += self.deg(i)
        return s/float(self.nb_nodes)

    def maxDeg(self):
        max_deg = 0
        for i in self.nodes():
            if self.deg(i) > max_deg:
                max_deg = self.deg(i)
        return max_deg

    def minDeg(self):
        min_deg = self.nb_nodes
        for i in self.nodes():
            if self.deg(i) < min_deg:
                min_deg = self.deg(i)
        return min_deg

    def dijkstra(self, source):
        sp = [-1 for i in self.nodes()]
        q = Q.PriorityQueue()
        q.put(source)
        sp[source] = 0
        while not q.empty():
            s = q.get()
            for e in self.out[s]+self.inc[s]:
                x,y,w = self.edges[e]
                o = x if y == s else y
                if sp[o] == -1 or sp[s] + w < sp[o]:
                    sp[o] = sp[s] + w                    
                    q.put(o)
        return sp

    def simplify(self):
        apsp = []
        for n in self.nodes():
            apsp.append(self.dijkstra(n))
        i = 0
        while i < self.nb_edges:
            s,d,w = self.edges[i]
            if apsp >= 0 and apsp[s][d] < w:
                self.remEdge(i)
            else:
                i += 1

    def __str__(self):
        res = ""
        res += "nb_nodes = " + str(self.nb_nodes) +"\n"
        res += "nb_edges = " + str(self.nb_edges) +"\n"
        res += "edges = " + str(self.edges) +"\n"
        res += "inc = " + str(self.inc) +"\n"
        res += "out = " + str(self.out) +"\n"
        return res

    def to_dzn(self):
        res = ""
        res += "nbV = "+str(self.nb_nodes) +";\n"
        res += "nbE = "+str(self.nb_edges) +";\n"
        res += "tails = ["
        for i in range(0,self.nb_edges):
            e = self.edges[i]
            res += str(e[0] + 1)
            if i < self.nb_edges -1:
                res += ","
        res +="];\n"
        res += "heads = ["
        for i in range(0,self.nb_edges):
            e = self.edges[i]
            res += str(e[1] + 1)
            if i < self.nb_edges -1:
                res += ","
        res +="];\n"
        res += "adjacent = [|"
        for n in self.nodes():
            for i in range(0,self.nb_edges):
                e = self.edges[i]
                if n == e[0] or n == e[1]:
                    res+= "true"
                else:
                    res += "false"
                if i < self.nb_edges -1:
                    res += ","
                else:
                    res += "|"
        res +="];\n"
        res += "ws = ["
        for i in range(0,len(self.edges)):
            e = self.edges[i]
            res+= str(e[2])+ (',' if i < len(self.edges) - 1 else '')
        res+= "];"
        return res

    def to_dot(self):
        res = "graph {\n"
        for i in range(0,self.nb_edges):
            e = self.edges[i]
            res += "\t"+str(e[0] + 1)+" -- "+str(e[1] + 1)+"[label="+str(e[2])+"]\n"
        res += "}\n"
        return res

    @staticmethod
    def from_graphml(content, simplify_graph = False):
        graphml_root = ET.fromstring(content)
        (type_label,weight_label) = map(str,find_tags(graphml_root))
        g = Graph()

        dic = {}  #  Conversion from node ID in the file to internal node IDs
        valid_types = ['ClassNode','DataNode','Attribute']
        for t in valid_types:
            if not t in g.node_types:
                g.node_types[t] = []
        def assign(g,n,t):
            if not t in valid_types:
                raise RuntimeError("Unknown node type")
            g.node_types[t].append(n)
        for n in graphml_root.findall(".//"+nsfy("node")):
            node_id = int(n.attrib['id'])        
            g.addNode(str(node_id))
            dic[node_id] = g.nb_nodes - 1
            node_type = n.findall(".//"+nsfy("data")+"[@key='"+type_label+"']")
            if len(node_type) != 1:
                raise RuntimeError("Wrong format in node tag:\n\tNo type indication")
            node_type = node_type[0].text
            assign(g,dic[node_id],node_type)
    

        for e in graphml_root.findall(".//"+nsfy("edge")):
            s = int(e.attrib["source"])
            d = int(e.attrib["target"])
            if not d in dic or not s in dic:
                sys.stderr.write("Ignoring edge ("+str(s)+","+str(d)+") because those nodes don't exist")
                continue
            s = dic[s]
            d = dic[d]
            if not g.isNode(d) or not g.isNode(s):
                raise RuntimeError("Internal state error: not finding valid nodes.")
            edge_weight = e.findall(".//"+nsfy("data")+"[@key='"+weight_label+"']")
            if len(edge_weight) != 1:
                raise RuntimeError("Wrong format in edge tag:\n\tNo weight indication")
            edge_weight = float(edge_weight[0].text)
            edge_weight = int(weight_conversion(edge_weight))
            g.addEdge(s,d,edge_weight)

        if simplify_graph:
            g.simplify()

        return g

