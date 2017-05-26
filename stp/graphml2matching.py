import sys
import numpy
from math import *


def median(lst):
    return numpy.median(numpy.array(lst))


class Graph:
    def __init__(self,nb_nodes = 0,nb_edges = 0):
        self.nb_nodes = nb_nodes
        self.nb_edges = nb_edges
        self.edges = []
        self.inc = [[] for i in self.nodes()]
        self.out = [[] for i in self.nodes()]

    def nodes(self):
        return range(0,self.nb_nodes)

    def isNode(self, n):
        return n >=0  and n < self.nb_nodes

    def addEdge(self, s, d, w = 0):
        if self.isNode(s) and self.isNode(d):
            self.edges.append((s,d,w))
            self.inc[d].append(s)
            self.out[s].append(d)
            self.nb_edges += 1

    def addNode(self):
        self.nb_nodes += 1
        self.inc.append([])
        self.out.append([])

    def remNode(self,n):
        self.edges = [e for e in self.edges if (e[0] != n and e[1] != n)]
        self.edges = map(lambda t: map(lambda x: x if x < n else x-1, t),self.edges)
        self.edges = map(tuple,self.edges)
        self.nb_edges = len(self.edges)
        del self.inc[n]
        del self.out[n]
        self.nb_nodes -= 1
        for i in range(0,len(self.inc)):
            l = self.inc[i]
            l = map(lambda x: x if x < n else x-1, l)
            l = [x for x in l if x != n]
            self.inc[i] = l
        for i in range(0,len(self.out)):
            l = self.out[i]
            l = map(lambda x: x if x < n else x-1, l)
            l = [x for x in l if x != n]
            self.out[i] = l

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
        return res

    def to_dot(self):
        res = "graph {\n"
        for i in range(0,self.nb_edges):
            e = self.edges[i]
            res += "\t"+str(e[0] + 1)+" -- "+str(e[1] + 1)+"[label="+str(e[2])+"]\n"
        res += "}\n"
        return res


content = []
with open(sys.argv[1]) as f:
    content = f.readlines()

type_label = None
weight_label = None
for l in content:
    if type_label != None and weight_label != None:
        break
    if "<key attr.name=\"type\" attr.type=\"string\" for=\"node\"" in l:
        type_label = l.split("id=\"")[1].split("\"")[0]
    if "<key attr.name=\"weight\" attr.type=\"double\" for=\"edge\"" in l:
        weight_label = l.split("id=\"")[1].split("\"")[0]

nodes = 0
dic = {}

attributes = []
attribute_name = []

for i in range(0,len(content)):
    l = content[i]
    if "<node id=" in l:
        id = int(l.split("<node id=\"")[1].split("\"")[0])
        dic[id] = nodes
        nodes += 1
        a = i+1
        while not "</node>" in content[a]:
            if type_label in content[a]:
                if "Attribute" in content[a]:
                    attributes.append(dic[id])
                    attribute_name.append(id)
                break
            a += 1


g = Graph(nodes)


for i in range(0,len(content)):
    l = content[i]
    if "<edge source=" in l:
        s = int(l.split("<edge source=\"")[1].split("\"")[0])
        d = int(l.split("<edge source=\"")[1].split("\"")[2])
        if (not d in dic or not s in dic):
            continue
        s = dic[s]
        d = dic[d]
        if (g.isNode(d) and g.isNode(s)) and (s in attributes or d in attributes):
            a = i+1
            d7 = -1
            while not "</edge>" in content[a]:
                if weight_label in content[a]:
                    f = float(content[a].split(">")[1].split("<")[0])
                    fac = 10000 if f < 100 else 100
                    d7 = int(fac*f)        
                    break
                a += 1
            if d7 == -1:
                print("Error parsing ",l)
                exit(1)
            g.addEdge(s,d,d7)

print("nbA =",len(attributes),";")


print("attribute_domains = [")
max_val = 0
max_att = max(attributes)
min_att = min(attributes)
for i,n in enumerate(attributes):
    dom = map(lambda x: x+1,g.inc[n])
    max_val = max(max_val,max(dom))
    print(str(dom).replace("[","{").replace("]","}")+("," if i < len(attributes) -1  else "];"))


mc = [[0 for j in range(0,max_val+1)] for i in attributes]
for i in range(0,len(g.edges)):
    e = g.edges[i]
    if e[0] in attributes:
        mc[e[0] - min_att][1 + e[1]] = e[2] #- med
    elif e[1] in attributes:
        #print e[1],min_att,e[0]
        mc[e[1] - min_att][1 + e[0]] = e[2] #- med
mcs = "match_costs = [|"

for i,r in enumerate(mc):
    c = str(r[1:]).replace("[","").replace("]","|") + ("];" if i == len(mc) - 1 else "\n")
    mcs += c
print(mcs)

mcs = "match_costs_sorted = [|"
for i,r in enumerate(mc):
    s = r[1:]
    x = sorted(range(len(s)), key=lambda k: s[k])
    x = map(lambda x: x+1, x)
    c = str(x).replace("[","").replace("]","|") + ("];" if i == len(mc) - 1 else "\n")
    mcs += c
print(mcs)

att_names = "attribute_names = ["
for i in range(0,len(attribute_name)):
    att_names += str(attribute_name[i])+ ("," if i < len(attribute_name) - 1 else "];")
print(att_names)

