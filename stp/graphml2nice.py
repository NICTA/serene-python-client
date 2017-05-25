import sys

from math import *
import Queue as Q

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
		self.nb_edges -= 1
	def addNode(self):
		self.nb_nodes += 1
		self.inc.append([])
		self.out.append([])
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
			#print "At",s
			for e in self.out[s]+self.inc[s]:
				x,y,w = self.edges[e]
				o = x if y == s else y
				if sp[o] == -1 or sp[s] + w < sp[o]:
					#print "   Update",e,o,sp[s],w
					#if o == 1:
					#	print "                        -------------"
					sp[o] = sp[s] + w					
					q.put(o)
		#print sp
		#exit(0)
		return sp	
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

classnodes = []
datanodes  = []
attributes = []

for i in range(0,len(content)):
	l = content[i]
	if "<node id=" in l:
		id = int(l.split("<node id=\"")[1].split("\"")[0])
		dic[id] = nodes
		nodes += 1
		a = i+1
		while not "</node>" in content[a]:
			if type_label in content[a]:
				if "Class" in content[a]:
					classnodes.append(id)
				elif "Data" in content[a]:
					datanodes.append(id)
				elif "Attribute" in content[a]:
					attributes.append(id) 
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
		if (g.isNode(d) and g.isNode(s)):
			a = i+1
			d7 = 0
			while not "</edge>" in content[a]:
				if weight_label in content[a]:
					f = float(content[a].split(">")[1].split("<")[0])
					fac = 10000 if f < 100 else 100
					d7 = int(fac*f)	
					break
				a += 1
			g.addEdge(s,d,d7)
#"""
apsp = []
for n in g.nodes():
	apsp.append(g.dijkstra(n))
i = 0
while i < g.nb_edges:
	s,d,w = g.edges[i]
	#print i,s,d,w,apsp[s][d]
	if apsp >= 0 and apsp[s][d] < w:
		g.remEdge(i)
	else:
		i += 1
#"""
res = "ws = ["
for i in range(0,len(g.edges)):
	e = g.edges[i]
	res+= str(e[2])+ (',' if i < len(g.edges) - 1 else '')
res+= "];"
print g.to_dzn()
print "cnodes = "+str(map(lambda x: x+1,classnodes)).replace("[","{").replace("]","}")+";"
print "dnodes = "+str(map(lambda x: x+1,datanodes)).replace("[","{").replace("]","}")+";"
print "anodes = "+str(map(lambda x: x+1,attributes)).replace("[","{").replace("]","}")+";"
#print g.to_dzn()
print res
