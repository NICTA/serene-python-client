%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% MiniZinc model to solve the Steiner Tree Problem arising
%% in the Rel2Onto problem. 
%%
%% Requires: 
%%           steiner_tree.mzn decomposition or a solver
%%              with a propagator for it (e.g. Chuffed)
%%           MiniZinc compiler (tested on 2.1.4)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



include "steiner_tree.mzn";
include "tree.mzn";


int: nbV;                               % Nb of nodes in alignment graph
int: nbE;                               % Nb of edges in alignment graph
set of int: nodes = 1..nbV;    
set of int: edges = 1..nbE;
set of int: cnodes;                     % Class nodes in alignment graph
set of int: dnodes;                     % Data nodes in alignment graph
set of int: anodes;                     % Attribute nodes in alignment graph (normally none)
array [nodes] of var bool: vs;          % Is a node of the alignment graph in the solution?
array [edges] of var bool: es;          % Is an edge of the alignment graph in the solution?
array [nodes,edges] of bool: adjacent;  % adjacent[n,e] <=> node n is an endnode of edge e
array [edges] of nodes: heads;          % endpoint 1 of each edge (The graph is undirected!!)
array [edges] of nodes: tails;          % endpoint 2 of each edge
array[1..2,edges] of int: endpairs = array2d(1..2,edges,tails++heads);
array [edges] of int: ws;               % Weights of edges in alignment graph
var 0..sum(ws): w;                             % Weight of the solution corresponding to the alignment part

int: nb_unknown_nodes;
set of int: unknowns = 3..2+nb_unknown_nodes;
array [1..2+nb_unknown_nodes] of int: all_unk_nodes;
int: node_all = all_unk_nodes[1];
int: node_unk = all_unk_nodes[2];


constraint steiner_tree(vs,es,adjacent,endpairs,w,ws);
%constraint tree(vs,es,adjacent,endpairs);
constraint w = sum(i in edges)(bool2int(es[i])*ws[i]);



int: nbA;                                           % Nb of attribute nodes
set of int: atts = 1..nbA;                 
array [atts] of set of int: attribute_domains;      % Data nodes to which each attribute can attach  
constraint forall(att in anodes) (vs[att] == true); % Attribute nodes (normally none) are in the graph.

%constraint sum(i in edges)(bool2int(es[i])) = sum(i in nodes)(bool2int(vs[i])) - 1;
%constraint forall(i in index_set(anodes)) ((sum(a in edges where adjacent[anodes[i],a]) (bool2int(es[a]))) == 1);
%constraint forall(i in index_set(dnodes)) (vs[dnodes[i]] <-> (sum (a in edges where adjacent[dnodes[i],a]) (bool2int(es[a]))) == 2);
%constraint forall(i in index_set(dnodes)) (not(vs[dnodes[i]]) <-> (sum (a in edges where adjacent[dnodes[i],a]) (bool2int(es[a]))) == 0);

set of int: extended_nodes = 1..nbV+nbA;
array[extended_nodes] of var bool: extended_vs;
constraint forall(n in nodes) (extended_vs[n] = vs[n]);

%Matching problem:
array [atts] of var int: match;                                       % Match between Attributes and Data nodes
array [atts] of int: attribute_names;                                 % Just for printing
%int: dummy = 0;
constraint forall (a in atts) (match[a] in (attribute_domains[a] union {attribute_names[a]}));     % Must match someone in the domain (i.e. someone there is an edge to)
constraint forall (a in atts) (extended_vs[match[a]]);                         % The data node matched MUST be in the subgraph of the alignment
constraint forall (a in atts) (match[a] != attribute_names[a] <-> extended_vs[attribute_names[a]] = false);

%include "alldifferent_except_0.mzn"; %ADD DUMMY TO THE DOMAINS!!!
%constraint alldifferent_except_0(match);
include "alldifferent.mzn";
constraint alldifferent(match);                                       % Exact match

%%% If there is no match to a data node, you should not have that datanode in the tree
constraint forall (d in dnodes) (exists(a in atts)(match[a] == d) <-> vs[d]);
%Only 2 edges per data node (for unknown nodes, there is only one edge to ClassNode "Unknown" anyway)
constraint forall (d in dnodes) (vs[d] <-> sum(e in edges where adjacent[d,e])(es[e]) == 1); 
constraint 
	if nb_unknown_nodes > 0 
	then
	%If ClassNode "Unknown" is used, ClassNode "All" is used as well
	vs[node_unk] <-> vs[node_all]
	%ClassNode "Unknown" must be connected to classNode "All"
	/\ vs[node_unk] <-> sum(e in edges where adjacent[node_unk,e] /\ adjacent[node_all,e])(es[e]) == 1
	%ClassNode "All" has degree 2
	/\ vs[node_all] <-> sum(e in edges where adjacent[node_all,e])(es[e]) == 2
	else
	true
	endif;




%%Patterns
int: nb_patterns;
set of int: PATTS = 1..nb_patterns;
array[PATTS] of int: patterns_w;
array[PATTS] of set of int: patterns;
array[PATTS] of var bool: c_p;
constraint forall (p in PATTS) ((sum (e in patterns[p]) (es[e]) == card(patterns[p])) <-> c_p[p]);
var 0..sum(patterns_w): wp;
constraint if nb_patterns = 0 then wp = 0 else wp = sum(p in PATTS) (c_p[p] * patterns_w[p]) endif;





array[atts,int] of int: match_costs;
array[atts,int] of int: extended_match_costs = array2d(atts,min(index_set_2of2(match_costs))..max([attribute_names[a] | a in atts]),
	[ if b <= max(index_set_2of2(match_costs)) then match_costs[a,b] else 0 endif | 
	    a in atts, b in min(index_set_2of2(match_costs))..max([attribute_names[a] | a in atts])]);
var 0..sum(a in atts)(sum(n in index_set_2of2(match_costs))(match_costs[a,n])): wm;
constraint wm = sum(a in atts) (extended_match_costs[a,match[a]]);
var int: obj = w + wm - wp;


%For the startegy
include "arg_sort.mzn";
array[atts,int] of int: match_costs_sorted; % = [arg_sort([match_costs[i,j]| j in index_set_2of2(match_costs)])| i in atts];
ann: cheap_matches = int_search([ match[a] == match_costs_sorted[a,i] |a in atts, i in index_set_2of2(match_costs_sorted)], input_order,indomain_max,complete);

array[edges] of int: sorted_edges = arg_sort(ws);
ann: naive_e = int_search(es,input_order,indomain_min,complete);
ann: kruskal  = int_search([es[sorted_edges[e]] | e in edges],input_order,indomain_max,complete);
ann: kruskal2 = int_search([es[sorted_edges[nbE + 1 - e]] | e in edges],input_order,indomain_min,complete);

ann: assume(array[int] of var bool, bool);

ann: cheap_edges_fill_pattern;
ann: custom(ann, array[int] of int, array[int] of var bool);
array[int] of ann: cefp = 
	if nb_patterns == 0 then [] 
	else [custom(cheap_edges_fill_pattern,[nbE]++[nb_patterns]++ws++[ card(patterns[p]) | p in PATTS]++patterns_w++array1d([ e | p in PATTS, e in patterns[p]]),es)] endif;
 
solve 
%::int_search(match, input_order, indomain_min, complete)
::seq_search([assume([match[a] != attribute_names[a] | a in atts],true)]++[cheap_matches]++cefp++[kruskal2])
minimize obj;

float: time = 0.0;
output ["digraph {\n"]++
[ "  \(n-1)"++ (if n in cnodes then "[fillcolor=\"#59d0a0\",style=filled];" else if n in dnodes then "[fillcolor=gold,style=filled];" else "" endif endif)++"\n" | n in nodes  where fix(vs[n])] ++ 
[ "  "++if fix(heads[e]) in cnodes then "\(heads[e]-1) -> \(tails[e]-1)" else "\(tails[e]-1) -> \(heads[e]-1)" endif ++" [label=\"w=\(ws[e]/10000)\"];\n" | e in edges  where fix(es[e])] ++ 
[ "  \(attribute_names[a]) [shape=hexagon];\n" | a in atts] ++
[ "  "++if fix(match[a] != attribute_names[a]) then "\(match[a]-1) -> \(attribute_names[a]) [label=\"w=\(match_costs[a,fix(match[a])]/10000)\"];\n" else "\(match[a]) -> \(match[a]) [label=\"Imposible match\"];\n" endif | a in atts]++
["  graph [label=\"\nobj = \(obj)\nwm = \(wm)\nwp = \(wp)\nw = \(w)\",time=\"\(time)\",obj=\"\(obj)\",cost=\"\(w)\",match_score=\"\(wm)\"]}"];

