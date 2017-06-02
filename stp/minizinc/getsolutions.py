import sys

import os
cwd = os.getcwd()+"/"

content = []
with open(sys.argv[1],"r") as f:
    content = f.readlines()

i = 0
while i < len(content):
    l = content[i]
    if "RUNNING BENCHMARK" in l:
        fname = l.split("BENCHMARK")[1]
        dest = ".".join(fname.split(".")[:-1])+".cpsol.dot"
        dest = dest.strip()
        dot = []
        j = i
        while not "vars" in content[j]:
            while not "digraph" in content[j]:
                j += 1
            dot = []
            while not "w =" in content[j-1]:
                dot.append(content[j])
                j += 1
            j += 1
        i = j
        print(cwd+dest)
        with open(cwd+dest,"w+") as f:
            for x in dot:
                f.write(x)
    i += 1
