from random import random, randint, shuffle, choice, sample
from math import floor, ceil, log, fabs, sqrt
from scipy.linalg import eig, inv
from scipy.sparse import identity, diags, lil_matrix
from numpy import mean, zeros, uint8, real, imag, argmax, median, std
from numpy.random import normal, exponential
from PIL import Image
from sys import argv, exit
from string import ascii_lowercase
import networkx as nx
from networkx.algorithms import community
from collections import Counter, defaultdict

ZERO = 1**-15
DEBUG = False # True
FORCE = False

### <AUXILIARY ROUTINES>

def roulette(d):
    total = sum([value for (key, value) in d])
    cut = random() * total
    accum = 0
    for (key, value) in d:
        accum += value
        if accum >= cut:
            return key

def prefix(a, b): # length of shared prefix for two strings
    l = 0
    la = len(a)
    lb = len(b)
    lmin = min(la, lb)
    while l < lmin and a[l] == b[l]:
        l += 1
    return l, la + lb

def densities(i, e, k):
    ind = 0
    if k > 1:
        ind = 2 * i / (k * (k - 1))              
    t = i + e
    red = 0
    if t > 0:
        red = i / t
    return ind, red


### </AUXILIARY ROUTINES> 

class Node: # a class for representing the hierarchy trees
    def __init__(self, c, s = None, e = None, h = None):
        self.c = c
        if self.c is not None:
            for child in self.c:
                child.parent = self
        if s is not None:
            self.start = s # the start of the vertex interval at this node
        else:
            assert self.c is not None
            self.start = self.c[0].start
        if e is not None:
            self.end = e # the end of the vertex interval at this node
        else:
            assert self.c is not None
            self.end = self.c[-1].end
        if h is not None:
            self.h = h # the height of this node (that is, the number of levels beneath)
        else:
            assert self.c is not None
            self.h = max([child.h for child in self.c]) + 1
        self.d = None # the depth of this node (that is, the number of levels above)
        self.w = None # the width of this node (that is, the number of vertices in the branch)
        self.parent = None # the parent node (none if root)
        self.x = None # the horizontal coordinate at which this node is drawn on a dendrogram
        self.y = None # the vertical coordinate at which this node is drawn on a dendrogram
        self.l = None
                   
    def depth(self, d): 
        self.d = d
        if self.c is not None:
            for child in self.c:
                child.depth(d + 1)

    def label(self, l, storage):
        self.l = l
        if self.c is not None:
            for pos in range(len(self.c)):
                self.c[pos].label(l + ascii_lowercase[pos], storage)
        else:
            for vertex in range(self.start, self.end + 1):
                storage[vertex] = l

    def width(self):
        if self.w is None: # if the width is not yet computed
            if self.c is not None: 
                self.w = sum([child.width() for child in self.c])
            else:
                self.w = self.end - self.start + 1 # if there are no children, compute the length of the interval
        return self.w
    
    def __str__(self): # a string representation for debugging and such
        ch = ''
        if self.c is not None:
            ch = '|'.join([str(child) for child in self.c])
        return '({:d}-{:d})[{:s}]'.format(self.start, self.end, ch)

    def __repr__(self):
        return str(self)

    def dendrogram(self, x, xw, top, bot, dest): 
        self.x = x + xw / 2.0 # place the node horizontally in the middle of the allotted area
        self.y = top # place the node vertically at the top of the allotted area
        print('\\pscircle[fillstyle=solid,fillcolor=black]({:f},{:f}){{0.2}}'.format(self.x, self.y), file = dest)
        if self.parent is not None: # connect to the parent node if any
            print('\\psline[linewidth=2pt]({:f},{:f})({:f},{:f})'.format(self.parent.x, self.parent.y, \
                                                                         self.x, self.y), file = dest)
        curr = x
        if self.c is not None:
            yh = (self.h - 1) * top / self.h
            for child in self.c: # position the children accordng to their height and width
                aw = xw * child.width() / self.width()
                child.dendrogram(curr, aw, yh, bot, dest)
                curr += aw
        else: # if there are no children, draw the vertices included in the interval of this node
            wu = xw / self.width()
            for v in range(self.start, self.end):
                print('\\psline[linewidth=2pt]({:f},{:f})({:f},{:f})'.format(self.x, self.y, curr, bot), file = dest)
                print('\\pscircle[fillstyle=solid,fillcolor=white]({:f},{:f}){{0.2}}'.format(curr, bot), file = dest)
                print('\\rput{{90}}({:f},{:f}){{\\large {:s}}}'.format(curr, bot / 2, str(v + 1)), file = dest) # labels 1 to n
                curr += wu
        return (self.x, self.y)
              
class Graph: # a class for representing the graphs
    def __init__(self, n = 300, prob = 0.5, minOrder = 8):
        self.name = "G"
        self.adj = dict() # adjacency lists (out-going neighbors)
        self.order = n # number of vertices
        self.size = 0
        self.height = None
        third = n // 3 
        orders = {'h': [], 'f': []}
        for kind in 'hf':
            remaining = third
            while remaining >= minOrder:
                current = minOrder
                remaining -= minOrder
                while remaining > 0 and random() < prob:
                    current += 1
                    remaining -= 1
                if remaining < minOrder:
                    current += remaining # absorb all the rest
                    remaining = 0
                orders[kind].append(current)
        print(orders.values(), sum([sum(v) for v in orders.values()]), 2 * third)
        assert sum([sum(v) for v in orders.values()]) == 2 * third
        start = 1
        nodes = []
        for order in orders['h']:
            end = start + order - 1
            nodes.append(Node(None, start, end, 1)) # a leaf (i.e., a base community) at height one
            start = end + 1
        print([node for node in nodes])
        print([node.width() for node in nodes])
        assert sum([node.width() for node in nodes]) == third
        while len(nodes) > 1: # combine into a hierarchy
            pos = randint(0, len(nodes) - 2) # combine two neighboring nodes into a branch
            nodes = nodes[:pos] + [Node(nodes[pos:(pos+2)])] + nodes[(pos+2):]
        assert len(nodes) == 1
        self.root = nodes[0]
        assert self.root.width() == third
        self.root.depth(0)
        self.height = self.root.h                 
        self.communities = dict()  # the communities in which the vertices belong        
        self.root.label(ascii_lowercase[0], self.communities) # the hierarchy forms the "a" branch
        flats = orders['f']
        nodes = []
        while len(flats) > 0:
            current = flats.pop(0)
            while random() < prob and len(flats) > 0:
                current += flats.pop(0)
            nodes.append(current)
        start = third + 1 # the first 1/3rd were the hierarchical vertices
        i = 1
        for order in nodes:
            end = start + order - 1 
            for v in range(start, end + 1):
                self.communities[v] = ascii_lowercase[i] # b, c, d, etc.
            start = end + 1
            i += 1
        print(self.root)
        print(self.communities)
        for v in range(1, 2*third + 1):
            if v not in self.communities:
                print(v, "in", 2*third)
            assert v in self.communities
        for v in range(2*third + 1, n + 1):
            if v in self.communities:
                print(v, "out", 2*third)            
            assert v not in self.communities
        self.similarity = dict() # the hierarchical similarities of the vertices for the rewiring
        self.unionfind = dict()
        self.components = dict()
        for v in range(n):
            self.unionfind[v] = v
        self.span() 
        self.connect() 
        assert len(self.dfs(0)) == self.order
        self.noise()
        self.wire(minOrder) # preferential edge addition 
        for v in range(self.order):
            d = len(self.adj[v])
            labels = [self.communities.get(w, "NA") for w in self.adj[v]]
            own = self.communities.get(v, "NA")
            top_three = [x[0] for x in Counter(labels).most_common(3)]
            if own not in top_three and own != "NA":
                print('# WARNING', d, own, top_three)
        assert len(self.dfs(0)) == self.order
        self.density = 2 * self.size / (self.order * (self.order - 1))
        
    def community(self, label):
        return [v for v, c in self.communities.items() if c.startswith(label)]

    def summary(self):
        base = set(G.communities.values())
        allLevels = set()
        for label in base:
            while len(label) > 0:
                allLevels.add(label)
                label = label[:-1]
        intraLow = 0
        intraHigh = 0
        non = 0
        lowest = float('Inf')
        for label in sorted(list(allLevels)):
            mem = self.community(label)
            smallest = min(mem)
            largest = max(mem)
            for vertex in range(smallest, largest + 1):
                assert vertex in mem
            i, e, k = self.quality(mem)
            ind, red = densities(i, e, k)
            if label in base:
                intraLow += i # number of community members to which the members are connected
                non += k * (self.order - k) - e # number of non-community members to which the members are _not_ connected 
            if 'a' not in label:
                intraHigh += i
                print("The flat base community {:s} covers {:.2f} % with {:d} members with a quality of {:.3f} and {:.3f} ({:d}-{:d}) #".format(label, 100 * k / self.order, k, ind, red, smallest, largest))
            else:
                level = len(label)
                kind = "base" if label in self.communities.values() else "meta"
                if level == 1:
                    intraHigh += i
                    if ind * red < lowest:
                        lowest = ind * red
                print("Level {:d} {:s} community {:s} covers {:.2f} % with {:d} members with a quality of {:.3f} and {:.3f} ({:d}-{:d}) #".format(level, kind, label, 100 * k / self.order, k, ind, red, smallest, largest))
        outcasts = set(self.adj.keys()) - set(self.communities.keys())
        i, e, k = self.quality(outcasts)
        ind, red = densities(i, e, k)
        non += k * (self.order - k) - e
        non /= 2 # every non-internal edge was counted twice
        print("The uniform outcast vertices cover {:.2f} % with {:d} vertices with a quality of {:.3f} and {:.3f} ({:d}-{:d}) #".format(100 * k / self.order, k, ind, red, smallest, largest))
        assert lowest > ind * red
        if DEBUG:
            print(base, intraLow, intraHigh, non, self.size)
        print('# The coverage [1] of the lowest-level partitions is {:.2f}'.format(intraLow / self.size))
        print('# The coverage [1] of the highest-level partitions is {:.2f}'.format(intraHigh / self.size))
        mMax = self.order * (self.order - 1) / 2
        print('# The performance [1] of the lowest-level partitions is {:.2f}'.format((intraLow + non) / mMax))
        print('# The performance [1] of the highest-level partitions is {:.2f}'.format((intraHigh + non) / mMax))
        print('# [1] S. Fortunato: Community Detection in Graphs. Physical Reports 486(3–5): 75-174, 2010.')

    def quality(self, members):
        order = len(members)
        if order < 2:
            return 0, 0, order
        internal = 0
        external = 0
        for v in members:
            for w in self.adj[v]:
                if w in members:
                    internal += 1
                else:
                    external += 1
        assert internal % 2 == 0
        return internal / 2, external, order
        
    def subgraph(self, pr):
        for v in self.communities:
            if v in self.communities and self.communities[v].startswith(pr):
                for w in self.adj[v]:
                    if w in self.communities and self.communities[w].startswith(pr):                    
                        print(v, w)
    
    def separate(self, v, w):
        assert w in self.adj[v]
        assert v in self.adj[w]
        self.adj[v].remove(w)
        self.adj[w].remove(v)
        assert w not in self.adj[v]
        assert v not in self.adj[w]        
        self.size -= 1
        assert len(self.adj[v]) > 0
        assert len(self.adj[w]) > 0
        
    def join(self, v, w): # connect the source and the target vertices with an edge
        if v == w:
            return False # no reflexive edges 
        if v not in self.adj:
            self.adj[v] = set()
        if w not in self.adj:
            self.adj[w] = set()
        if w in self.adj[v] or v in self.adj[w]:
            assert v in self.adj[w]
            assert w in self.adj[v]
            return False
        self.adj[v].add(w)
        self.adj[w].add(v)
        self.size += 1
        if self.unionfind[v] != self.unionfind[w]:
            kv = self.unionfind[v]
            kw = self.unionfind[w]
            replacement = self.components.get(kv, {v}) | self.components.get(kw, {w})
            self.components.pop(kv, None)
            self.components.pop(kw, None)
            key = min(replacement)
            for u in replacement:
                self.unionfind[u] = key
            self.components[key] = replacement
        return True

    def __repr__(self): # a string representation for debugging
        return '\n'.join(['{:d} {:s} {:d}'.format(v, str(self.adj[v]), \
                                                  len(self.adj[v])) for v in self.adj])

    def sim(self, v1, v2, epsilon = 0.001): # compute the common-prefix similarity for two vertices
        u = min(v1, v2)
        v = max(v1, v2)
        if u not in self.communities or v not in self.communities:
            return 2 * epsilon
        lu = self.communities[u]
        lv = self.communities[v]
        if (lu, lv) in self.similarity:
            return self.similarity[(lu, lv)]
        elif lu == lv:
            self.similarity[(lu, lv)] = 1            
            return 1
        shared, total = prefix(lu, lv)
        if shared == 0:
            self.similarity[(lu, lv)] = epsilon
            return epsilon
        s = 2 * shared / total 
        self.similarity[(lu, lv)] = s**2
        return s

    ### <ROUTINES FOR DRAWINGS>

    def graphml(self, dest):
        print('<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">', file = dest)
        print('<graph id="G" edgedefault="undirected">', file = dest)
        print('<key id="cluster" for="node" attr.name="cluster" attr.type="string">', file = dest)
        print('<default>NA</default>\n</key>', file = dest)
        k = 1
        for v in self.adj:
            print('<node id="{:d}">'.format(v), file = dest)
            if v in self.communities:
                print('<data key="cluster">c_{:s}</data>'.format(self.communities[v]), file = dest) # member
            else:
                print('<data key="cluster">s_{:d}</data>'.format(v), file = dest) # singleton
            print('</node>', file = dest)
        for v in self.adj:
            for w in self.adj[v]:
                if v < w: # undirected 
                    print('<edge id="e{:d}" source="{:d}" target="{:d}"/>'.format(k, v, w), file = dest)
                    k += 1
        print('</graph>\n</graphml>', file = dest)
    
    def adj2png(self, target): # draw the adjacency matrix as a PNG file
        d = self.order
        fig = zeros((d,d), uint8)
        for r in range(d):
            for c in range(d):
                if c in self.adj[r]:
                    fig[r, c] = 0 # black pixel
                else: 
                    fig[r, c] = 255 # white pixel
        (Image.fromarray(fig)).save(target) # save into a file 

    def adj2tex(self, dest, width): # draw the adjacency matrix as a PSTricks figure
        print('\\resizebox{{{:d}cm}}{{!}}{{'.format(width), file = dest) # direct output to a file
        print('\\begin{{pspicture}}({:d},{:d})'.format(self.order, self.order), file = dest)
        print('\\psframe[linecolor=black](0, 0)({:d},{:d})'.format(self.order, self.order), file = dest)
        for i in range(self.order):
            row = self.order - i - 1
            for j in range(self.order):
                if j in self.adj[i]: # draw a black box for each edge
                    print('\\psframe*[linecolor=black]({:d},{:d})({:d},{:d})'.format(j, row, \
                                                                                    j + 1, row + 1), \
                          file = dest)
        print('\\end{pspicture}}', file = dest)

    def dendrogram(self, d, width, standalone = True):
        unit = 1
        wu = unit
        n = self.order
        hu = int(ceil(1.5 * max(ceil(log(log(n, 10) + 1, 2)), ceil(log(self.height + 1, 10))) * unit))
        w = wu * (n + 2)
        h = hu * (self.height + 2)
        y = h - hu / 2
        pos = []
        if standalone:
            print('''\\documentclass{article}
            \\usepackage{color}
            \\usepackage{pstricks}
            \\usepackage{graphicx}
            \\begin{document}
            \\centering
            \\begin{figure}''', file = d)
        print('\\resizebox{{{:d}cm}}{{!}}{{'.format(width), file = d)
        print('\\begin{{pspicture}}({:d},{:d})'.format(w, h), file = d)
        pos.append(self.root.dendrogram(0, wu * self.root.width(), h - hu, hu, d))
        xs = min([x for (x, y) in pos])
        xe = max([x for (x, y) in pos])
        for (rx, ry) in pos:
            print('\\psline[linewidth=2pt]({:f},{:f})({:f},{:f})'.format(rx, y, rx, ry), file = d)
        print('\\psline[linewidth=2pt]({:f},{:f})({:f},{:f})'.format(xs, y, xe, y), file = d)
        sx = sorted([x for (x, y) in pos])
        ls = len(sx)
        if ls % 2 == 0:
            mx = (sx[len(sx) // 2 - 1] + sx[len(sx) // 2]) / 2.0
        else:
            mx = sx[len(sx) // 2] 
        print('\\psline[linewidth=2pt]({:f},{:f})({:f},{:f})'.format(mx, y, mx, h), file = d)
        print('\\pscircle[fillstyle=solid,fillcolor=black]({:f},{:f}){{0.2}}'.format(mx, h), file = d)
        print('\\end{pspicture}}', file = d)        
        if standalone:
            print('''\\end{figure}
            \\end{document}''', file = d)

    def span(self):
        comm = set()
        for c in self.communities.values(): 
            while len(c) > 0:
                comm.add(c)
                c = c[:-1]
        coms = sorted(list(comm), key=len, reverse=True) # lowest-level communities first
        for c in coms: # ensure connectivity of each community
            mem = self.community(c)
            while len(set([self.unionfind[v] for v in mem])) > 1:
                chosen = sample(mem, 2)
                v = chosen.pop()
                w = chosen.pop()
                if self.unionfind[v] != self.unionfind[w]:
                    self.join(v, w)

    def noise(self, dens = 0.001):
        for v in range(self.order - 1):
            for w in range(v + 1, self.order):
                if random() < dens:
                    self.join(v, w)
                    
    def wire(self, targetDegree):
        missing = dict()
        for v in range(self.order):
            degree = len(self.adj[v])
            if degree < targetDegree:
                missing[v] = targetDegree - degree
        while len(missing) > 0:
            v = None
            if len(missing) > 1:
                v = roulette(missing.items())
            else: # the last one left
                v = list(missing.keys()).pop(0)
            candidates = self.adj.keys() - self.adj[v] # not yet neighbors of v
            candidates.discard(v) # not v itself
            w = roulette([(c, self.sim(v, c)) for c in candidates])
            self.join(v, w)
            missing[v] -= 1
            if missing[v] == 0:
                del missing[v]
            if w in missing:
                missing[w] -= 1
                if missing[w] == 0:
                    del missing[w]

    def dfs(self, start):
        visited = set()
        stack = [start]
        while len(stack) > 0:
            curr = stack[-1]
            visited.add(curr)
            unvisited = list(self.adj[curr] - visited)
            if len(unvisited) > 0:
                stack.append(choice(unvisited))
            else:
                stack.pop()
        return visited
        
    def connect(self):
        while True:
            pending = list(set(self.unionfind.values()))
            if len(pending) == 1:
                return
            s = choice(pending)
            t = choice(pending)
            if s != t:
                v = choice(list(self.components.get(s, {s})))
                w = choice(list(self.components.get(t, {t})))
                assert self.unionfind[v] != self.unionfind[w]
                self.join(v, w)

    def walklen(self, seed):
        curr = seed
        visited = {seed}
        while True:
            cand = (self.adj[curr] - visited) & self.adj[seed]
            if len(cand) == 0:
                break # dead end
            curr = choice(list(cand)) # pick one at random
            visited.add(curr)
        return len(visited) / (1 + len(self.adj[seed]))
            
    def kind(self, v):
        k = 'Uniform'
        d = None
        label = 'NA'
        if v in self.communities:
            label = self.communities[v]
            if 'a' not in label:
                k = 'Flat'
                d = 0
            else:
                k = 'Hierarchical'
                d = len(label)
        return (k, d, label)
    
    def analyze(self, d):
        with open('analysis.csv', 'w') as output:
            for v in range(self.order): # for each vertex
                (ki, de, la) = self.kind(v)
                print(la, ki, de, d[v], v, len(self.adj[v]), v, file = output)
                        
    def draw(self, forms): # draw the graph in the requested format
        for form in forms:
            if form == "tex":
                with open("adj.tex", 'w') as d:
                    self.adj2tex(d, 5)
                    with open("dendro.tex", 'w') as d:
                        self.dendrogram(d, 10)
            elif form == "png":
                self.adj2png("adj.png")
            elif form == 'gml':
                with open(self.name + ".graphml", 'w') as d:
                    self.graphml(d)
            else:
                print('Drawing formats available: tex, png, gml')
        

    def kl(self, G, scores=[], depth=0):
        U, V = community.kernighan_lin_bisection(G)
        lU = set([self.communities.get(u, 'NA') for u in U]) - {'NA'}
        lV = set([self.communities.get(v, 'NA') for v in V]) - {'NA'}
        shared = lU & lV
        match = 0
        mismatch = 0
        if len(shared) > 0:
            for label in shared:
                mem = set(self.community(label))
                ku = len(mem & U) # one side
                kv = len(mem & V) # the other
                mismatch += ku * kv
                match += ku * ku + kv * kv
        for label in lU | lV - shared: # the ones that were not split
            mem = set(self.community(label))
            k = len(mem)
            match += k * k 
        agreement = match / (match + mismatch)
        scores.append(agreement**(1/(depth + 1)))
        if len(lU) > 1:
            scores = self.kl(G.subgraph(U), scores, depth + 1)
        if len(lV) > 1:
            scores = self.kl(G.subgraph(V), scores, depth + 1)
        return scores
            
    def validate(self): # validation with networkx
        G = nx.Graph()
        for v in self.adj:
            for w in self.adj[v]:
                G.add_edge(v, w)
        return mean(self.kl(G))

G = Graph()
agree = G.validate()
print('# Average depth-sensitive agreement with KL is {:.4f}'.format(agree))
if agree < 0.975:
    print('Aborting due to poor agreement with KL.')
    exit(2) # poor agreement with KL
G.summary()
d = G.density
me = dict()
C = dict()
reps = int(ceil(log(G.order, 2)))
for v in sorted(G.adj.keys()):
    (vk, vd, la) = G.kind(v)
    if vd is None:
        vd = "NA" # for R processing
    me[v] = max([G.walklen(v) for r in range(reps)])
    print(v, d, la, len(G.adj[v]), vk, vd, me[v])
G.analyze(me) 
if 'quiet' not in argv:
    r = input("Draw [Y/N/Q]: ").lower()
    if 'y' in r:
        G.draw(input("Type [tex/png/gml]: " ).split()) # visualization of the graph
    elif 'q' in r:
        quit()
    r = input("Reduce to a subgraph [Y/N/Q]: ").lower()
    if 'y' in r:
        G.subgraph(input("Prefix: "))
    
