from anytree import Node, RenderTree, AsciiStyle
import numpy as np
a1 = Node ("a1")
b1 = Node ("b1", parent = a1)
b2 = Node ("b2", parent = a1)
c1 = Node ("c1", parent = b1)
c2 = Node ("c2", parent = b1)
c3 = Node ("c3", parent = b1)

print a1
print ""
print c1
print ""
print c2
print ""

# print (RenderTree(a1, style=AsciiStyle()))
