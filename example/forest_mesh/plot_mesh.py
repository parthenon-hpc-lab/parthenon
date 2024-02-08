import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
import numpy as np 


fig, ax = plt.subplots()



tree = [(0.2, 0.1), (0.3, 0.2), (0.1, 0.2), (0.2, 0.3)]


tree1 = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]
tree2 = [(1.0, 0.0), (1.0, 1.0), (0.5, 0.0), (0.5, 0.5)]
tree3 = [(0.0, 0.5), (0.5, 0.5), (0.0, 1.0), (1.0, 1.0)]
#tree = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
def GetPosition(t, a1, a2):
  x = t[0][0]*(1 - a1)*(1 - a2) + t[1][0]*a1*(1-a2) + t[2][0]*a2*(1-a1) + t[3][0]*a1*a2;
  y = t[0][1]*(1 - a1)*(1 - a2) + t[1][1]*a1*(1-a2) + t[2][1]*a2*(1-a1) + t[3][1]*a1*a2;
  return (x, y)


#rect = patches.Rectangle((0.1, 0.2), 0.1, 0.2, linewidth=1, edgecolor='r', facecolor='none')
#rect = patches.Polygon(tree, linewidth=1, edgecolor='r', facecolor='none')
#ax.add_patch(rect) 
trees = [(tree1, "tree1.txt"), (tree2, "tree2.txt"), (tree3, "tree3.txt")]
for (t, fname) in trees:
  dat = np.loadtxt(fname, delimiter=",")
  for i in range(len(dat)): 
    h = 1.0 / 2**dat[i, 0] 
    a1 = dat[i, 1] * h
    a2 = dat[i, 2] * h
    rect = patches.Polygon([GetPosition(t, a1, a2), 
                            GetPosition(t, a1 + h, a2), 
                            GetPosition(t, a1 + h, a2 + h), 
                            GetPosition(t, a1, a2 + h)], 
                            linewidth=0.75, edgecolor='r', facecolor='none', alpha=1.0)
    ax.add_patch(rect)
  
  # Plot bounds of the tree
  rect = patches.Polygon([GetPosition(t, 0.0, 0.0), 
                          GetPosition(t, 1.0, 0.0), 
                          GetPosition(t, 1.0, 1.0), 
                          GetPosition(t, 0.0, 1.0)], 
                          linewidth=1, edgecolor='k', facecolor='none', alpha=1.0)
  ax.add_patch(rect)

  # Plot the logical axes of the tree
  start = GetPosition(t, 0.1, 0.1)
  end = GetPosition(t, 0.3, 0.1)
  plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1])
  end = GetPosition(t, 0.35, 0.1)
  plt.text(end[0], end[1], 'i')
  
  start = GetPosition(t, 0.1, 0.1)
  end = GetPosition(t, 0.1, 0.3)
  plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1])
  end = GetPosition(t, 0.1, 0.35)
  plt.text(end[0], end[1], 'j')

plt.axis("off")
plt.gca().set_aspect('equal')
plt.savefig("test.pdf", bbox_inches='tight')
