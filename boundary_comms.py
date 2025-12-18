class BndInfo:
    """Struct describing whether a var/comm buffer is allocated
    and where to send it"""
    def __init__(self, var_allocated,
                 buf_allocated,
                 same_to_same,
                 buf, var,
                 var_alloc_threshold,
                 ntopological_elements,
                 topological_idxs,
                 nt,nu,nv,
                 kbstart, kbend, jbstart, jbend, ibstart, ibend):
        self.var_allocated = var_allocated
        self.buf_allocated = buf_allocated
        self.same_to_same = same_to_same
        self.buf = buf
        self.var = var
        self.var_alloc_threshold
        self.topological_idxs = topological_idxs
        self.nt = nt
        self.nu = nu
        self.nv = nv
        self.kbstart = kbstart
        self.kbend = kbend
        self.jbstart = jbstart
        self.jbend = jbend
        self.ibstart = ibstart
        self.ibend = ibend

def SendBoundBufs(mesh, buffs, allbuffinfo):
    """copies the ghost cells on the 7D numpy array mesh
    into appropriate rows of the ragged 2D array buffs
    using allbuffinfo to describe copies"""

    numbuffs = mesh.shape[0]
    sending_nonzero = np.empty((numbuffs), dtype=bool)

    # In our code this is a kokkos parallel for
    # team policy.
    for b in range(numbuffs): 
        info = allbuffinfo[b]
        # In our code this needs to be atomic
        if info.same_to_same or not info.var_allocated:
            sending_nonzero[b] = False
            continue
        # This is a normal for loop in our code
        for iel in topological_idxs:
            # This is a parallel reduce over a team vector range in Kokkos
            for
            
