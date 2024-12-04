import itertools as it
from sys import maxsize
import multiprocessing as mp
import numpy as np


def load_problem(fname):
    with open(fname) as inp:
        dim = int(inp.readline())
        ls = np.fromstring(inp.readline().strip(), dtype=int, sep=' ')
        c = np.loadtxt(inp)
        c = np.maximum( c, c.transpose())
        i, j = c.shape

        return (ls, c)

def srflp_d(l, q, r, perm):
    length = l[perm[q]]/2 + l[perm[r]]/2
    # the number of possible elements between
    for s in range(q + 1, r):
        length = length + l[perm[s]]

    return length

def srflp_permutation(perm, instance, best_found):
    l, c = instance[0], instance[1]
    n = len(l)
    fit = 0

    for q in range(n - 1):
        for r in range(q + 1, n):
            #print('', perm[q], perm[r], c[perm[q]][perm[r]], srflp_d(l, q, r, perm))
            fit = fit + c[perm[q]][perm[r]] * srflp_d(l, q, r, perm)
            if fit >= best_found:
                return False, q
    return True, fit

def srflp_bnb(l, c, n, s, best_found, lock):
    perm = [a for a in range(n) if a != s]
    skip = False
    val = 0
    current = 0
    success = True
    count = 0

    for p in it.permutations(perm):
        perm = [s]
        perm.extend(p)
        count += 1

        if skip:
            if perm[val] == current:
                continue
            else:
                skip = False
                success, val = srflp_permutation(perm, (l,c), best_found)
        else:
            success, val = srflp_permutation(perm, (l,c), best_found)

        if success:
            with lock:
                 if val < best_found:
                     best_found = val
                     path = perm
        else:
          current = perm[val]
          skip = True
    #print(count)
    print(path, best_found)
    return(best_found)

def srflp():
    l, c = load_problem('Y-10_t.txt')
    n = len(l)
    Is = list(range(0, n))

    with mp.Manager() as manager:
        lock = manager.Lock()
        best_found = maxsize

        with mp.Pool(processes=4) as pool:
            ret = pool.starmap(srflp_bnb, zip(it.repeat(l), it.repeat(c), it.repeat(n), Is, it.repeat(best_found), it.repeat(lock)))

    print(ret)