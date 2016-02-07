import time
import numpy
import sppy
import sppy.linalg
import matplotlib.pyplot as plt
import scipy.sparse
import os
from scipy.sparse.linalg import svds
from pypropack import svdp
from sparsesvd import sparsesvd
from sklearn.decomposition import TruncatedSVD
from sppy.linalg import GeneralLinearOperator


def time_reps(func, params, reps):
    start = time.time()
    for s in range(reps):
        func(*params)
    print("Executed " + str(func))
    return (time.time()-start)/reps

"""
Compare and contrast different methods of computing the SVD
"""

# Fix issue with CPU affinity
os.system("taskset -p 0xff %d" % os.getpid())

k = 50
p = 0
reps = 2
density = 10**-3
n_iter = 5
truncated_svd = TruncatedSVD(k, n_iter=5)
var_range = numpy.arange(1, 11)


def time_densities():
    n = 10**4
    densities = var_range * 2 * 10**-3
    times = numpy.zeros((5, densities.shape[0]))

    for i, density in enumerate(densities):
        # Generate random sparse matrix
        inds = numpy.random.randint(n, size=(2, n * n * density))
        data = numpy.random.rand(n * n * density)
        A = scipy.sparse.csc_matrix((data, inds), (n, n))
        A_sppy = sppy.csarray(A, storagetype="row")
        L = GeneralLinearOperator.asLinearOperator(A_sppy, parallel=True)
        print(A.shape, A.nnz)

        times[0, i] = time_reps(svds, (A, k), reps)
        times[1, i] = time_reps(svdp, (A, k), reps)
        # Remove SparseSVD since it is significantly slower than the other methods
        # times[2, i] = time_reps(sparsesvd, (A, k), reps)
        times[3, i] = time_reps(truncated_svd.fit, (A,), reps)
        times[4, i] = time_reps(sppy.linalg.rsvd, (L, k, p, n_iter), reps)
        print(n, density, times[:, i])

    plt.figure(0)
    plt.plot(densities, times[0, :], 'k-', label="ARPACK")
    plt.plot(densities, times[1, :], 'r-', label="PROPACK")
    # plt.plot(densities, times[2, :], 'b-', label="SparseSVD")
    plt.plot(densities, times[3, :], 'k--', label="sklearn RSVD")
    plt.plot(densities, times[4, :], 'r--', label="sppy RSVD")
    plt.legend(loc="upper left")
    plt.xlabel("density")
    plt.ylabel("time (s)")
    plt.savefig("time_densities.png", format="png")


# Next, vary the matrix size
def time_ns():
    density = 10**-3
    ns = var_range * 10**4
    times = numpy.zeros((5, ns.shape[0]))

    for i, n in enumerate(ns):
        # Generate random sparse matrix
        inds = numpy.random.randint(n, size=(2, n * n * density))
        data = numpy.random.rand(n * n * density)
        A = scipy.sparse.csc_matrix((data, inds), (n, n))
        A_sppy = sppy.csarray(A, storagetype="row")
        L = GeneralLinearOperator.asLinearOperator(A_sppy, parallel=True)
        print(A.shape, A.nnz)

        times[0, i] = time_reps(svds, (A, k), reps)
        times[1, i] = time_reps(svdp, (A, k), reps)
        # times[2, i] = time_reps(sparsesvd, (A, k), reps)
        times[3, i] = time_reps(truncated_svd.fit, (A,), reps)
        times[4, i] = time_reps(sppy.linalg.rsvd, (L, k, p, n_iter), reps)
        print(n, density, times[:, i])

    plt.figure(1)
    plt.plot(ns, times[0, :], 'k-', label="ARPACK")
    plt.plot(ns, times[1, :], 'r-', label="PROPACK")
    # plt.plot(ns, times[2, :], 'b-', label="SparseSVD")
    plt.plot(ns, times[3, :], 'k--', label="sklearn RSVD")
    plt.plot(ns, times[4, :], 'r--', label="sppy RSVD")
    plt.legend(loc="upper left")
    plt.xlabel("n")
    plt.ylabel("time (s)")
    plt.savefig("time_ns.png", format="png")


def time_ks():
    n = 10**4
    density = 10**-3
    ks = var_range * 20
    times = numpy.zeros((5, ks.shape[0]))

    for i, k in enumerate(ks):
        # Generate random sparse matrix
        inds = numpy.random.randint(n, size=(2, n * n * density))
        data = numpy.random.rand(n * n * density)
        A = scipy.sparse.csc_matrix((data, inds), (n, n))
        A_sppy = sppy.csarray(A, storagetype="row")
        L = GeneralLinearOperator.asLinearOperator(A_sppy, parallel=True)
        print(A.shape, A.nnz)

        times[0, i] = time_reps(svds, (A, k), reps)
        times[1, i] = time_reps(svdp, (A, k), reps)
        # times[2, i] = time_reps(sparsesvd, (A, k), reps)
        truncated_svd = TruncatedSVD(k, n_iter=5)
        times[3, i] = time_reps(truncated_svd.fit, (A,), reps)
        times[4, i] = time_reps(sppy.linalg.rsvd, (L, k, p, n_iter), reps)
        print(n, density, times[:, i])

    plt.figure(2)
    plt.plot(ks, times[0, :], 'k-', label="ARPACK")
    plt.plot(ks, times[1, :], 'r-', label="PROPACK")
    # plt.plot(ks, times[2, :], 'b-', label="SparseSVD")
    plt.plot(ks, times[3, :], 'k--', label="sklearn RSVD")
    plt.plot(ks, times[4, :], 'r--', label="sppy RSVD")
    plt.legend(loc="upper left")
    plt.xlabel("k")
    plt.ylabel("time (s)")
    plt.savefig("time_ks.png", format="png")

time_densities()
time_ks()
time_ns()
plt.show()
