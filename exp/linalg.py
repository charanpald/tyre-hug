import numpy
import timeit
import matplotlib.pyplot as plt
import theano.tensor as T
import tensorflow
from theano import function


def benchmark_dot():
    times = []
    num_repeats = 20

    x = T.dmatrix('x')
    y = T.dmatrix('y')
    z = T.dot(x, y)
    global f
    f = function([x, y], z)

    for i in range(100, 1001, 50):
        global A, B
        A = numpy.random.rand(i, i).astype(numpy.float32)
        B = numpy.random.rand(i, i).astype(numpy.float32)

        timer = timeit.Timer("numpy.dot(A, B)", "import numpy; from __main__ import A, B")
        numpy_times_list = timer.repeat(num_repeats, 1)

        timer = timeit.Timer("f(A, B)", "from __main__ import A, B, f")
        theano_times_list = timer.repeat(num_repeats, 1)

        # Set up the tensorflow stuff
        A = tensorflow.constant(A)
        B = tensorflow.constant(B)
        global sess
        sess = tensorflow.Session()

        timer = timeit.Timer("sess.run(product)", setup="import tensorflow; from __main__ import sess, A, B; product = tensorflow.matmul(A, B)")
        tensorflow_times_list = timer.repeat(num_repeats, 1)
        sess.close()

        times.append((i, numpy.min(numpy_times_list), numpy.min(theano_times_list), numpy.min(tensorflow_times_list)))

        print(i)

    times = numpy.array(times)
    print(times)

    plt.plot(times[:, 0], times[:, 1], 'k', label="Numpy")
    plt.plot(times[:, 0], times[:, 2], 'r', label="Theano")
    plt.plot(times[:, 0], times[:, 3], 'b', label="Tensorflow")
    plt.legend()
    plt.show()


def benchmark_eig():
    times = []
    num_repeats = 20

    x = T.dmatrix('x')
    z = T.nlinalg.Eigh()(x)

    global f
    f = function([x], z)

    for i in range(100, 501, 50):
        global A
        A = numpy.random.rand(i, i).astype(numpy.float32)
        A = A.dot(A.T)

        timer = timeit.Timer("numpy.linalg.eigh(A)", "import numpy; from __main__ import A")
        numpy_times_list = timer.repeat(num_repeats, 1)

        timer = timeit.Timer("f(A)", "from __main__ import A, f")
        theano_times_list = timer.repeat(num_repeats, 1)

        # Set up the tensorflow stuff
        A = tensorflow.constant(A)
        global sess
        sess = tensorflow.Session()

        timer = timeit.Timer("sess.run(product)", setup="import tensorflow; from __main__ import sess, A; product = tensorflow.self_adjoint_eig(A)")
        tensorflow_times_list = timer.repeat(num_repeats, 1)

        times.append((i, numpy.min(numpy_times_list), numpy.min(theano_times_list), numpy.min(tensorflow_times_list)))
        sess.close()
        print(i)

    times = numpy.array(times)
    print(times)

    plt.plot(times[:, 0], times[:, 1], 'k', label="Numpy")
    plt.plot(times[:, 0], times[:, 2], 'r', label="Theano")
    plt.plot(times[:, 0], times[:, 3], 'b', label="Tensorflow")
    plt.legend()
    plt.show()


def benchmark_svd():
    times = []
    num_repeats = 20

    x = T.dmatrix('x')
    z = T.nlinalg.SVD()(x)

    global f
    f = function([x], z)

    for i in range(100, 501, 50):
        global A
        A = numpy.random.rand(i, i).astype(numpy.float32)

        timer = timeit.Timer("numpy.linalg.svd(A)", "import numpy; from __main__ import A")
        numpy_times_list = timer.repeat(num_repeats, 1)

        timer = timeit.Timer("f(A)", "from __main__ import A, f")
        theano_times_list = timer.repeat(num_repeats, 1)

        times.append((i, numpy.min(numpy_times_list), numpy.min(theano_times_list)))
        print(i)

    times = numpy.array(times)
    print(times)

    plt.plot(times[:, 0], times[:, 1], 'k', label="Numpy")
    plt.plot(times[:, 0], times[:, 2], 'r', label="Theano")
    plt.legend()
    plt.show()

benchmark_svd()
