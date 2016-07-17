import numpy
import timeit
import matplotlib.pyplot as plt
import sys
from settings import DATA_DIR, get_dir

numpy.random.seed(21)
num_repeats = 20
min_size = 500
max_size = 5001
step = 500

data_dir = get_dir(DATA_DIR)

def benchmark_numpy():
    times = []
    print("Benchmarking Numpy " + str(numpy.__version__))

    for i in range(min_size, max_size, step):
        print(i)
        global A, B
        A = numpy.random.rand(i, i).astype(numpy.float32)
        B = numpy.random.rand(i, i).astype(numpy.float32)

        current_times = [i]

        timer = timeit.Timer("numpy.dot(A, B)", "import numpy; from __main__ import A, B")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))

        timer = timeit.Timer("A + B", "import numpy; from __main__ import A, B")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))

        timer = timeit.Timer("numpy.exp(A)", "import numpy; from __main__ import A")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))

        timer = timeit.Timer("numpy.sum(A)", "import numpy; from __main__ import A")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))

        timer = timeit.Timer("numpy.mean(A)", "import numpy; from __main__ import A")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))

        timer = timeit.Timer("numpy.min(A)", "import numpy; from __main__ import A")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))

        times.append(current_times)

    times = numpy.array(times)
    print(times)
    numpy.save(data_dir + "numpy_times", times)


def benchmark_theano():
    from theano import function
    import theano
    import theano.tensor as T
    print("Benchmarking Theano " + str(theano.__version__))

    times = []

    for i in range(min_size, max_size, step):
        print(i)
        global A, B
        A = numpy.random.rand(i, i).astype(numpy.float32)
        B = numpy.random.rand(i, i).astype(numpy.float32)
        current_times = [i]

        global f
        A = theano.shared(A)
        B = theano.shared(B)

        z = T.dot(A, B)
        f = function([], z)
        timer = timeit.Timer("f()", "from __main__ import f")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))

        z = T.exp(A)
        f = function([], z)
        timer = timeit.Timer("f()", "from __main__ import f")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))

        z = T.sum(A)
        f = function([], z)
        timer = timeit.Timer("f()", "from __main__ import f")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))

        z = T.add(A, B)
        f = function([], z)
        timer = timeit.Timer("f()", "from __main__ import f")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))

        z = T.mean(A)
        f = function([], z)
        timer = timeit.Timer("f()", "from __main__ import f")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))

        z = T.min(A)
        f = function([], z)
        timer = timeit.Timer("f()", "from __main__ import f")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))

        times.append(current_times)

    times = numpy.array(times)
    print(times)
    numpy.save(data_dir + "theano_times", times)


def benchmark_tensorflow():
    import tensorflow
    print("Benchmarking Tensorflow " + str(tensorflow.__version__))

    times = []

    for i in range(min_size, max_size, step):
        print(i)
        global A, B
        A = numpy.random.rand(i, i).astype(numpy.float32)
        B = numpy.random.rand(i, i).astype(numpy.float32)
        current_times = [i]

        global f
        # Set up the tensorflow stuff
        A = tensorflow.constant(A)
        B = tensorflow.constant(B)
        global sess, result

        sess = tensorflow.Session()
        result = tensorflow.matmul(A, B)
        timer = timeit.Timer("sess.run(result)", setup="import tensorflow; from __main__ import sess, A, B, result")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))
        sess.close()

        sess = tensorflow.Session()
        result = tensorflow.exp(A)
        timer = timeit.Timer("sess.run(result)", setup="import tensorflow; from __main__ import sess, A, B, result")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))
        sess.close()

        sess = tensorflow.Session()
        result = tensorflow.exp(A)
        timer = timeit.Timer("sess.run(result)", setup="import tensorflow; from __main__ import sess, A, B, result")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))
        sess.close()

        sess = tensorflow.Session()
        result = tensorflow.add(A, B)
        timer = timeit.Timer("sess.run(result)", setup="import tensorflow; from __main__ import sess, A, B, result")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))
        sess.close()

        sess = tensorflow.Session()
        result = tensorflow.reduce_mean(A)
        timer = timeit.Timer("sess.run(result)", setup="import tensorflow; from __main__ import sess, A, B, result")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))
        sess.close()

        sess = tensorflow.Session()
        result = tensorflow.reduce_min(A)
        timer = timeit.Timer("sess.run(result)", setup="import tensorflow; from __main__ import sess, A, B, result")
        current_times.append(numpy.min(timer.repeat(num_repeats, 1)))
        sess.close()

        times.append(current_times)

    times = numpy.array(times)
    print(times)
    numpy.save(data_dir +  "tensorflow_times", times)


def plot_times():
    numpy_times = numpy.load(data_dir + "numpy_times.npy")
    theano_times = numpy.load(data_dir + "theano_times.npy")
    tensorflow_times = numpy.load(data_dir + "tensorflow_times.npy")

    titles = ["dot", "exp", "sum", "add", "mean", "min"]

    for i, title in enumerate(titles):
        plt.figure(i)
        plt.title(title)
        plt.plot(numpy_times[:, 0], numpy_times[:, i + 1], label="numpy")
        plt.plot(theano_times[:, 0], theano_times[:, i + 1], label="theano")
        plt.plot(tensorflow_times[:, 0], tensorflow_times[:, i + 1], label="tensorflow")
        plt.legend(loc="upper left")
        plt.xlabel("size")
        plt.ylabel("time (s)")
        plt.plot()

        # print(title)
        # print(numpy.c_[numpy_times[:, i + 1], theano_times[:, i + 1], tensorflow_times[:, i + 1]])

    plt.show()

if len(sys.argv) == 2 and sys.argv[1] == '1':
    benchmark_numpy()
elif len(sys.argv) == 2 and sys.argv[1] == '2':
    benchmark_theano()
elif len(sys.argv) == 2 and sys.argv[1] == '3':
    benchmark_tensorflow()
else:
    plot_times()
