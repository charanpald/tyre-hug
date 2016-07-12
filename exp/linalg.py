import numpy
import timeit
import matplotlib.pyplot as plt
import theano.tensor as T
import tensorflow
from theano import function

times = []
num_repeats = 20

x = T.dmatrix('x')
y = T.dmatrix('y')
z = T.dot(x, y)
f = function([x, y], z)

for i in range(100, 1001, 50):
    A = numpy.random.rand(i, i).astype(numpy.float32)
    B = numpy.random.rand(i, i).astype(numpy.float32)

    timer = timeit.Timer("numpy.dot(A, B)", "import numpy; from __main__ import A, B")
    numpy_times_list = timer.repeat(num_repeats, 1)

    timer = timeit.Timer("f(A, B)", "from __main__ import A, B, f")
    theano_times_list = timer.repeat(num_repeats, 1)

    # Set up the tensorflow stuff
    A = tensorflow.constant(A)
    B = tensorflow.constant(B)
    product = tensorflow.matmul(A, B)
    sess = tensorflow.Session(config=tensorflow.ConfigProto(log_device_placement=True))

    timer = timeit.Timer("sess.run(product)", "from __main__ import sess, product")
    tensorflow_times_list = timer.repeat(num_repeats, 1)

    times.append((i, numpy.min(numpy_times_list), numpy.min(theano_times_list), numpy.min(tensorflow_times_list)))
    sess.close()
    print(i)

times = numpy.array(times)
print(times)

plt.plot(times[:, 0], times[:, 1], 'k')
plt.plot(times[:, 0], times[:, 2], 'r')
plt.plot(times[:, 0], times[:, 3], 'b')
plt.show()
