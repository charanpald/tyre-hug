import numpy
import matplotlib.pyplot as plt

# L2 norm
y = numpy.linspace(-1, 1, 100)
x = numpy.sqrt((1 - y**2))

plt.figure(figsize=(15, 7))
plt.subplot(121)
plt.plot(x, y, 'k', -x, y, 'k')
plt.grid()


# L1 norm
y = numpy.linspace(-1, 1, 100)
x = numpy.sqrt((1 - numpy.abs(y))**2)

plt.subplot(122)
plt.plot(x, y, 'k', -x, y, 'k')

plt.grid()
plt.show()
