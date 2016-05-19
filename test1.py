import sys
from email import header

import scipy
import numpy
import matplotlib
import pandas
import sklearn

filename = 'fakeprofiledataset.csv'
numpytestfile = 'numpytest.csv'


# Lesson 1
# print('Python {}'.format(sys.version))
# print('Scipy {}'.format(scipy.__version__))
# print('numpy {}'.format(numpy.__version__))
# print('matplotlib {}'.format(matplotlib.__version__))
# print('pandas {}'.format(pandas.__version__))
# print('sklearn {}'.format(sklearn.__version__))

# Lesson 2
# myarray = numpy.array([[1,2,3],[4,5,6]])
# rownames = ['a','b']
# colnames = ['one','two','three']
# mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
# print(mydataframe)

# Lesson 3

# names = ['Age']
# pandas_data = pandas.read_csv(filename, names=names)
# descr = pandas_data.describe()
# print(descr)




# import csv
# f = open(filename, 'r')
# reader = csv.DictReader(f)  # <-- this is the shit
# for row in reader:
#     # print(','.join(row).encode('utf-8'))  # Encode met utf-8 fixt alot.
#     print(row['EmailAddress'])

import urllib
# numpy_data = numpy.loadtxt(numpytestfile, delimiter=',')
# for row in numpy_data:
#     for val in row:
#         # Someting Something --
#         exit()







