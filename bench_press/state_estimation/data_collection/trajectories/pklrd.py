import pickle
import sys

a = pickle.load(open(sys.argv[1], 'rb'))

print(a)
