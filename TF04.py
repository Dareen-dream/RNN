import FZ
from tensorflow.examples.tutorials.mnist import input_data

FZ.fozu()

print("111")
mnist = input_data.read_data_sets(r'C:\Users\mac\Downloads\mnist.npz', one_hot=True)
print("222")