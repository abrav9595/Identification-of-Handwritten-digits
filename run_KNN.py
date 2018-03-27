import pip
pip.main(['install', 'python-mnist']);

from mnist import MNIST

data = MNIST(".\\training_data");
train_images, train_labels = data.load_training();

data = MNIST(".\\test_data");
test_images, test_labels = data.load_testing();

import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plot

train_limit, test_limit = 6000, 1000;

k_values = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99];
error_rate = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

test_indices = range(len(test_labels)-1000, len(test_labels));
train_indices = range(0, 6000);

train_images, train_labels = np.asarray(train_images[0:6000]), np.asarray(train_labels[0:6000]);

for test_index in test_indices:
	test_data, test_label = test_images[test_index], test_labels[test_index];
		
	test_data = np.asarray(test_data)
	
	distances = np.argsort(np.sqrt(np.sum(train_images**2 + test_data**2 - 2 * np.multiply(train_images, test_data), axis=1)))[0:99]

	index = 0
	for k in k_values:
		top_values = distances[0:k]
		prediction = mode(train_labels[top_values.tolist()])[0][0]

		if int(prediction) is not int(test_label):
			error_rate[index] = error_rate[index] + 0.1;

		index=index+1;

plot.plot(k_values, error_rate, '-', label="K Nearest Neighbours")
plot.xlabel('K Value')
plot.ylabel('Error Rate %')
plot.show()