import matplotlib.pyplot as plt

from sklearn import datasets

dataset = datasets.load_digits()

index = 122

plt.subplot(1, 1, 1)
plt.axis('off')
plt.imshow(dataset.images[index], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title('label: %i' % dataset.target[index])
plt.show()

print(dataset.images)

