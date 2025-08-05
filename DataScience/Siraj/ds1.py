from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

clf_tree = tree.DecisionTreeClassifier()
clf_knn = KNeighborsClassifier()

# CHALLENGE - create 3 more classifiers...
# 1
# 2
# 3

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf_tree = clf_tree.fit(X, Y)
clf_knn = clf_knn.fit(X, Y)

sample = [[190, 70, 43]]
prediction_tree = clf_tree.predict(sample)
prediction_knn = clf_knn.predict(sample)

# CHALLENGE compare their results and print the best one!

print(prediction_tree)
print(prediction_knn)

