import sklearn.datasets as datasets
import numpy as np
import pandas
samples = 10000
dataset_X,dataset_y = datasets.make_moons(n_samples=samples, noise=0.4)
dataset = pandas.DataFrame(dataset_X)
dataset["label"] = dataset_y
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_set = shuffled_indices[:test_set_size]
    train_set = shuffled_indices[test_set_size:]
    return data.iloc[train_set], data.iloc[test_set]


train_set, test_set = split_train_test(dataset, 0.2)

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

param_grid = {"max_depth":[2,3,4,5], "max_leaf_nodes":range(5,15,1)}

tree_grid = GridSearchCV(decision_tree,param_grid=param_grid,n_jobs=-1,verbose=3,cv=3)
tree_grid.fit(train_set.drop("label",axis=1), train_set["label"])
print(tree_grid.best_estimator_)
print(tree_grid.best_score_)

from sklearn.metrics import accuracy_score

print(accuracy_score(test_set["label"], tree_grid.predict(test_set.drop("label",axis=1),)))