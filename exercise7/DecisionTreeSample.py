import sklearn.datasets as datasets
import numpy as np
import pandas
# Generating the dataset
samples = 10000
dataset_X,dataset_y = datasets.make_moons(n_samples=samples, noise=0.4)
dataset = pandas.DataFrame(dataset_X)
dataset["label"] = dataset_y

# Function to divide the dataset into train and test set.
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_set = shuffled_indices[:test_set_size]
    train_set = shuffled_indices[test_set_size:]
    return data.iloc[train_set], data.iloc[test_set]


train_set, test_set = split_train_test(dataset, 0.2)

# Decision Tree instantiation and GridSearch fit to find the best combination of hyperparameters
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

param_grid = {"max_depth":[2,3,4,5], "max_leaf_nodes":range(5,15,1)}

tree_grid = GridSearchCV(decision_tree,param_grid=param_grid,n_jobs=-1,verbose=3,cv=3)
tree_grid.fit(train_set.drop("label",axis=1), train_set["label"])
print(tree_grid.best_estimator_)
print(tree_grid.best_score_)

# Evaluation through prediction on the test_set
from sklearn.metrics import accuracy_score
print(accuracy_score(test_set["label"], tree_grid.predict(test_set.drop("label",axis=1),)))
predictions = pandas.DataFrame()
for i in range(1,10000):
    sub_decision_tree = DecisionTreeClassifier(max_depth=decision_tree.max_depth,max_leaf_nodes=decision_tree.max_leaf_nodes)
    shuffled_indices = np.random.permutation(len(train_set))
    subset = train_set.iloc[shuffled_indices[:100]]
    sub_decision_tree.fit(subset.drop("label",axis=1),subset["label"])
    sub_prediction = sub_decision_tree.predict(test_set.drop("label",axis=1))
    predictions[i] =  sub_prediction


from scipy import stats

test_set_t = predictions.values.T
finalPrediction = stats.mode(test_set_t)[0].astype(int)
print(finalPrediction.T)
print(accuracy_score(test_set["label"], finalPrediction.T))
