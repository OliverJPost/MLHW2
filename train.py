from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def split_data(data, labels):
    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    return train_data, test_data, train_labels, test_labels


def train_svm_linear(train_data, train_labels):
    # Train a linear SVM
    svm_linear = SVC(kernel="linear")
    svm_linear.fit(train_data, train_labels)

    return svm_linear


def train_svm_poly(train_data, train_labels):
    param_grid_poly = {
        'C': [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1_000_000_000, 10_000_000_000],
        'degree': [2, 3, 4, 5],
        'kernel': ['poly']
    }

    # Create an instance of the SVM classifier
    svm_poly = SVC()

    # Set up GridSearchCV with cross-validation
    grid_search_poly = GridSearchCV(svm_poly, param_grid_poly, cv=5, verbose=2, n_jobs=-1)

    # Perform the grid search on the training data
    grid_search_poly.fit(train_data, train_labels)
    svm_poly = grid_search_poly.best_estimator_
    print("Found best parameters for poly kernel:")
    print(grid_search_poly.best_params_)
    svm_poly.fit(train_data, train_labels)

    return svm_poly


def train_svm_rbf(train_data, train_labels):
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1_000_000_000, 10_000_000_000,
              100_000_000_000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001,
                  0.00000000001, 0.000000000001],
        'kernel': ['rbf']
    }

    svm_rbf = SVC()

    # Set up GridSearchCV with cross-validation
    grid_search = GridSearchCV(svm_rbf, param_grid, cv=5, verbose=2, n_jobs=-1)

    # Perform the grid search on the training data
    grid_search.fit(train_data, train_labels)

    # Print the best hyperparameters
    print("Best hyperparameters found by grid search:")
    print(grid_search.best_params_)

    svm_rbf = grid_search.best_estimator_
    svm_rbf.fit(train_data, train_labels)
    return svm_rbf

def train_random_forest(train_data, train_labels):
    param_grid = {
        'n_estimators': [10, 50, 100, 200, 500, 1000],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    rfc = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    grid_search.fit(train_data, train_labels)

    print("Best hyperparameters found by grid search:")
    print(grid_search.best_params_)

    best_rf = grid_search.best_estimator_
    best_rf.fit(train_data, train_labels)

    return best_rf
