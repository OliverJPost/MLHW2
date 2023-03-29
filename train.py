from sklearn.model_selection import train_test_split
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
    svm_poly = SVC(kernel="poly", degree=3)
    svm_poly.fit(train_data, train_labels)
    return svm_poly


def train_svm_rbf(train_data, train_labels):
    svm_rbf = SVC(kernel="rbf")
    svm_rbf.fit(train_data, train_labels)
    return svm_rbf
