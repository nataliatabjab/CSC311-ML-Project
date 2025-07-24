import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    acc = None

    numbers = KNNImputer(n_neighbors=k)
    mat = numbers.fit_transform(matrix.T) 
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]
    user_accuracies = []
    item_accuracies = []

    # User-based collaborative filtering
    for i in k_values:
        accuracy = knn_impute_by_user(sparse_matrix, val_data, i)
        user_accuracies.append(accuracy)

    best_k_user = k_values[np.argmax(user_accuracies)]
    test_accuracy = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    
    print("Best k: ", best_k_user)
    print("Test accuracy: ", test_accuracy)
    
    # Plot for user
    plt.figure(figsize=(15, 10))
    plt.plot(k_values, user_accuracies, marker='o')
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("KNN for User")
    plt.grid(True)
    plt.show()
    
    # Item-based collaborative filtering
    for i in k_values:
        accuracy = knn_impute_by_item(sparse_matrix, val_data, i)
        item_accuracies.append(accuracy)

    best_k_item = k_values[np.argmax(item_accuracies)]
    test_accuracy_item = knn_impute_by_item(sparse_matrix, test_data, best_k_item)
    print("Best k: ", best_k_item)
    print("Test accuracy: ", test_accuracy_item)

    # Plot for item
    plt.figure(figsize=(15, 10))
    plt.plot(k_values, item_accuracies, marker='o')
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("KNN for Item")
    plt.grid(True)
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
