from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = theta[u] - beta[q]
        log_lklihood += data["is_correct"][i] * x - np.log(1 + np.exp(x))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = theta[u] - beta[q]
        p_a = sigmoid(x)
        theta[u] += lr * (data["is_correct"][i] - p_a)
        beta[q] -= lr * (data["is_correct"][i] - p_a)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # Initialize theta and beta using the maximum values from both training and validation data
    max_user = max(max(data["user_id"]), max(val_data["user_id"])) + 1
    max_question = max(max(data["question_id"]), max(val_data["question_id"])) + 1
    
    theta = np.random.normal(size=max_user)
    beta = np.random.normal(size=max_question)

    val_acc_lst = []
    val_log_lst = []
    trn_log_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        trn_log_lst.append(-neg_lld)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)

        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_log_lst.append(-val_neg_lld)

        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, val_acc_lst, val_log_lst, trn_log_lst

# Helper function to evaluate algorithm extension in Part B
def evaluate_2pl(data, theta, alpha, beta):
    predictions = []
    for i, q in enumerate(data["question_id"]):
        predictions.append(sigmoid(alpha[q] * (theta[data["user_id"][i]] - beta[q])) >= 0.5)
    return np.mean(data["is_correct"] == np.array(predictions))

# Algorithm for Part B
def irt_2pl(train_data, val_data, item_features, lr, iterations, n_components, tau):
    max_question = max(max(train_data["question_id"]), max(val_data["question_id"])) + 1

    # Cluster mean initialization
    pca = PCA(n_components=n_components)
    item_reduced = pca.fit_transform(item_features)
    kmeans = KMeans(n_clusters=20, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(item_reduced)

    # Initialization
    theta, beta, _, _, _ = irt(train_data, val_data, 0.01, 50)
    alpha = np.ones(max_question)
    mu_alpha = np.array([np.mean(alpha[clusters == k]) for k in range(20)])
    mu_beta = np.array([np.mean(beta[clusters == k]) for k in range(20)])

    # Adam optimizer initialization
    theta_a = torch.tensor(theta, dtype=torch.float32, requires_grad=True)
    alpha_a = torch.tensor(alpha, dtype=torch.float32, requires_grad=True)
    beta_a = torch.tensor(beta, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([theta_a, alpha_a, beta_a], lr=lr)

    validation_acc, train_log_likelihood = [], []

    for i in range(1, iterations + 1):
        optimizer.zero_grad()
        log_likelihood = torch.tensor(0.0, dtype=torch.float32)
        # Update student abilities
        for u, q, c in zip(train_data["user_id"], train_data["question_id"], train_data["is_correct"]):
            p = torch.sigmoid(alpha_a[q] * (theta_a[u] - beta_a[q]))
            log_likelihood += c * torch.log(p + 1e-5) + (1 - c) * torch.log(1 - p + 1e-5)

         # Update item parameters
        for q in range(max_question):
            cluster = clusters[q]
            log_likelihood -= ((alpha_a[q] - mu_alpha[cluster]) ** 2) / (2 * tau**2)
            log_likelihood -= ((beta_a[q] - mu_beta[cluster]) ** 2) / (2 * tau**2)

        (-log_likelihood).backward()
        optimizer.step()

        # Adam update for alpha
        with torch.no_grad():
            alpha_a.clamp_(0.1, 5.0)

         # Adam update for beta 
        alpha = alpha_a.detach().numpy()
        beta = beta_a.detach().numpy()
        theta = theta_a.detach().numpy()

        # Recompute cluster means
        for k in range(20):
            if np.any(clusters == k):
                mu_alpha[k] = np.mean(alpha[clusters == k])
                mu_beta[k] = np.mean(beta[clusters == k])

        train_log_likelihood.append(log_likelihood.item())
        validation_acc.append(evaluate_2pl(val_data, theta, alpha, beta))

        # To print validation accuracies: print("Iteration: ", i, "Validation accuracy:", validation_acc[-1])

    return theta, alpha, beta, validation_acc, train_log_likelihood

# Hypothesis testing for Part B question 3
def run_experiment(train_data, val_data, test_data, item_features, lr, iterations):
    results = {}
    # Run experiment on all training data
    theta_1pl, beta_1pl, validation_acc_1pl, _, _ = irt(train_data, val_data, lr, iterations)
    test_accuracy_1pl = evaluate(test_data, theta_1pl, beta_1pl)
    results["1PL_full"] = (validation_acc_1pl[-1], test_accuracy_1pl)

    theta_2pl, alpha_2pl, beta_2pl, validation_acc_2pl, _ = irt_2pl(train_data, val_data, item_features, lr=lr, iterations=iterations, n_components=10, tau=0.1)
    test_accuracy_2pl = evaluate_2pl(test_data, theta_2pl, alpha_2pl, beta_2pl)
    results["2PL_full"] = (validation_acc_2pl[-1], test_accuracy_2pl)

    # Run experiment on 50% of the trainin data
    reduced_index = np.random.choice(len(train_data["user_id"]), size=len(train_data["user_id"]) // 2, replace=False)
    reduced_train_data = {"user_id": np.array(train_data["user_id"])[reduced_index], "question_id": np.array(train_data["question_id"])[reduced_index],
                          "is_correct": np.array(train_data["is_correct"])[reduced_index]}

    theta_1pl_r, beta_1pl_r, validation_acc_1pl_r, _, _ = irt(reduced_train_data, val_data, lr, iterations)
    test_accuracy_1pl_r = evaluate(test_data, theta_1pl_r, beta_1pl_r)
    results["1PL_reduced"] = (validation_acc_1pl_r[-1], test_accuracy_1pl_r)

    theta_2pl_r, alpha_2pl_r, beta_2pl_r, validation_acc_2pl_r, _ = irt_2pl(reduced_train_data, val_data, item_features, lr=lr, iterations=iterations, n_components=10, tau=0.1)
    test_accuracy_2pl_r = evaluate_2pl(test_data, theta_2pl_r, alpha_2pl_r, beta_2pl_r)
    results["2PL_reduced"] = (validation_acc_2pl_r[-1], test_accuracy_2pl_r)

    # Plot graphs
    plt.figure(figsize=(15, 10))
    plt.bar(["1PL (Full)", "2PL (Full)", "1PL (Reduced)", "2PL (Reduced)"], [results["1PL_full"][0], results["2PL_full"][0],
             results["1PL_reduced"][0], results["2PL_reduced"][0]], color=["blue", "orange", "blue", "orange"], alpha=0.7)
    plt.ylabel("Validation Accuracy")
    plt.title("Regularization Hypothesis Test")
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.bar(["1PL (Full)", "2PL (Full)", "1PL (Reduced)", "2PL (Reduced)"], [results["1PL_full"][1], results["2PL_full"][1],
             results["1PL_reduced"][1], results["2PL_reduced"][1]], color=["blue", "orange", "blue", "orange"], alpha=0.7)
    plt.ylabel("Test Accuracy")
    plt.title("Regularization Hypothesis Test")
    plt.show()

    return results

def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    bst_val_acc = 0.0
    bst_lr = 0.0
    bst_iter = 0

    for lr in [0.001, 0.01, 0.1]:
        for iterations in [50, 100, 200]:
            theta, beta, val_acc_lst, val_log_lst, trn_log_lst = irt(
                train_data, val_data, lr=lr, iterations=iterations
            )
            val_acc = val_acc_lst[-1]
            if val_acc > bst_val_acc:
                bst_val_acc = val_acc
                bst_lr = lr
                bst_iter = iterations

    print("Best Learning Rate:", bst_lr)
    print("Best Number of Iterations:", bst_iter)
    print("Best Validation Accuracy:", bst_val_acc)
    test_acc = evaluate(test_data, theta, beta)
    print("Test Accuracy:", test_acc)

    iterations = range(bst_iter)
    plt.plot(iterations, trn_log_lst, label='Training Log-likelihood')
    plt.plot(iterations, val_log_lst, label='Validation Log-likelihood')
    plt.xlabel('# of Iterations')
    plt.ylabel('Log-likelihood')
    plt.title('Training & Validation Log-likelihood vs. Number of Iterations')
    plt.legend()
    plt.savefig("q2b.png")

    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)
    questions = [100, 500, 1500]
    plt.figure()

    for question in questions:
        abilities = np.linspace(-3, 3, 100)
        probabilities = []
        for theta in abilities:
            p_correct = sigmoid(theta - beta[question])
            probabilities.append(p_correct)
        plt.plot(abilities, probabilities, label=f"Question {question}")

    plt.title("Probability of Correct Response vs. Ability")
    plt.xlabel("Ability in terms of theta")
    plt.ylabel("Probability of Correct Response")
    plt.legend()
    plt.grid(True)
    plt.savefig("q2d.png")
    #####################################################################

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    ### Part B Question 3 ###
    # theta_1pl, beta_1pl, validation_acc_1pl, _, _ = irt(train_data, val_data, lr=0.01, iterations=50)
    # test_accuracy_1pl = evaluate(test_data, theta_1pl, beta_1pl)

    # num_questions = max(max(train_data["question_id"]), max(val_data["question_id"])) + 1
    # item_features = np.eye(num_questions) 
    # theta_2pl, alpha_2pl, beta_2pl, validation_acc_2pl, _ = irt_2pl(train_data, val_data, item_features, lr=0.001, iterations=50, n_components=10, tau=0.1)
    # test_accuracy_2pl = evaluate_2pl(test_data, theta_2pl, alpha_2pl, beta_2pl)

    # models = ["1PL IRT", "2PL IRT"]
    # val_accuracies = [validation_acc_1pl[-1], validation_acc_2pl[-1]]
    # test_accuracies = [test_accuracy_1pl, test_accuracy_2pl]

    # x = np.arange(len(models))
    # plt.figure(figsize=(15, 10))
    # plt.bar(x - 0.4/2, val_accuracies, 0.4, label="Validation Accuracy")
    # plt.bar(x + 0.4/2, test_accuracies, 0.4, label="Test Accuracy")
    # plt.xticks(x, models)
    # plt.ylabel("Accuracy")
    # plt.ylim(0, 1)
    # plt.title("Comparing 1PL IRT and 2PL IRT")
    # plt.legend()
    # plt.show()

    # # Run experiment
    # results = run_experiment(train_data, val_data, test_data, item_features, 0.001, 50)

if __name__ == "__main__":
    main()