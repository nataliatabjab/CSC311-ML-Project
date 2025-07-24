# TODO: complete this file.
from utils import load_train_csv, load_valid_csv, load_public_test_csv, evaluate
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

# Helper function to convert data into a format that can be used for LogisticRegression
def feature_matrix(data, num_users, num_questions):
    matrix = []
    for user_id, question_id in zip(data["user_id"], data["question_id"]):
        # create one-hot vector: set encoding to have the user_id and question_id be 1
        feature_vector = np.zeros(num_users + num_questions)
        feature_vector[user_id] = 1
        feature_vector[num_users + question_id] = 1
        matrix.append(feature_vector)
    return np.array(matrix)

def ensemble(train_data, validation_data, test_data):
    num_users = len(set(train_data["user_id"]))
    num_questions = len(set(train_data["question_id"]))

    x_train = feature_matrix(train_data, num_users, num_questions)
    x_test = feature_matrix(test_data, num_users, num_questions)
    x_validation = feature_matrix(validation_data, num_users, num_questions)
    y_train = np.array(train_data["is_correct"])

    models = []
    # Uses logistic regression for all 3 base models
    for i in range(3):
        x_sample, y_sample = resample(x_train, y_train)
        model = LogisticRegression(max_iter=50)
        model.fit(x_sample, y_sample)
        models.append(model)

    validation_predictions = np.mean([model.predict_proba(x_validation)[:, 1] for model in models], axis=0)
    test_predictions = np.mean([model.predict_proba(x_test)[:, 1] for model in models], axis=0)

    validation_accuracy = evaluate(validation_data, [1 if prediction >= 0.5 else 0 for prediction in validation_predictions])
    test_accuracy = evaluate(test_data,  [1 if prediction >= 0.5 else 0 for prediction in test_predictions])

    return validation_accuracy, test_accuracy

def main():
    train_data = load_train_csv("./data")
    validation_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    
    validation_accuracy, test_accuracy = ensemble(train_data, validation_data, test_data)
    print("Validation Accuracy: ", validation_accuracy)
    print("Test Accuracy: ", test_accuracy)

if __name__ == "__main__":
    main()