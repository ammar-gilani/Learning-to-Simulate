from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import Config
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC

from Dataset import Dataset
from GaussianPolicy2D import GaussianPolicy2D

print('Hi!')


# Works for only two-class problem
def reformat_dataset(my_dataset):
    X = np.concatenate((my_dataset[0], my_dataset[1]))
    y = np.concatenate((np.zeros(my_dataset[0].shape[0]), np.ones(my_dataset[1].shape[0])))
    return Dataset(my_X=X, my_y=y)


def generate_K_simulation_pars(policy):
    return policy.act()


def generate_K_datasets(K, my_simulation_pars):
    datasets = []
    for _ in range(K):
        dataset = generate_single_dataset(my_simulation_pars[0], my_simulation_pars[1])
        datasets.append(dataset)
    return datasets


def generate_single_dataset(my_means, my_stds):
    dataset = []
    for i in range(Config.num_classes):
        X_tmp, _ = make_blobs(n_samples=Config.num_samples_training, n_features=2,
                              centers=my_means[0][i],
                              cluster_std=my_stds[0][i], random_state=0)
        dataset.append(X_tmp)
    return reformat_dataset(dataset)


def train_models(my_datasets):
    classifiers = []
    for dataset in my_datasets:
        classifiers.append(train_single_model(dataset))
    return classifiers


def train_single_model(my_dataset):
    classifier = SVC(kernel='rbf', gamma='auto')
    classifier.fit(X=my_dataset.X, y=my_dataset.y)
    return classifier


def calculate_accs(my_models, my_validation_set):
    accs = []
    for model in my_models:
        accs.append(calculate_single_acc(my_model=model, my_validation_set=my_validation_set))
    return accs


def calculate_single_acc(my_model, my_validation_set):
    y_pred = my_model.predict(X=my_validation_set.X)
    return accuracy_score(y_true=my_validation_set.y, y_pred=y_pred)


def generate_valid_set():
    dataset = []
    for i in range(Config.num_classes):
        X_tmp, _ = make_blobs(n_samples=Config.num_samples_training, n_features=2,
                              centers=Config.components_means_valid[i],
                              cluster_std=Config.components_stds_valid[i], random_state=0)
        dataset.append(X_tmp)
    return reformat_dataset(dataset)


def compute_advantage_estimates(my_R, my_baseline):
    return my_R - my_baseline


def update_policy_pars(my_policy, my_A, my_log_probs):
    # policy loss
    J = -my_log_probs * my_A
    # You can use this instead:
    # J = -my_log_probs * np.mean(my_A)
    # J is a vector, and I do not know if it works in this way or not.
    # You can simply use a for loop if necessary
    J.backward()
    nn.utils.clip_grad_norm_(my_policy.parameters(), 5.0)
    optim.Adam(J).step()


def learn_to_simulate(policy):
    baseline = 0
    validation_set = generate_valid_set()
    for iteration in range(Config.num_iterations):
        if not Config.silent_mode:
            print('starting iteration number ' + str(iteration) + '...')
        simulation_pars, action_log_probs = generate_K_simulation_pars(policy=policy)
        datasets = generate_K_datasets(Config.K, my_simulation_pars=simulation_pars)
        models = train_models(datasets)
        # reward
        R = calculate_accs(models, validation_set)
        # advantage estimate
        A = compute_advantage_estimates(R, baseline)
        update_policy_pars(my_policy=policy, my_A=A, my_log_probs=action_log_probs)


# X, y = generate_single_dataset([[[-5, -5]], [[5, 5]]], [[1], [5]])
# datasets = generate_K_datasets(
#     [[[[[-5, -5], [-6, -6]], [[5, 5], [6, 6]]], [[1, 1], [5, 5]]], [[[[-5, -5]], [[5, 5]]], [[1], [5]]]])
# print(calculate_accs(train_models(datasets), generate_valid_set()))

learn_to_simulate(GaussianPolicy2D(2, 2))
