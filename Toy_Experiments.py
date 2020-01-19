from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
import Config
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC
import numpy as np
from MyDataset import MyDataset
from GaussianPolicy2D import GaussianPolicy2D


def sample_gaussian(my_n, my_n_features, my_means, my_std):
    X = np.zeros([my_n, my_n_features])
    for i in range(my_n_features):
        tmp = np.random.normal(loc=my_means[i], scale=my_std[i], size=my_n)
        X[:, i] = tmp
    return X


def sample_gaussian_mixture(my_n, my_n_features, my_means, my_std):
    X = sample_gaussian(my_n=my_n, my_n_features=my_n_features, my_means=my_means[0], my_std=my_std[0])
    num_components = len(my_means)
    for i in range(num_components - 1):
        X_tmp = sample_gaussian(my_n=my_n, my_n_features=my_n_features, my_means=my_means[i + 1], my_std=my_std[i + 1])
        X = np.concatenate((X, X_tmp))
    return X


def generate_dataset(my_means, my_std):
    dataset = []
    for i in range(Config.num_classes):
        X_tmp = sample_gaussian_mixture(my_n=Config.num_samples_training, my_n_features=2, my_means=my_means[i],
                                        my_std=my_std[i])
        dataset.append(X_tmp)
    return reformat_dataset(dataset)


# Works for only two-class problems
def reformat_dataset(my_dataset):
    X = np.concatenate((my_dataset[0], my_dataset[1]))
    y = np.concatenate(
        (np.zeros(shape=my_dataset[0].shape[0], dtype=int), np.ones(shape=my_dataset[1].shape[0], dtype=int)))
    return MyDataset(X, y)


def train_model(my_dataset):
    classifier = SVC(kernel='rbf', gamma='auto')
    classifier.fit(X=my_dataset.X, y=my_dataset.y)
    return classifier


def calculate_acc(my_model, my_validation_set):
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


def calculate_advantage_estimates(my_R, my_baseline):
    return my_R - my_baseline


def generate_single_simulation_pars(my_policy):
    return my_policy.act()


def learn_to_simulate():
    policy = GaussianPolicy2D(2, 2)
    validation_set = generate_valid_set()
    optimizer = optim.Adam(params=policy.parameters())
    baseline = 0
    for i in range(Config.num_iterations):
        if np.mod(i + 1, 100) == 0:
            print('iteration #', i + 1, ', accuracy: ', np.mean(rewards))
        rewards = []
        for j in range(Config.K):
            optimizer.zero_grad()
            simulation_parameters, log_probs = generate_single_simulation_pars(my_policy=policy)
            dataset = generate_dataset(my_means=simulation_parameters[0], my_std=simulation_parameters[1])
            model = train_model(my_dataset=dataset)
            R = calculate_acc(my_model=model, my_validation_set=validation_set)
            rewards.append(R)
            A = calculate_advantage_estimates(my_R=R, my_baseline=baseline)
            J = -log_probs * A
            J.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()
        baseline = baseline * (1 - 0.8) + np.mean(rewards) * 0.8


learn_to_simulate()
