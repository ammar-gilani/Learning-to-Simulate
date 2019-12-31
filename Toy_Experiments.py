import Config
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

print('Hi!')


# TODO
def generate_K_simulation_pars():
    return None


# TODO
def generate_K_datasets(my_simulation_pars):
    return None


def generate_single_dataset(my_means, my_stds):
    X = []
    for i in range(Config.num_classes):
        X_tmp, _ = make_blobs(n_samples=Config.num_samples_training, n_features=2,
                              centers=my_means[i],
                              cluster_std=my_stds[i], random_state=0)
        X.append(X_tmp)
        if not Config.silent_mode:
            plt.scatter(X_tmp[:, 0], X_tmp[:, 1], s=40, cmap='viridis', c=Config.colors[i])
    if not Config.silent_mode:
        plt.show()

    return X


# TODO
def initialize_models():
    pass


# TODO
def train_models(my_models, my_datasets):
    pass


# TODO
def compute_accs(my_models, my_validation_set):
    pass


def generate_valid_set():
    X = []
    for i in range(Config.num_classes):
        X_tmp, _ = make_blobs(n_samples=Config.num_samples_training, n_features=2,
                              centers=Config.components_means_valid[i],
                              cluster_std=Config.components_stds_valid[i], random_state=0)
        X.append(X_tmp)
        if not Config.silent_mode:
            plt.scatter(X_tmp[:, 0], X_tmp[:, 1], s=40, cmap='viridis', c=Config.colors[i])
    if not Config.silent_mode:
        plt.show()
    return X


# TODO
def compute_advantage_estimates(my_R):
    pass


def update_policy_pars(my_A):
    pass


# TODO
def learn_to_simulate():
    models = initialize_models()
    validation_set = generate_valid_set()
    for iteration in range(Config.num_iterations):
        if not Config.silent_mode:
            print('starting iteration number ' + str(iteration) + '...')
        simulation_pars = generate_K_simulation_pars()
        datasets = generate_K_datasets(simulation_pars)
        models = train_models(models, datasets)
        R = compute_accs(models, validation_set)
        A = compute_advantage_estimates(R)
        w = update_policy_pars(A)
