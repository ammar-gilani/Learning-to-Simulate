import Config

print('Hi!')


# TODO
def generate_K_simulation_pars():
    return None


# TODO
def generate_K_datasets(my_simulation_pars):
    return None


# TODO
def initialize_models():
    pass


# TODO
def train_models(my_models, my_datasets):
    pass


# TODO
def compute_accs(my_models, my_validation_set):
    pass


# TODO
def generate_valid_set():
    pass


# TODO
def compute_advantage_estimates(my_R):
    pass


def update_policy_pars(A):
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
        w = update_policy_pars()
