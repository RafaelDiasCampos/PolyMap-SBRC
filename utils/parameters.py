# Default parameters
random_state = 232323
classification_results_filename = "results/classification_results.json"
attack_results_filename = "results/attack_results.json"

# Number of dataset copies for training
n_copies = 7

# Detection Network parameters
n_hidden_layers = 3
n_neurons = 32
batch_size = 2048
learning_rate_fnn = 0.004
learning_rate_snn = 0.008
dropout_prob_fnn = 0.3
dropout_prob_snn = 0.1
training_epochs = 1000

# Parameters for the attacks
n_trials = {
    "idsgan": 5,
    "gen_aal": 5,
    "polytope": 1
}
attack_train_size = 0.6

# Parameters for the IDSGAN attack
noise_dim = 7
n_hidden_layers_generator = 3
n_neurons_generator = 128
learning_rate_generator = 4e-4
steps_generator = 1
n_hidden_layers_discriminator = 2
n_neurons_discriminator = 256
learning_rate_discriminator = 4e-4
steps_discriminator = 10
weight_clip = 0.1
idsgan_epochs = 100

# Parameters for the GenAAL attack
latent_dim = 128
lambda_kl = 1e-3
vae_hidden = [2048, 1024, 512, 256]
sids_hidden = [64, 32]
lr_vae = 1e-4
lr_sids = 5e-3
lambda_label = 3.0
lambda_l2    = 0.3
lambda_recon = 0.1
gen_lr = 2e-4
sid_lr = 5e-3
label_query = 15
max_iterations = 3
candidate_pool_k = 10
nquery = 10
pretrain_epochs = 500
sids_epochs = 300
gen_epochs = 500

# Parameters for the Polytope attack
n_rays = 50
step_size = 0.01

# Parameters for plotting graphs
dataset_names = {
    "ton_iot": "TON_IoT",
    "bot_iot": "Bot-IoT",
    "nsl_kdd": "NSL-KDD",
    "ctu_13": "CTU-13"
}

attack_types = {
    'idsgan': 'IDSGAN',
    'genaal': 'Gen-AAL',
    'polytope': 'PolyMap',
    'polytope_move_inside_0.001': 'PolyMap test',
}

marker_map = {
    'idsgan': 'o',
    'genaal': 's',
    'polytope': '^',
    'polytope_move_inside_0.001': 'D'
}

color_map = {
    'idsgan': 'blue',
    'genaal': 'green',
    'polytope': 'red',
    'polytope_move_inside_0.001': 'orange'
}