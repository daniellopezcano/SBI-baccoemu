import sys, os
import numpy as np
import sklearn as skl
import pickle

import SBIBE as sbibe

import wandb

import yaml
from pathlib import Path

def wandb_single_train(
        num_hidden_features,
        num_transforms,
        num_blocks,
        training_batch_size,
        learning_rate,
        savepath,
        model_name
    ):
    
    # ------------------ extract train ------------------ #
    
    dict_bounds=dict(
      omega_cold=[0.23, 0.4],
      omega_baryon=[0.04, 0.06],
      hubble=[0.6, 0.8],
      ns=[0.92, 1.01],
      sigma8_cold=[0.73, 0.9]
    )        
    
    theta_train = sbibe.sbi_data_utils.sample_latin_hypercube(dict_bounds)
    xx_train, kk = sbibe.sbi_data_utils.get_xx(dict_bounds, theta_train)
    
    # ------------------ train NF ------------------ #
    
    # scaler = skl.preprocessing.StandardScaler()
    # norm_xx_train = scaler.fit_transform(xx_train)

    inference, posterior =sbibe.sbi_utils.train_model(
        theta_train,
        xx_train,
        prior=sbibe.sbi_utils.get_prior(dict_bounds),
        num_hidden_features = num_hidden_features,
        num_transforms =      num_transforms,
        num_blocks =          num_blocks,
        training_batch_size = training_batch_size,
        learning_rate =       learning_rate
    )
    
    # ------------------ save posteriors and inference ------------------ #
    
    if model_name is not None:
        savepath = os.path.join(savepath, model_name)
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        with open(os.path.join(savepath, "posterior.pkl"), "wb") as handle:
            pickle.dump(posterior, handle)
        with open(os.path.join(savepath, "inference.pkl"), "wb") as handle:
            pickle.dump(inference, handle)
    
    loss = inference.summary['best_validation_log_prob'][0]
        
    return loss
    
    
def train(config=None):
    
    with wandb.init(config=config) as run:
        
        config = wandb.config
        
        loss = wandb_single_train(
            **config,
            model_name=run.name
        )
        
        if "savepath" in config.keys():
            with open(os.path.join(config["savepath"], 'register_sweeps.txt'), 'a') as ff:
                ff.write(run.name + ' ' + str(loss) +'\n')
        
        wandb.run.summary["loss"] = loss # The key of this deictionary must correspond to the key specified in the wandb config file
        
        
def wandb_sweep(path_to_wandb_config, wandb_project_name, N_samples_hyperparmeters=5):
    
    sweep_config = yaml.safe_load(Path(path_to_wandb_config).read_text())

    wandb.login()
    
    wandb.agent(
        wandb.sweep(
            sweep_config,
            project=wandb_project_name
        ),
        train,
        count=N_samples_hyperparmeters
    )
    
# -------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------- Loading Utils ------------------------------------------------ #
# -------------------------------------------------------------------------------------------------------------- #

def load_wandb_sweep_register(path_wandb_sweep):

    tmp_file = os.path.join(path_wandb_sweep, 'register_sweeps.txt')
    sweep_names = [x.split(' ')[0] for x in open(tmp_file).readlines()]
    losses = np.loadtxt(tmp_file, delimiter=" ",usecols=1)

    indexes_sorted = np.argsort(losses)[::-1]
    losses = losses[indexes_sorted]
    sweep_names = [sweep_names[ii] for ii in indexes_sorted]
    
    return sweep_names, losses
    

def load_posterior(savepath, name_posterior_file="posterior.pkl"):
    with open(os.path.join(savepath, name_posterior_file), "rb") as handle:
        return pickle.load(handle)


def load_posteriors(path_sweep, sweep_names):
    posteriors = []
    for ii in range(len(sweep_names)):
        savepath = os.path.join(path_sweep, sweep_names[ii])
        posteriors.append(load_posterior(savepath))
    return posteriors
