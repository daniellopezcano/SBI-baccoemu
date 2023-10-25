import os
import numpy as np
import torch
import pickle
from sbi.inference import SNPE
from sbi import utils

 
def get_prior(dict_bounds):

    lower_bound = np.vstack(tuple(dict_bounds[key] for key in dict_bounds))[:,0]
    upper_bound = np.vstack(tuple(dict_bounds[key] for key in dict_bounds))[:,1]

    lower_bound, upper_bound = (
        torch.from_numpy(lower_bound.astype('float32')), 
        torch.from_numpy(upper_bound.astype('float32'))
    )
    prior = utils.BoxUniform(lower_bound, upper_bound)
    
    return prior
    
    
def train_model(theta_train, xx_train, prior, num_hidden_features=64, num_transforms=8, num_blocks=4, training_batch_size=16, learning_rate=0.00031, validation_fraction=0.2):
    
    torch.manual_seed(0)
    
    density_estimator_build_fun = utils.get_nn_models.posterior_nn(
        model='maf',
        hidden_features=num_hidden_features,
        num_transforms=num_transforms,
        num_blocks=num_blocks
    )
    
    inference = SNPE(
        prior=prior,
        density_estimator=density_estimator_build_fun
    )
    
    inference.append_simulations(
        torch.from_numpy(theta_train.astype('float32')), 
        torch.from_numpy(xx_train.astype('float32'))
    )
    
    density_estimator = inference.train(
        training_batch_size=training_batch_size,
        validation_fraction=validation_fraction,
        learning_rate=learning_rate,
        show_train_summary=True
    )
    
    posterior = inference.build_posterior(
        density_estimator
    )
    
    return inference, posterior


def sample_posteriors_theta_test(posterior, xx_test, dict_bounds, N_samples=1000):

    inferred_theta = np.zeros((xx_test.shape[0],) + (N_samples,) + (len(dict_bounds),))
    for ii in range(xx_test.shape[0]):
        if ii%20 == 0: print(ii)
        inferred_theta[ii] = posterior.sample(
            (N_samples,),
            x=torch.from_numpy(xx_test[ii].astype('float32'))
        ).detach().numpy()
    
    return inferred_theta
    
    
def compute_ranks(theta, inferred_theta):
    
    tmp_extended = np.concatenate((theta[:,np.newaxis], inferred_theta), axis=1)
    ranks = np.argsort(tmp_extended, axis=1)[:, 0, :]

    return ranks
