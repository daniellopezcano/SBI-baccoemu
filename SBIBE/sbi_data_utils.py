import numpy as np
import baccoemu
import scipy as sp
import sklearn as skl

# ------------------------------------------------------------------------------------------------------ #
# ------------------------------------------ get dataset ----------------------------------------------- #
# ------------------------------------------------------------------------------------------------------ #

# def convert_params_to_baccoemu_input(theta, cosmo_params_keys, dict_fixed=dict(neutrino_mass=0.0, w0=-1.0, wa=0.0, expfactor=1.)):
def convert_params_to_baccoemu_input(theta, cosmo_params_keys, dict_fixed=dict(neutrino_mass=0.0, w0=-1.0, wa=0.0, expfactor=1.), baryonic_mode=False, dict_baryonic=dict(M_c=11.5, eta=.0, beta=0.0, M1_z0_cen=11.5, theta_out=0., theta_inn=-1., M_inn=12)):

    baccoemu_input = {}
    for ii, key in enumerate(cosmo_params_keys):
        baccoemu_input[key] = theta[:, ii]
    for ii, key in enumerate(dict_fixed):
        baccoemu_input[key] = np.repeat(dict_fixed[key], theta.shape[0])
    if baryonic_mode:
        for ii, key in enumerate(dict_baryonic):
            baccoemu_input[key] = np.repeat(dict_baryonic[key], theta.shape[0])
        
    return baccoemu_input


def compute_baccoemu_predictions(theta, cosmo_params_keys, kmin=-2.3, kmax=0.6, N_kk=100, mode='nonlinear'):
    
    assert mode in ['linear', 'nonlinear', 'baryons'], "mode must be 'linear', 'nonlinear',  or 'baryons'"
    
    baccoemu_input = convert_params_to_baccoemu_input(theta, cosmo_params_keys)
    emulator = baccoemu.Matter_powerspectrum()
    if mode == 'linear':
        kk, pk = emulator.get_linear_pk(k=np.logspace(kmin, kmax, num=N_kk), cold=False, **baccoemu_input)
    if mode == 'nonlinear':
        kk, pk = emulator.get_nonlinear_pk(k=np.logspace(kmin, kmax, num=N_kk), cold=False, **baccoemu_input)
    if mode == 'baryons':
        kk, pk = emulator.get_nonlinear_pk(k=np.logspace(kmin, kmax, num=N_kk), baryonic_boost=True, **baccoemu_input)
    
    return np.log10(pk), kk


def compute_baccoemu_predictions_batch(batch_theta, keys_cosmo_params):

    tmp_xx, tmp_kk = compute_baccoemu_predictions(
        np.reshape(batch_theta, (batch_theta.shape[0]*batch_theta.shape[1], batch_theta.shape[-1])),
        keys_cosmo_params
    )
    batch_xx = np.reshape(tmp_xx, (batch_theta.shape[0], batch_theta.shape[1], tmp_xx.shape[-1]))
    
    return batch_xx


def sample_latin_hypercube(dict_bounds, N_points=3000, seed=0):
    
    DD = len(dict_bounds)

    sample = sp.stats.qmc.LatinHypercube(d=DD, seed=seed).random(n=N_points)
    
    l_bounds = []
    u_bounds = []
    for key in dict_bounds.keys():
        l_bounds.append(dict_bounds[key][0])
        u_bounds.append(dict_bounds[key][1])
    
    theta_latin_hypercube = sp.stats.qmc.scale(sample, l_bounds, u_bounds)
    
    return theta_latin_hypercube


def get_xx(dict_bounds, theta):
    xx, kk = compute_baccoemu_predictions(theta, list(dict_bounds.keys()))
    return xx, kk

