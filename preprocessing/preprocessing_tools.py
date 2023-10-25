import os
import scipy as sp
import numpy as np
    
def save_theta(path_save_data, theta, data_name='theta.npy'):
    if not os.path.exists(path_save_data):
        os.makedirs(path_save_data)
    with open(os.path.join(path_save_data, data_name), 'wb') as ff:
        np.save(ff, theta)
    return


def save_model(path_save_data, model_name, xx, data_name='Pk.npy'):
    tmp_save_path = os.path.join(path_save_data, model_name)
    if not os.path.exists(tmp_save_path):
        os.makedirs(tmp_save_path)
    with open(os.path.join(tmp_save_path, data_name), 'wb') as ff:
        np.save(ff, xx)
    return
    
    
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


def generate_ordered_meshlist(separate_arrays):
    original_meshgrid = np.array(np.meshgrid(*separate_arrays, indexing='ij'))
    tuple_for_correct_transposition = tuple(np.arange(1, len(separate_arrays)+1)) + (0,)
    transposed_meshgrid = np.transpose(original_meshgrid, tuple_for_correct_transposition)
    tuple_for_reshape = (-1,) + (len(separate_arrays),)
    meshlist = np.reshape(transposed_meshgrid, tuple_for_reshape)
    return meshlist


def get_Pk_vary_baryon_param(theta, min_value=9, max_value=15, NN=4, baryon_key='M_c'):
    aug_params_names = [baryon_key]
    aug_params = generate_ordered_meshlist((np.linspace(min_value, max_value, NN), np.linspace(0., 0., 1)))
    xx = []
    for ii in range(aug_params.shape[0]):
        tmp_params = aug_params[ii]
        baccoemu_input = parse_thetas_params_baccoemu(
            theta,
            np.repeat(tmp_params[np.newaxis], theta.shape[0], axis=0),
            aug_params_names
        )
        xx.append(bacco_emulator(baccoemu_input))
    xx = np.array(xx)
    return xx


def parse_thetas_params_baccoemu(thetas, params, model_specific_baryonic_keys=None):
        
    cosmo_params_keys = ['omega_cold', 'omega_baryon', 'hubble', 'ns', 'sigma8_cold']
    baccoemu_input = {}
    for ii, key in enumerate(cosmo_params_keys):
        baccoemu_input[key] = thetas[:, ii]

    dict_fixed=dict(neutrino_mass=0.0, w0=-1.0, wa=0.0, expfactor=1.)
    for ii, key in enumerate(dict_fixed):
        baccoemu_input[key] = np.repeat(dict_fixed[key], thetas.shape[0])

    dict_baryonic=dict(M_c=11.5, eta=.0, beta=0.0, M1_z0_cen=11.5, theta_out=0., theta_inn=-1., M_inn=12)
    for ii, key in enumerate(dict_baryonic):
        baccoemu_input[key] = np.repeat(dict_baryonic[key], thetas.shape[0])
    
    if model_specific_baryonic_keys!=None:
        assert thetas.shape[0] == params.shape[0], "thetas and params must have same batch size"
        for ii, key in enumerate(model_specific_baryonic_keys):
            baccoemu_input[key] = params[:, ii]
        
    return baccoemu_input


def bacco_emulator(baccoemu_input, kmin=-2.3, kmax=0.6, N_kk=100, mode='baryons', return_kk=False):
    import baccoemu
    
    emulator = baccoemu.Matter_powerspectrum()
    
    kk = np.logspace(kmin, kmax, num=N_kk)
    if mode == 'linear':
        kk, pk = emulator.get_linear_pk(k=kk, cold=False, baryonic_boost=False, **baccoemu_input)
    if mode == 'nonlinear':
        kk, pk = emulator.get_nonlinear_pk(k=kk, cold=False, baryonic_boost=False, **baccoemu_input)
    if mode == 'baryons':
        kk, pk = emulator.get_nonlinear_pk(k=kk, cold=False, baryonic_boost=True, **baccoemu_input)
    
    if return_kk:
        xx = np.log10(pk)
        return kk, xx
    else:
        xx = np.log10(pk)
        return xx
    
def model_linear(theta):
    model_name = 'model_linear'
    xx = []
    baccoemu_input = parse_thetas_params_baccoemu(theta, None)
    xx.append(bacco_emulator(baccoemu_input, mode='linear'))
    xx = np.array(xx)
    return model_name, xx
    

def model_nonlinear(theta):
    model_name = 'model_nonlinear'
    xx = []
    baccoemu_input = parse_thetas_params_baccoemu(theta, None)
    xx.append(bacco_emulator(baccoemu_input, mode='nonlinear'))
    xx = np.array(xx)
    return model_name, xx
    

def aug_model1(theta):
    model_name = 'model1'
    aug_params_names = ['M_c', 'eta']
    aug_params = generate_ordered_meshlist((np.linspace(9, 15, 4), np.linspace(-0.68, 0.68, 3)))
    xx = []
    for ii in range(aug_params.shape[0]):
        tmp_params = aug_params[ii]
        baccoemu_input = parse_thetas_params_baccoemu(
            theta,
            np.repeat(tmp_params[np.newaxis], theta.shape[0], axis=0),
            aug_params_names
        )
        xx.append(bacco_emulator(baccoemu_input))
    xx = np.array(xx)
    return model_name, aug_params, xx


def aug_model2(theta):
    model_name = 'model2'
    aug_params_names = ['M_c', 'eta', 'beta']
    aug_params = generate_ordered_meshlist((np.linspace(9, 15, 2), np.linspace(-0.68, 0.68, 3), np.linspace(-1., 0.6, 2)))
    xx = []
    for ii in range(aug_params.shape[0]):
        tmp_params = aug_params[ii]
        baccoemu_input = parse_thetas_params_baccoemu(
            theta,
            np.repeat(tmp_params[np.newaxis], theta.shape[0], axis=0),
            aug_params_names
        )
        xx.append(bacco_emulator(baccoemu_input))
    xx = np.array(xx)
    return model_name, aug_params, xx


def aug_model3(theta):
    model_name = 'model3'
    aug_params_names = ['M1_z0_cen', 'theta_out']
    aug_params = generate_ordered_meshlist((np.linspace(10, 13, 2), np.linspace(0., 0.45, 2)))
    xx = []
    for ii in range(aug_params.shape[0]):
        tmp_params = aug_params[ii]
        baccoemu_input = parse_thetas_params_baccoemu(
            theta,
            np.repeat(tmp_params[np.newaxis], theta.shape[0], axis=0),
            aug_params_names
        )
        xx.append(bacco_emulator(baccoemu_input))
    xx = np.array(xx)
    return model_name, aug_params, xx
    
    
# ------------------------------ preprocessing tools Akhmetzhanova ------------------------------ #

def generate_Pk_Akhmetzhanova(kk, AA, BB, DD, k_pivot=0.5, k_F=7*10**-3):
    
    try:
        NN_cosmo = len(AA)
    except:
        NN_cosmo=1
        AA = np.array(AA)
        BB = np.array(BB)
        DD = np.array(DD)
        
    NN_k=len(kk)
    
    CC = AA * (k_pivot**(BB-DD))

    kk_tile = np.tile(kk[np.newaxis], (NN_cosmo,1))
    AA_tile = np.tile(AA[np.newaxis], (NN_k,1)).T
    BB_tile = np.tile(BB[np.newaxis], (NN_k,1)).T
    DD_tile = np.tile(DD[np.newaxis], (NN_k,1)).T
    CC_tile = np.tile(CC[np.newaxis], (NN_k,1)).T

    Pk1 = AA_tile*kk_tile**BB_tile
    Pk2 = CC_tile*kk_tile**DD_tile

    index_cut = np.argmax(kk > k_pivot)

    Pk = np.zeros((NN_cosmo, NN_k))
    Pk[:, :index_cut] = Pk1[:, :index_cut]
    Pk[:, index_cut:] = Pk2[:, index_cut:]
    
    N_k = (4 * np.pi * kk**2 * k_F) / k_F**3
    var_k = np.sqrt(2/N_k) * Pk
    Pk_obs = np.random.normal(loc=Pk, scale=var_k)
    
    return Pk, Pk_obs, AA, BB, DD


def wrapper_generate_Pk_Akhmetzhanova(
        NN_cosmo=1000, NN_aug=10, NN_k=140, k_F=7*10**-3, kmin=3, kmax=142, k_pivot=0.5,
        A_min=0.1, A_max=1., B_min=-1, B_max=0., D_min=-0.5, D_max=0.5, seed=0
    ):
    
    kk = np.linspace(kmin, kmax, NN_k)*k_F
    
    np.random.seed(seed=seed)
    
    AA = np.random.uniform(A_min, A_max, NN_cosmo)
    BB = np.random.uniform(B_min, B_max, NN_cosmo)
    DD = np.random.uniform(D_min, D_max, (NN_cosmo, NN_aug))
    
    Pk = np.zeros((NN_aug, NN_cosmo , NN_k))
    Pk_obs = np.zeros((NN_aug, NN_cosmo, NN_k))
    for ii in range(NN_aug):
        Pk[ii], Pk_obs[ii], _, _, _ = generate_Pk_Akhmetzhanova(kk, AA, BB, DD[:, ii], k_pivot=k_pivot, k_F=k_F)
    
    return kk, Pk, Pk_obs, AA, BB, DD


def get_delta_LN_form_Pk(kk, Pk, ngrid=100, BoxSize=1000):
    
    import bacco
    from bacco.lss_scaler import lss
    
    delta_LN = np.zeros((Pk.shape[0], ngrid,ngrid,ngrid))
    for ii in range(Pk.shape[0]):
        print(ii)
        delta_G = (
            lss.lpt_field(
                kk,
                Pk[ii],
                1, # Di
                0, # Di2
                0, # Di3a
                0, # Di3b
                1, # growthrate
                1, # LPT_order
                1e4, # damping_scale
                0., #kmin
                1e10, #kmax
                False, # order_by_order
                ngrid, # config['scaling']['disp_ngrid']
                BoxSize, # self.header['BoxSize']
                0, # self.header['Seed']
                1., # 1. / self.Cosmology.expfactor - 1,
                1, # self.header["FixedInitialAmplitude"]
                0.0, # self.header["InitialPhase"]
                0, # self.phase_type
                1, # return_density
                0, # redshift_space
                0, # sphere_mode
                4, # config['number_of_threads']
                False # config['verbose']
            )
        )[0] * ngrid**3

        var_G = np.var(delta_G)
        delta_LN[ii] = np.exp(delta_G - var_G**2/2)-1

    return delta_LN


