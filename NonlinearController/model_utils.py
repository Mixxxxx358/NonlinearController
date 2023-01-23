import numpy as np
from casadi import *

def CasADi_Hn(ss_enc, cas_x):
    n_hidden_layers = 2#ss_enc.h_n_hidden_layers

    params = {}
    for name, param in ss_enc.hn.named_parameters():
        params[name] = param.detach().numpy()
    params_list = list(params.values())

    temp_nn = cas_x
    for i in range(n_hidden_layers):
        W_NL = params_list[2+i*2]
        b_NL = params_list[3+i*2]
        temp_nn = mtimes(W_NL, temp_nn)+b_NL
        temp_nn = tanh(temp_nn)
    W_NL = params_list[2+n_hidden_layers*2]
    b_NL = params_list[3+n_hidden_layers*2]
    nn_NL = mtimes(W_NL, temp_nn)+b_NL

    W_Lin = params_list[0]
    b_Lin = params_list[1]
    nn_Lin = mtimes(W_Lin,cas_x) + b_Lin

    return nn_NL + nn_Lin

def CasADi_Fn(ss_enc, cas_x, cas_u):
    n_hidden_layers = 2#ss_enc.f_n_hidden_layers
    nu = ss_enc.nu if ss_enc.nu is not None else 1

    params = {}
    for name, param in ss_enc.fn.named_parameters():
        params[name] = param.detach().numpy()
    params_list = list(params.values())
    
    cas_xu = vertcat(cas_x,cas_u)

    temp_nn = cas_xu
    for i in range(n_hidden_layers):
        W_NL = params_list[2+i*2]
        b_NL = params_list[3+i*2]
        temp_nn = mtimes(W_NL, temp_nn)+b_NL
        temp_nn = tanh(temp_nn)
    W_NL = params_list[2+n_hidden_layers*2]
    b_NL = params_list[3+n_hidden_layers*2]
    nn_NL = mtimes(W_NL, temp_nn)+b_NL

    W_Lin = params_list[0]
    b_Lin = params_list[1]
    nn_Lin = mtimes(W_Lin,cas_xu) + b_Lin

    return nn_NL + nn_Lin
