import pickle
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import numpy as np
import torch
from data import get_data
from tools import progress, hyperparameter_tuning
from cvae import ConditionalVAE
from hsr import HeteroskedasticRegression


sns.set_theme(style='whitegrid')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(train_params, data=None, load_path=None, save_path=None):
    """
    Initialize, load, and train a Conditional VAE on the training data, returns the callable regressor / sampler
    """
    net = train_params.pop('model')(**train_params.pop('model_params')).to(device)
    if load_path is not None:
        net.load_state_dict(torch.load(load_path))
    if data is not None:
        net.trainer(data, save=save_path, **train_params)
    net.eval()
    return partial(net.sample, random=False), partial(net.sample, random=True)


def eval(model, data, metrics, sample=None, plot=True, save_preds=False, save_samples=False):
    """
    Evaluate model on the validation data, returns a dict with entries {metric: value}
    """
    results = {m: 0 for m in metrics}
    all_preds = []
    all_samples = None
    if save_samples:
        hf_s = h5py.File('tmp_samples.h5', 'w')
        all_samples = hf_s.create_dataset('pred', (0, 128, 128), maxshape=(None, 128, 128))
    with torch.no_grad():
        for batch in progress(data, text='Evaluating'):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            # Mean prediction
            y_hat, y_hat_std = model(x)

            # Re-normalize
            norm_x, norm_y = [1, 1]  # [n.to(device) for n in data.dataset.norm]
            y_hat *= norm_y
            y_hat_std *= norm_y
            x *= norm_x
            y *= norm_y

            # Sample
            y_hats = torch.stack([sample(x) * norm_y for _ in range(128)], 2).sort(dim=-1)[0]

            if save_preds:
                all_preds += [y_hat]
            if save_samples:
                all_samples.resize(all_samples.shape[0] + y_hats.shape[0], axis=0)
                all_samples[-y_hats.shape[0]:] = y_hats.cpu().numpy()

            # Compute metrics
            for m in metrics:
                if m == 'mse':
                    ths_res = ((y - y_hat) ** 2).sum(axis=0)
                elif m == 'mae':
                    ths_res = torch.abs(y - y_hat).sum(axis=0)
                elif m == 'mle':
                    ths_res = (((y - y_hat) / y_hat_std) ** 2 - torch.log(y_hat_std)).sum(axis=0)
                # elif m == 'crps':
                #     # Analytically
                #     w = ((y - mu) * prec**0.5).cpu()
                #     lk = (w * (2 * norm().cdf(w) - 1) + 2 * norm().pdf(w) - np.pi**(-0.5)) / prec.cpu()**0.5
                #     ths_res = lk.mean(axis=0)
                elif m == 'crps_ecdf':
                    n = y_hats.shape[2]
                    # E[Y - y]
                    mae = torch.abs(y[..., None] - y_hats).mean(axis=(0, -1))
                    # E[Y - Y'] = sum_i sum_j |Y_i - Y_j| / n^2
                    diff = y_hats[..., 1:] - y_hats[..., :-1]
                    count = torch.arange(1, n, device=device) * torch.arange(n - 1, 0, -1, device=device)
                    ths_res = mae - (diff * count).sum(axis=-1).mean(axis=0) / (2 * n * (n-1))
                elif m == 'std':
                    # Test sampler
                    ths_res = y_hats.std(dim=-1).mean(axis=0)
                elif m == 'cond-std':
                    ths_res = y_hat_std.mean(axis=0)
                # elif m == 'nz':
                #     ths_res = p0.mean(axis=0)
                else:
                    raise ValueError('Unknown metric')
                results[m] += ths_res
    if save_preds:
        hf = h5py.File('tmp_preds.h5', 'w')
        all_preds = torch.cat(all_preds).cpu().numpy()
        hf.create_dataset('pred', data=all_preds)
        hf.close()
    if save_samples:
        hf_s.close()
    for i, m in enumerate(metrics):
        results[m] /= len(data.dataset)
        if plot:
            plt.figure()
            res = results[m]
            plt.bar(range(len(res)), res.cpu())
            plt.semilogy()
            plt.ylabel(m)
            plt.title('%s: %.4g' % (m, res.mean()))
            plt.tight_layout()
            plt.savefig('results/tmp_metrics_%d.png' % i)
        results[m] = results[m].mean().cpu().numpy()
        print('%s: %.4g' % (m, results[m]))
    if plot:
        plt.show()
        plt.close('all')
    return results, all_preds


def train_and_eval(train_params, metrics, save_path=None):
    """
    Trains a model and evaluates on the validation data

    Input:
    ------
    training_params - [nested dict] the parameter configuration file including values for batch_size, model_params, etc
    metrics - [list] list of metrics to evaluate
    save_path - [str] file path to save trained model to
    """
    data_train, data_val = get_data(batch_size=train_params.pop('batch_size'), shuffle=True)
    model, sample = train(train_params, data_train, load_path=None, save_path=save_path)
    # eval(model, data_train, metrics)
    return eval(model, data_val, metrics, sample=sample)


def load_eval(save_dir, metrics, i=0, save_preds=False, save_samples=False, retrain=False, train_params=None):
    try:
        # load from sweep
        with open(save_dir + 'records.pickle', 'rb') as f:
            record = pickle.load(f)
            id, train_params = record[1][i], record[2][i]
            load_path = save_dir + '%d.cp' % id
    except FileNotFoundError:
        # Load individual
        load_path = save_dir
    if retrain:
        # train_params['lr'] = train_params['lr'] / 10
        # train_params['model_params']['beta'] = train_params['model_params']['beta']
        data_train, data_val = get_data(batch_size=train_params.pop('batch_size'), shuffle=True)
        model, sample = train(train_params, data_train, save_path=save_dir + 'tmp.cp')
    else:
        _, data_val = get_data(batch_size=train_params.pop('batch_size'), val_only=True)
        model, sample = train(train_params, load_path=load_path)
    _, preds = eval(model, data_val, metrics, sample=sample, save_preds=save_preds, save_samples=save_samples)
    from aziz import rs, skill
    skill(data_val.dataset.datay.cpu().numpy(), preds)
    rs(data_val.dataset.datay.cpu().numpy(), preds)


def hsr(tasks):
    """
    Test, evaluate, and fine-tune a Conditional VAE

    Input
    -----
    task - list of tasks from ['test', 'eval', 'sweep']
    """
    metrics = ['mse', 'mae', 'mle', 'crps_ecdf']
    if 'sweep' in tasks:
        sweep = {
            'model': HeteroskedasticRegression,
            'model_params': {
                'hidden_dims': {
                    'values': [256, 512, 1024, 2048]
                },
                'layers': {
                    'values': [2, 3, 4]
                }
            },
            'epochs': 12,
            'optimizer': {
                'values': ['adam', 'sgd']
            },
            'lr': {
                'max': 0.001,
                'min': 0.000001,
                'distribution': 'log_uniform'
            },
            'gamma': {
                'max': 0.1,
                'min': 0.001,
                'distribution': 'log_uniform'
            },
            'loss_type': 'mle',
            'batch_size': {
                'values': [1024, 2048, 4096, 8192, 16384],
            }
        }

        hyperparameter_tuning(
            sweep, partial(train_and_eval, metrics=metrics), 'crps_ecdf', runs=200, save_dir='models/hetreg_sweep+/')

    if 'test' in tasks:
        train_params = {
            'model': HeteroskedasticRegression,
            'model_params': {
                'hidden_dims': 1024,
                'layers': 3,
            },
            'epochs': 12,
            'optimizer': 'adam',
            'lr': 0.0001,
            'gamma': 0.01,
            'loss_type': 'mle',
            'batch_size': 4096,
        }
        # train_and_eval(train_params, metrics, save_path='models/tmp_hsr.cp')
        load_eval(train_params=train_params, metrics=metrics, save_dir='models/test_hsr.cp', save_preds=True, save_samples=True)

    if 'eval' in tasks:
        load_eval('models/hetreg_sweep/', metrics)

    '''
    mse: 0.004088
    mae: 0.02186
    mle: 6.59
    crps_ecdf: 6.016e-06
    '''


def cvae(tasks):
    """
    Test, evaluate, and fine-tune a Conditional VAE

    Input
    -----
    task - list of tasks from ['test', 'eval', 'sweep']
    """
    metrics = ['mse', 'mae', 'crps_ecdf', 'std', 'cond-std']
    if 'sweep' in tasks:
        sweep = {
            'model': ConditionalVAE,
            'model_params': {
                'latent_dims': {
                    'values': [4, 8, 16, 32]
                },
                'hidden_dims': {
                    'values': [256, 512, 1024, 2048]
                },
                'layers': {
                    'values': [2, 3, 4]
                },
                'beta': {
                    'max': 10,
                    'min': 0.01,
                    'distribution': 'log_uniform'
                },
            },
            'epochs': 5,
            'optimizer': {
                'values': ['adam', 'sgd']
            },
            'lr': {
                'max': 0.001,
                'min': 0.000001,
                'distribution': 'log_uniform'
            },
            'weight_decay': {
                'max': 0.001,
                'min': 0.00001,
                'distribution': 'log_uniform'
            },
            'loss_type': 'mse',
            'batch_size': {
                'values': [1024, 2048, 4096, 8192, 16384],
            }
        }

        hyperparameter_tuning(
            sweep, partial(train_and_eval, metrics=metrics), 'crps_ecdf', runs=200, save_dir='models/cvae_sweep+/')

    if 'test' in tasks:
        train_params = {
            'model': ConditionalVAE,
            'model_params': {
                'latent_dims': 4,
                'hidden_dims': 1024,
                'layers': 3,
                'beta': 0.5,
                'dropout': 0.05
            },
            'epochs': 10,
            'optimizer': 'adam',
            'lr': 0.00005,
            'weight_decay': 0.001,
            'loss_type': 'mse',
            'batch_size': 4096,
        }
        # train_and_eval(train_params, metrics, save_path='models/tmp_cvae.cp')
        load_eval(train_params=train_params, metrics=metrics, save_dir='models/test_cvae.cp', save_preds=True, save_samples=True)

    if 'eval' in tasks:
        load_eval('models/cvae_sweep/', metrics, save_preds=True)
        # load_eval('models/cvae_sweep/', metrics, retrain=True, save_preds=False)
        # load_eval('models/cvae_sweep2/', metrics)
        # load_eval('models/cvae_sweep3/', metrics)

    '''
    mse: 0.00413
    mae: 0.02422
    crps_ecdf: 5.194e-05
    std: 7.894e-05
    ----- 5
    mse: 0.004601
    mae: 0.02384
    crps_ecdf: 5.724e-06
    std: 3.025e-06
    cond-std: 8.94e-07
    ----- 50
    mse: 0.004084
    mae: 0.02167
    crps_ecdf: 5.098e-06
    std: 1.63e-06
    cond-std: 8.514e-07
    '''


if __name__ == '__main__':
    # cvae(tasks=['test'])
    hsr(tasks=['sweep'])
