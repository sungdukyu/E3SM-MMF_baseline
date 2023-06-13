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
    from ptflops import get_model_complexity_info
    from fvcore.nn import FlopCountAnalysis
    flops, _ = get_model_complexity_info(net, (4096, 124))
    print(flops / 4096 / 1024 / 1024)
    flops = FlopCountAnalysis(net, torch.rand(4096, 124).to(device))
    print(flops.total() / 4096 / 1024 / 1024)
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
        all_samples = hf_s.create_dataset('pred', (0, 128, 32), maxshape=(None, 128, 32))
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
            y_hats = torch.stack([sample(x) * norm_y for _ in range(32)], 2).sort(dim=-1)[0]

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
        # plt.show()
        plt.close('all')
    return results, [all_preds, all_samples]


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
    except FileNotFoundError as e:
        # Load individual
        load_path = save_dir
    if retrain:
        data_train, data_val = get_data(batch_size=train_params.pop('batch_size'), shuffle=True)
        model, sample = train(train_params, data_train, save_path=save_dir + 'tmp.cp')
    else:
        _, data_val = get_data(batch_size=train_params.pop('batch_size'), val_only=True)
        model, sample = train(train_params, load_path=load_path)
    _, [preds, samples] = eval(model, data_val, metrics, sample=sample, save_preds=save_preds, save_samples=save_samples)

    # todo: remove
    # from aziz import rs, skill, crps
    # skill(data_val.dataset.datay.cpu().numpy(), preds)
    # rs(data_val.dataset.datay.cpu().numpy(), preds)
    # crps(data_val.dataset.datay.cpu().numpy(), samples)


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
            sweep, partial(train_and_eval, metrics=metrics), 'crps_ecdf', runs=200, save_dir='models/hsr_sweep+/')

    if 'test' in tasks:
        train_params = {
            'model': HeteroskedasticRegression,
            'model_params': {
                'hidden_dims': 1024,
                'layers': 4,
            },
            'epochs': 12,
            'optimizer': 'adam',
            'lr': 0.000377,
            'gamma': 0.00796,
            'loss_type': 'mle',
            'batch_size': 4096,
        }
        # train_and_eval(train_params, metrics, save_path='models/tmp_hsr.cp')
        load_eval(train_params=train_params, metrics=metrics, save_dir='models/test_hsr.cp', save_preds=True, save_samples=True)

    if 'eval' in tasks:
        load_eval('models/hsr_sweep+/', metrics, save_preds=False, save_samples=False)


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
        load_eval('models/cvae_sweep/', metrics, save_preds=False, save_samples=False)


if __name__ == '__main__':
    # Train manually
    # cvae(tasks=['test'])
    # hsr(tasks=['test'])

    # Tune hyperparamters
    # cvae(tasks=['sweep'])
    # hsr(tasks=['sweep'])

    # Evaluate best model
    # cvae(tasks=['eval'])
    hsr(tasks=['eval'])
