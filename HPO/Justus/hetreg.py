import pickle
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from data import get_data
from tools import progress, hyperparameter_tuning
from scipy.stats import norm

sns.set_theme(style='whitegrid')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MLP(torch.nn.Module):
    """
    MLP Estimator for mean and log precision
    """
    def __init__(self, in_dims, out_dims, hidden_dims=512, layers=1):
        super().__init__()
        self.linears = []
        for i in range(layers):
            self.linears += [torch.nn.Linear(in_dims if i == 0 else hidden_dims, hidden_dims)]
            self.add_module('linear%d' % i, self.linears[-1])
        self.final_linear = torch.nn.Linear(hidden_dims, out_dims)

    def forward(self, x):
        torch.flatten(x, start_dim=1)
        for linear in self.linears:
            x = torch.nn.functional.relu(linear(x))
        x = self.final_linear(x)
        return x


class Regressor(torch.nn.Module):
    def __init__(self, in_dims=124, out_dims=128, hidden_dims=512, layers=1):
        """
        Heteroskedastic Regression model, computing MLE estimates of mean and precision via regularized MLPs

        Inputs:
        -------
        in_dims - [int] size of x
        out_dims - [int] size of y
        hidden_dims - [int] size of hidden layers
        layers - [int] number of layers, including hidden layer
        """
        super().__init__()
        self.mean = MLP(in_dims, out_dims, hidden_dims, layers=layers)
        self.logprec = MLP(in_dims, out_dims, hidden_dims, layers=layers)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)

    def forward(self, x):
        torch.flatten(x, start_dim=1)
        return self.mean(x), self.logprec(x)

    def sample(self, x):
        """
        Sample from learned data model

        Inputs:
        -------
        x - [BxN array] label
        """
        mu, logprec = self.forward(x)
        y = mu + self.N.sample(mu.shape) / torch.exp(logprec)
        return y

    def trainer(self, data, epochs=20, save="models/vae.cp", plot=True, loss_type='mle',
                optimizer='adam', lr=0.0001, gamma=0.01, rho=None):
        """
        Train the Heteroskedastic Regression model

        Inputs:
        -------
        data - [DataLoader] - training data
        epochs - [int] number of epochs
        loss_type - [str] type of loss
        optimizer - [str] type of optimizer
        lr - [float] learning rate
        gamma - [float] trade-off between regularization and likelihood maximization
        rho - [float] trade-off between mean regularization and precision regularization
        save - [str] file path to save trained model to after training (and after every 20 minutes)
        plot - [boolean] if plots of loss curves and samples should be produced
        """
        # Training parameters
        # Regularization, reduce to a line search
        gamma = gamma
        rho = rho if rho is not None else 1 - gamma
        # L2 weight decay
        alpha = (1 - rho) / rho * gamma
        beta = (1 - rho) / rho * (1 - gamma)

        pms = [{'params': self.mean.parameters(), 'lr': lr, 'weight_decay': alpha},
               {'params': self.logprec.parameters(), 'lr': lr, 'weight_decay': beta}]
        if optimizer == 'adam':
            opt = torch.optim.Adam(pms)
        elif optimizer == 'sgd':
            opt = torch.optim.SGD(pms)
        else:
            raise ValueError('Unknown optimizer')

        # Train and checkpoint every 20 minutes
        losses = []
        for epoch, batch in progress(range(epochs), inner=data, text='Training',
                                     timed=[(1200, lambda: torch.save(self.state_dict(), save))]):
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            opt.zero_grad()
            mu, logprec = self(x)
            prec = torch.exp(logprec)
            if loss_type == 'mle':
                if epoch <= epochs // 2:
                    # first only mean, via MSE
                    loss = ((y - mu) ** 2).mean()
                else:
                    # non-iid gaussians -> maximum likelihood
                    loss = (prec * (y - mu) ** 2 - logprec).mean()
            else:
                raise ValueError('Unknown loss')

            torch.clip(loss, max=1e5).backward()
            losses += [loss.item()]
            opt.step()
        print('Last-epoch loss: %.2f' % sum(losses[-len(data):-1]))
        print('Finished Training')

        if plot:
            plt.plot(np.array(losses)[:-1])
            plt.savefig('results/tmp_loss.png')
            plt.figure()
            plt.plot((y[0:500] - mu[0:500]).detach().cpu().numpy().T, c="C0", alpha=1/255)
            plt.plot(- np.sqrt(1 / prec).detach().cpu().numpy().T, c="C1", alpha=1/255)
            plt.plot(np.sqrt(1 / prec).detach().cpu().numpy().T, c="C1", alpha=1/255)
            plt.tight_layout()
            plt.savefig('results/tmp_last_batch.png')
            # plt.show()
            plt.close('all')


def train(train_params, data=None, load_path=None, save_path=None):
    """
    Initialize, load, and train a Conditional VAE on the training data, returns the callable regressor / sampler
    """
    vae = Regressor(**train_params.pop('model_params')).to(device)
    if load_path is not None:
        vae.load_state_dict(torch.load(load_path))
    if data is not None:
        vae.trainer(data, save=save_path, **train_params)
    vae.eval()
    return vae.forward, vae.sample


def eval(model, data, metrics, sample=None, plot=True):
    """
    Evaluate model on the validation data, returns a dict with entries {metric: value}
    """
    results = {m: 0 for m in metrics}
    with torch.no_grad():
        for batch in progress(data, text='Evaluating'):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            mu, logprec = model(x)
            prec = torch.exp(logprec)

            # Compute metrics
            for m in metrics:
                if m == 'mse':
                    ths_res = ((y - mu) ** 2).sum(axis=0)
                elif m == 'mae':
                    ths_res = torch.abs(y - mu).sum(axis=0)
                elif m == 'mle':
                    ths_res = (prec * (y - mu) ** 2 - logprec).sum(axis=0)
                elif m == 'crps':
                    # Analytically
                    w = (y - mu) * prec**0.5
                    ths_res = (w * (2 * norm().cdf(w) - 1) + 2 & norm().pdf(w) - np.pi**(-0.5)) / prec**0.5
                elif m == 'crps_ecdf':
                    # Using the ecdf
                    y_hats = torch.cat([sample(x) for _ in range(100)])

                else:
                    raise ValueError('Unknown metric')
                results[m] += ths_res
    for m in metrics:
        results[m] /= len(data.dataset)
        if plot:
            plt.figure()
            res = results[m]
            plt.bar(range(len(res)), res.cpu())
            plt.semilogy()
            plt.ylabel(m)
            plt.tight_layout()
        results[m] = results[m].mean().cpu().numpy()
        print('%s: %.4g' % (m, results[m]))
    if plot:
        # plt.show()
        plt.savefig('results/tmp_metrics.png')
        plt.close('all')
    return results


def train_and_eval(train_params, metrics, save_path=None):
    """
    Trains a model and evaluates on the validation data

    Input:
    ------
    training_params - [nested dict] the parameter configuration file including values for batch_size, model_params, etc
    metrics - [list] list of metrics to evaluate
    save_path - [str] file path to save trained model to
    """
    data_train, data_val = get_data(batch_size=train_params.pop('batch_size'))
    model, sample = train(train_params, data_train, load_path=None, save_path=save_path)
    # eval(model, data_train, metrics)
    return eval(model, data_val, metrics, smaple=sample)


def load_eval(save_dir, metrics, i=0):
    with open(save_dir + 'records.pickle', 'rb') as f:
        record = pickle.load(f)
        id, train_params = record[1][i], record[2][i]
        _, data_val = get_data(batch_size=train_params.pop('batch_size'), val_only=True)
        model, sample = train(train_params, load_path=save_dir + '%d.cp' % id)
        eval(model, data_val, metrics, sample=sample)


if __name__ == '__main__':
    sweep = {
        'model_params': {
            'hidden_dims': {
                'values': [128, 256, 512, 1024, 2048]
            },
            'layers': {
                'values': [1, 2, 3, 4]
            }
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
        'gamma': {
            'max': 1,
            'min': 0.13,
            'distribution': 'log_uniform'
        },
        'loss_type': 'mle',
        'batch_size': {
            'values': [1024, 2048, 4096, 8192, 16384],
        }
    }
    metrics = ['mse', 'mae', 'mle', 'crps', 'crps_ecdf']

    hyperparameter_tuning(
        sweep, partial(train_and_eval, metrics=metrics), 'mse', runs=100, save_dir='models/hetreg_sweep/')

    # train_params = {
    #     'model_params': {
    #         'latent_dims': 15,
    #         'hidden_dims': 1024,
    #         'layers': 2,
    #         'beta': 0.0001
    #     },
    #     'epochs': 1,
    #     'optimizer': 'adam',
    #     'lr': 0.0001,
    #     'weight_decay': 0.001,
    #     'loss_type': 'mse',
    #     'batch_size': 4096,
    # }
    # train_and_eval(train_params, data_train, data_val, metrics, save_path='models/test.cp')

    # load_eval('models/hetreg_sweep_old/', metrics)
