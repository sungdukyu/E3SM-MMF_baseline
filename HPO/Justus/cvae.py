import pickle
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from data import get_data
from tools import progress, hyperparameter_tuning

sns.set_theme(style='whitegrid')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VariationalEncoder(torch.nn.Module):
    """
    Conditional VAE Encoder with <layers>+1 fully connected layer
    """
    def __init__(self, in_dims, hidden_dims=512, latent_dims=3, layers=1):
        super().__init__()
        self.linears = []
        for i in range(layers):
            self.linears += [torch.nn.Linear(in_dims if i == 0 else hidden_dims, hidden_dims)]
            self.add_module('linear%d' % i, self.linears[-1])
        self.linear_mean = torch.nn.Linear(hidden_dims, latent_dims)
        self.linear_logstd = torch.nn.Linear(hidden_dims, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, y, x, return_latent=False):
        y = torch.cat([y, x], 1)
        y = torch.flatten(y, start_dim=1)
        for linear in self.linears:
            y = torch.nn.functional.relu(linear(y))
        mu = self.linear_mean(y)
        if return_latent:
            return mu
        else:
            sigma = torch.exp(self.linear_logstd(y))
            z = mu + sigma * self.N.sample(mu.shape)
            self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
            return z


class Decoder(torch.nn.Module):
    """
    Conditional VAE Decoder with <layers>+1 fully connected layer
    """
    def __init__(self, out_dims, hidden_dims=512, latent_dims=3, layers=1):
        super().__init__()
        self.linears = []
        for i in range(layers):
            self.linears += [torch.nn.Linear(latent_dims if i == 0 else hidden_dims, hidden_dims)]
            self.add_module('linear%d' % i, self.linears[-1])
        self.final_linear = torch.nn.Linear(hidden_dims, out_dims)

    def forward(self, z, x):
        z = torch.cat([z, x], 1)
        for linear in self.linears:
            z = torch.nn.functional.relu(linear(z))
        return self.final_linear(z)


class ConditionalVAE(torch.nn.Module):
    def __init__(self, beta=0.01, data_dims=124, label_dims=128,
                 latent_dims=3, hidden_dims=512, layers=2):
        """
        Conditional VAE
        Encoder: [y x] -> [mu/sigma] -sample-> [z]
        Decoder: [z x] -> [y_hat]

        Inputs:
        -------
        beta - [float] trade-off between KL divergence (latent space structure) and reconstruction loss
        data_dims - [int] size of x
        label_dims - [int] size of y
        latent_dims - [int] size of z
        hidden_dims - [int] size of hidden layers
        layers - [int] number of layers, including hidden layer
        """
        super().__init__()
        self.latent_dims = latent_dims
        self.label_dims = label_dims
        # Encoder and Decoder are conditioned on x of size data_dims
        self.encoder = VariationalEncoder(label_dims + data_dims, hidden_dims, latent_dims, layers=layers)
        self.decoder = Decoder(label_dims, hidden_dims, latent_dims + data_dims, layers=layers)
        self.beta = beta

    def forward(self, y, x, return_latent=False):
        z = self.encoder(y, x, return_latent)
        if return_latent:
            return z
        else:
            return self.decoder(z, x)

    def sample(self, x, random=True):
        """
        Sample conditionally on x

        Inputs:
        -------
        x - [BxN array] label
        random - [boolean] if true sample latent variable from prior else use all-zero vector
        """
        if random:
            # Draw from prior
            z = self.encoder.N.sample([x.shape[0], self.latent_dims])
        else:
            # Set to prior mean
            z = torch.zeros([x.shape[0], self.latent_dims]).to(device)
        return self.decoder(z, x)

    def trainer(self, data, epochs=20, save="models/vae.cp", plot=True, loss_type='mse',
                optimizer='adam', lr=0.0001, weight_decay=0):
        """
        Train the Conditional VAE

        Inputs:
        -------
        data - [DataLoader] - training data
        epochs - [int] number of epochs
        loss_type - [str] type of loss
        optimizer - [str] type of optimizer
        lr - [float] learning rate
        weight_decay - [float] L2 regularization
        save - [str] file path to save trained model to after training (and after every 20 minutes)
        plot - [boolean] if plots of loss curves and samples should be produced
        """
        # Training parameters
        if optimizer == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'sgd':
            opt = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Unknown optimizer')

        # Train and checkpoint every 20 minutes
        losses = []
        for epoch, batch in progress(range(epochs), inner=data, text='Training',
                                     timed=[(1200, lambda: torch.save(self.state_dict(), save))]):
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            opt.zero_grad()
            y_hat = self(y, x)
            if loss_type == 'mse':
                # iid gaussians -> mse
                loss = ((y - y_hat) ** 2).sum() / self.label_dims + self.beta * self.encoder.kl / self.latent_dims
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
            fig, ax = plt.subplots(3, 1, sharey=True)
            ax[0].plot((y[0:500] - y_hat[0:500]).detach().cpu().numpy().T, c="C0", alpha=1/255)
            ax[1].plot((y[0:500] - self.sample(x[0:500])).detach().cpu().numpy().T, c="C0", alpha=1/255)
            ax[2].plot((y[0:500] - self.sample(x[0:500], random=False)).detach().cpu().numpy().T, c="C0", alpha=1/255)
            plt.tight_layout()
            plt.savefig('results/tmp_last_batch.png')
            # plt.show()
            plt.close('all')


def train(train_params, data=None, load_path=None, save_path=None):
    """
    Initialize, load, and train a Conditional VAE on the training data, returns the callable regressor / sampler
    """
    vae = ConditionalVAE(**train_params.pop('model_params')).to(device)
    if load_path is not None:
        vae.load_state_dict(torch.load(load_path))
    if data is not None:
        vae.trainer(data, save=save_path, **train_params)
    vae.eval()
    return partial(vae.sample, random=False), vae.sample


def eval(model, data, metrics, sample=None, plot=True):
    """
    Evaluate model on the validation data, returns a dict with entries {metric: value}
    """
    results = {m: 0 for m in metrics}
    with torch.no_grad():
        for batch in progress(data, text='Evaluating'):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            y_hat = model(x)

            # Compute metrics
            for m in metrics:
                if m == 'mse':
                    ths_res = ((y - y_hat) ** 2).sum(axis=0)
                elif m == 'mae':
                    ths_res = torch.abs(y - y_hat).sum(axis=0)
                elif m == 'crps_ecdf':
                    n = 20
                    y_hats = torch.stack([sample(x) for _ in range(n)], 2).sort(dim=-1)[0]
                    # E[Y - y]
                    mae = torch.abs(y[..., None] - y_hats).mean(axis=(0, -1))
                    # E[Y - Y'] = sum_i sum_j |Y_i - Y_j| / n^2
                    diff = y_hats[..., 1:] - y_hats[..., :-1]
                    count = torch.arange(1, n, device=device) * torch.arange(n - 1, 0, -1, device=device)
                    ths_res = mae - (diff * count).sum(axis=-1).mean(axis=0) / (n*(n-1))
                elif m == 'std':
                    # Test sampler
                    n = 20
                    ths_res = torch.stack([sample(x) for _ in range(n)], 2).std(dim=-1).mean(axis=0)
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
            plt.title('%s: %.4g' % (m, res.mean()))
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
    return eval(model, data_val, metrics, sample=sample)


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
            'latent_dims': {
                'values': [4, 8, 16, 32, 64]
            },
            'hidden_dims': {
                'values': [128, 256, 512, 1024, 2048]
            },
            'layers': {
                'values': [1, 2, 3, 4]
            },
            'beta': {
                'max': 1,
                'min': 0.00001,
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
            'max': 0.1,
            'min': 0.00001,
            'distribution': 'log_uniform'
        },
        'loss_type': 'mse',
        'batch_size': {
            'values': [1024, 2048, 4096, 8192, 16384],
        }
    }
    metrics = ['mse', 'mae', 'crps_ecdf', 'std']

    hyperparameter_tuning(
        sweep, partial(train_and_eval, metrics=metrics), 'mse', runs=200, save_dir='models/cvae_sweep2/')

    # train_params = {
    #     'model_params': {
    #         'latent_dims': 16,
    #         'hidden_dims': 512,
    #         'layers': 3,
    #         'beta': 0.01
    #     },
    #     'epochs': 5,
    #     'optimizer': 'adam',
    #     'lr': 0.0006,
    #     'weight_decay': 0.004,
    #     'loss_type': 'mse',
    #     'batch_size': 4096,
    # }
    # train_and_eval(train_params, data_train, data_val, metrics, save_path='models/test.cp')

    # load_eval('models/cvae_sweep/', metrics)
