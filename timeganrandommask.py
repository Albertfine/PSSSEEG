"""
Created on Sat Oct 29 15:46:43 2022

@author: 12960
"""

from itertools import chain

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from time_dataset import  batch_generator, batch_mask
from visualization_metrics import visualization


class Net(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        rnn=nn.GRU,
        activation_fn=torch.sigmoid,
    ):
        super().__init__()
        self.rnn = rnn(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.activation_fn = activation_fn

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


def train(
    batch_size,
    max_steps,
    dataset,
    device,
    embedder,
    generator,
    supervisor,
    recovery,
    discriminator,
    gamma,
    ratio
):
    def _loss_e_t0(x_tilde, x):
        return F.mse_loss(x_tilde, x)

    def _loss_e_0(loss_e_t0):
        return torch.sqrt(loss_e_t0) * 10

    def _loss_e(loss_e_0, loss_s):
        return loss_e_0 + 0.1 * loss_s

    def _loss_s(h_hat_supervise, h):
        return F.mse_loss(h[:, 1:, :], h_hat_supervise[:, 1:, :])

    def _loss_g_u(y_fake):
        return F.binary_cross_entropy_with_logits(y_fake, torch.ones_like(y_fake))

    def _loss_g_u_e(y_fake_e):
        return F.binary_cross_entropy_with_logits(y_fake_e, torch.ones_like(y_fake_e))

    def _loss_g_v(x_hat, x):
        loss_g_v1 = torch.mean(
            torch.abs(torch.sqrt(torch.var(x_hat, 0) + 1e-6) - torch.sqrt(torch.var(x, 0) + 1e-6))
        )
        loss_g_v2 = torch.mean(torch.abs(torch.mean(x_hat, 0) - torch.mean(x, 0)))
        return loss_g_v1 + loss_g_v2

    def _loss_g(loss_g_u, loss_g_u_e, loss_s, loss_g_v):
        return loss_g_u + gamma * loss_g_u_e + 100 * torch.sqrt(loss_s) + 100 * loss_g_v

    def _loss_d(y_real, y_fake, y_fake_e):
        loss_d_real = F.binary_cross_entropy_with_logits(y_real, torch.ones_like(y_real))
        loss_d_fake = F.binary_cross_entropy_with_logits(y_fake, torch.zeros_like(y_fake))
        loss_d_fake_e = F.binary_cross_entropy_with_logits(y_fake_e, torch.zeros_like(y_fake_e))
        return loss_d_real + loss_d_fake + gamma * loss_d_fake_e

    optimizer_er = optim.Adam(chain(embedder.parameters(), recovery.parameters()))
    optimizer_gs = optim.Adam(chain(generator.parameters(), supervisor.parameters()))
    optimizer_d = optim.Adam(discriminator.parameters())

    embedder.train()
    generator.train()
    supervisor.train()
    recovery.train()
    discriminator.train()

    print("Start Embedding Network Training")
    for step in range(1, max_steps + 1):
        x = batch_generator(dataset, batch_size).to(device)

        h = embedder(x)
        x_tilde = recovery(h)

        loss_e_t0 = _loss_e_t0(x_tilde, x)
        loss_e_0 = _loss_e_0(loss_e_t0)
        optimizer_er.zero_grad()
        loss_e_0.backward()
        optimizer_er.step()

        if step % 1000 == 0:
            print(
                "step: "
                + str(step)
                + "/"
                + str(max_steps)
                + ", loss_e: "
                + str(np.round(np.sqrt(loss_e_t0.item()), 4))
            )
    print("Finish Embedding Network Training")

    print("Start Training with Supervised Loss Only")
    for step in range(1, max_steps + 1):
        # x = batch_generator(dataset, batch_size).to(device)
        # _,z,_ = batch_mask(x, ratio)
        x,_,z,_ = batch_mask(dataset, batch_size, ratio)
        x = x.to(device)

        h = embedder(x) 
        h_hat_supervise = supervisor(h)

        loss_s = _loss_s(h_hat_supervise, h)
        optimizer_gs.zero_grad()
        loss_s.backward()
        optimizer_gs.step()

        if step % 1000 == 0:
            print(
                "step: "
                + str(step)
                + "/"
                + str(max_steps)
                + ", loss_s: "
                + str(np.round(np.sqrt(loss_s.item()), 4))
            )
    print("Finish Training with Supervised Loss Only")

    print("Start Joint Training")
    for step in range(1, max_steps + 1):
        for _ in range(2):
            # x = batch_generator(dataset, batch_size).to(device)
            # z = torch.randn(batch_size, x.size(1), x.size(2)).to(device)
            x,_,_,z = batch_mask(dataset, batch_size, ratio)
            x = x.to(device)
            z = z.to(device)

            h = embedder(x)
            e_hat = generator(z)
            h_hat = supervisor(e_hat)
            h_hat_supervise = supervisor(h)
            x_hat = recovery(h_hat)
            y_fake = discriminator(h_hat)
            y_fake_e = discriminator(e_hat)

            loss_s = _loss_s(h_hat_supervise, h)
            loss_g_u = _loss_g_u(y_fake)
            loss_g_u_e = _loss_g_u_e(y_fake_e)
            loss_g_v = _loss_g_v(x_hat, x)
            loss_g = _loss_g(loss_g_u, loss_g_u_e, loss_s, loss_g_v)
            optimizer_gs.zero_grad()
            loss_g.backward()
            optimizer_gs.step()

            h = embedder(x)
            x_tilde = recovery(h)
            h_hat_supervise = supervisor(h)

            loss_e_t0 = _loss_e_t0(x_tilde, x)
            loss_e_0 = _loss_e_0(loss_e_t0)
            loss_s = _loss_s(h_hat_supervise, h)
            loss_e = _loss_e(loss_e_0, loss_s)
            optimizer_er.zero_grad()
            loss_e.backward()
            optimizer_er.step()

        # x = batch_generator(dataset, batch_size).to(device)
        # z = torch.randn(batch_size, x.size(1), x.size(2)).to(device)
        x,_,_,z = batch_mask(dataset, batch_size, ratio)
        x = x.to(device)
        z = z.to(device)

        h = embedder(x)
        e_hat = generator(z)
        h_hat = supervisor(e_hat)
        y_fake = discriminator(h_hat)
        y_real = discriminator(h)
        y_fake_e = discriminator(e_hat)

        loss_d = _loss_d(y_real, y_fake, y_fake_e)
        if loss_d.item() > 0.15:
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

        if step % 1000 == 0:
            print(
                "step: "
                + str(step)
                + "/"
                + str(max_steps)
                + ", loss_d: "
                + str(np.round(loss_d.item(), 4))
                + ", loss_g_u: "
                + str(np.round(loss_g_u.item(), 4))
                + ", loss_g_v: "
                + str(np.round(loss_g_v.item(), 4))
                + ", loss_s: "
                + str(np.round(np.sqrt(loss_s.item()), 4))
                + ", loss_e_t0: "
                + str(np.round(np.sqrt(loss_e_t0.item()), 4))
            )
    print("Finish Joint Training")


def visualize(dataset, device, generator, supervisor, recovery, batch_size, ratio):
    # generator.load_state_dict(torch.load("generator.pt"))
    # supervisor.load_state_dict(torch.load("supervisor.pt"))
    # recovery.load_state_dict(torch.load("recovery.pt"))
    seq_len = dataset[0].shape[0]
    input_size = dataset[0].shape[1]
    dataset_size = 10*len(dataset)

    generator.eval()
    supervisor.eval()
    recovery.eval()
    
    if dataset_size > batch_size:
        n = int(dataset_size/batch_size)
        # z = torch.Tensor([])
        # e_hat = torch.Tensor([])
        # h_hat = torch.Tensor([])
        x_hat = torch.Tensor([])
        x_origin = torch.Tensor([])
        for i in range(n):
            z0,_,_,z = batch_mask(dataset, batch_size, ratio)
            x_origin = torch.cat((x_origin, z0), dim=0)
            # z = torch.cat((z, zp), dim=0)
            z = z.to(device)
            with torch.no_grad():
                e_hat = generator(z)
                h_hat = supervisor(e_hat)
                x_hat = torch.cat((x_hat, (recovery(h_hat).cpu())), dim=0)
                
        if dataset_size/n -batch_size !=0:
            z0,_,_,z = batch_mask(dataset, dataset_size-n*batch_size, ratio)
            x_origin = torch.cat((x_origin, z0), dim=0)
            z = z.to(device)
            with torch.no_grad():
                e_hat = generator(z)
                h_hat = supervisor(e_hat)
                x_hat = torch.cat((x_hat, (recovery(h_hat).cpu())), dim=0)
            
    else:
        z0,_,_,z = batch_mask(dataset, dataset_size, ratio)
        x_origin = torch.cat((x_origin, z0), dim=0)
        z = z.to(device)
        with torch.no_grad():
            e_hat = generator(z)
            h_hat = supervisor(e_hat)
            x_hat = recovery(h_hat)
    


    # visualization(dataset, generated_data_curr, "pca")
    # visualization(dataset, generated_data_curr, "tsne")
    
    return generated_data_curr, generated_data_origin

