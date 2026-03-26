import torch
from dataset import WorldGhibliDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

def train_fn(disc_W, disc_G, gen_G, gen_W, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (ghibli, world) in enumerate(loop):
        ghibli = ghibli.to(config.DEVICE)
        world = world.to(config.DEVICE)

        # Train Discriminators W and G
        with torch.amp.autocast("cuda"):
            fake_ghibli = gen_G(world)
            D_W_real = disc_W(ghibli)
            D_W_fake = disc_W(fake_ghibli.detach())
            D_W_real_loss = mse(D_W_real, torch.ones_like(D_W_real))
            D_W_fake_loss = mse(D_W_fake, torch.zeros_like(D_W_fake))
            D_W_loss = D_W_real_loss + D_W_fake_loss

            fake_world = gen_W(ghibli)
            D_G_real = disc_G(world)
            D_G_fake = disc_G(fake_world.detach())
            D_G_real_loss = mse(D_G_real, torch.ones_like(D_G_real))
            D_G_fake_loss = mse(D_G_fake, torch.zeros_like(D_G_fake))
            D_G_loss = D_G_real_loss + D_G_fake_loss

            D_loss = (D_W_loss + D_G_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators W and G
        with torch.amp.autocast("cuda"):
            # Adversarial loss for both generators
            D_W_fake = disc_W(fake_ghibli)
            D_G_fake = disc_G(fake_world)
            loss_G_W = mse(D_W_fake, torch.ones_like(D_W_fake))
            loss_G_G = mse(D_G_fake, torch.ones_like(D_G_fake))

            # Cycle loss
            cycle_ghibli = gen_G(fake_world)
            cycle_world = gen_W(fake_ghibli)
            cycle_ghibli_loss = L1(ghibli, cycle_ghibli)
            cycle_world_loss = L1(world, cycle_world)

            #identity loss
            identity_ghibli = gen_G(ghibli)
            identity_world = gen_W(world)
            identity_ghibli_loss = L1(ghibli, identity_ghibli)
            identity_world_loss = L1(world, identity_world)

            # add all together
            G_loss = (
                loss_G_W
                + loss_G_G
                + cycle_ghibli_loss * config.LAMBDA_IDENTITY
                + cycle_world_loss * config.LAMBDA_IDENTITY
                + identity_ghibli_loss * config.LAMBDA_CYCLE
                + identity_world_loss * config.LAMBDA_CYCLE
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_ghibli * 0.5 + 0.5, f"saved_images/fake_ghibli_{idx}.png")

def main():
    disc_W = Discriminator(in_channels=3).to(config.DEVICE)
    disc_G = Discriminator(in_channels=3).to(config.DEVICE)
    gen_W = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_G = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_W.parameters()) + list(disc_G.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5,0.999),
    )
    opt_gen = optim.Adam(
        list(gen_W.parameters()) + list(gen_G.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5,0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_W, gen_W, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_G, gen_G, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_w, disc_W, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_G, disc_G, opt_disc, config.LEARNING_RATE,
        )

    dataset = WorldGhibliDataset(
        root_world = "dataset/trainA", root_ghibli= "dataset/trainB_ghibli", transform = config.transforms
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    g_scaler = torch.amp.GradScaler("cuda")
    d_scaler = torch.amp.GradScaler("cuda")

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_W, disc_G, gen_G, gen_W, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_W, opt_gen, filename=config.CHECKPOINT_GEN_W)
            save_checkpoint(gen_G, opt_gen, filename=config.CHECKPOINT_GEN_G)
            save_checkpoint(disc_W, opt_disc, filename=config.CHECKPOINT_CRITIC_W)
            save_checkpoint(disc_G, opt_disc, filename=config.CHECKPOINT_CRITIC_G)


if __name__ == "__main__":
    main()