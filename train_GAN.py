import os
import shutil
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from nowcasting.models.generator import Generator
from nowcasting.models.evonet import EvoNet
from nowcasting.models.temporal_discriminator import Temporal_Discriminator
from torchvision.utils import save_image
import torch.nn.functional as F

from tqdm import tqdm
import nowcasting.evaluator as evaluator
import time
import sys
from omegaconf import OmegaConf
from sevir_loader.sevir_torch_wrap import get_sevir_datamodule
from sevir_loader.sevir_dataloader import SEVIR_CATALOG, SEVIR_DATA_DIR, SEVIR_RAW_SEQ_LEN, \
    SEVIR_LR_CATALOG, SEVIR_LR_DATA_DIR, SEVIR_LR_RAW_SEQ_LEN, SEVIRDataLoader
import datetime


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='NowcastNet')

parser.add_argument('--device_ids', nargs='+', type=int, default=[0,1,2,3,4,5,6,7], help='List of device IDs for DataParallel')
parser.add_argument('--input_length', type=int, default=9)
parser.add_argument('--total_length', type=int, default=29)  # 9+13=22
parser.add_argument('--img_height', type=int, default=384)
parser.add_argument('--img_width', type=int, default=384)
parser.add_argument('--img_ch', type=int, default=1)
parser.add_argument('--mrms_pretrained', type=bool, default=True)
parser.add_argument('--generator_ckpt_path', type=str, default='./checkpoints/')
parser.add_argument('--evonet_ckpt_path', type=str, default='./checkpoints/')
# parser.add_argument('--discriminator_ckpt_path', type=str, default='/mnt/ssd/liyiyao/projects/NowcastNet/checkpoints/epoch_0/ckpt.pth')
parser.add_argument('--save_path', type=str, default='/ckpoints/test/')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--ngf', type=int, default=32)


args = parser.parse_args()

args.evo_ic = args.total_length - args.input_length
args.gen_oc = args.total_length - args.input_length
args.ic_feature = args.ngf * 10

def InitDataLoader():
    start_date = datetime.datetime(2017, 1, 1)
    end_date = datetime.datetime(2019, 1, 1)
    # start_date = datetime.datetime(2019, 5, 27)
    # end_date = datetime.datetime(2019, 5, 29)
    dataloader = SEVIRDataLoader(
        data_types=["vil", ],
        seq_len=args.total_length,
        raw_seq_len=49,
        sample_mode="sequent",
        stride=5,
        batch_size=args.batch_size,
        layout="NTHWC",
        num_shard=1, rank=0, split_mode="uneven",
        sevir_catalog=SEVIR_CATALOG,
        sevir_data_dir=SEVIR_DATA_DIR,
        shuffle=True,
        start_date=start_date, end_date=end_date)
    return dataloader


if __name__ == '__main__':
    device = torch.device(args.device_ids[0])
    # Generator
    G = Generator(args).to(device)
    G = nn.DataParallel(G, device_ids=args.device_ids)
    # Evolution Network
    E = EvoNet(args).to(device)
    E = nn.DataParallel(E, device_ids=args.device_ids)
    # Discriminator
    D = Temporal_Discriminator(args).to(device)
    # D.load_state_dict(torch.load("/mnt/ssd/liyiyao/projects/NowcastNet/checkpoints/epoch_0/ckpt.pth")['model_state_dict'])
    D = nn.DataParallel(D, device_ids=args.device_ids)

    G_optimizer = optim.Adam(G.parameters(), lr=3e-6)
    D_optimizer = optim.Adam(D.parameters(), lr=3e-5)
    max_pool = nn.MaxPool2d(kernel_size=(5, 5), stride=(2, 2)).to(device)
    real_labels = torch.ones(args.batch_size, 1).to(device)
    fake_labels = torch.zeros(args.batch_size, 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(os.path.join(args.save_path, 'discriminator')):
        os.mkdir(os.path.join(args.save_path, 'discriminator'))
    if not os.path.exists(os.path.join(args.save_path, 'generator')):
        os.mkdir(os.path.join(args.save_path, 'generator'))
    print("Model Init")

    epoch_G, epoch_D = 0, 0
    while True:
        # Train Discriminator for 10 epoch
        print('Discriminator Training Start')
        loss_total_D, loss_real_total, loss_fake_total = 0, 0, 0
        for e in range(epoch_D, epoch_D + 2):
            log = open(os.path.join(args.save_path, 'log.txt'), 'a')
            # G.eval()
            # E.eval()
            D.train()

            dataloader = InitDataLoader()
            for batch in tqdm(dataloader):
                real_images = torch.FloatTensor(batch['vil']).to(device)

                with torch.no_grad():
                    evo_result = E(real_images)
                    fake_images = G(real_images, evo_result)

                # Discriminator true_loss
                d_outputs_real = D(real_images.squeeze())
                loss_d_real = criterion(d_outputs_real, real_labels)

                # Discriminator fake_loss
                d_outputs_fake = D(torch.cat((real_images[:, :9, :, :, :], fake_images.detach()), dim=1).squeeze())
                loss_d_fake = criterion(d_outputs_fake, fake_labels)

                # Discriminator backward
                loss_D = loss_d_real + loss_d_fake
                D_optimizer.zero_grad()
                loss_D.backward()
                D_optimizer.step()

                loss_total_D += loss_D.item()
                loss_real_total += loss_d_real.item()
                loss_fake_total += loss_d_fake.item()

            loss_total_D /= len(dataloader)
            loss_real_total /= len(dataloader)
            loss_fake_total /= len(dataloader)
            loss_D_str = "{:.6f}".format(loss_total_D)
            loss_real_str = "{:.6f}".format(loss_real_total)
            loss_fake_str = "{:.6f}".format(loss_fake_total)

            print(f"Epoch_D:{e} LossD:{loss_D_str} Loss_real:{loss_real_str} Loss_fake:{loss_fake_str}\n ")
            log.write(f"Epoch_D:{e} LossD:{loss_D_str} Loss_real:{loss_real_str} Loss_fake:{loss_fake_str}\n ")
            log.close()
        # Save Discriminator
        epoch_D += 2
        if not os.path.exists(os.path.join(args.save_path, 'discriminator', f'epoch_{epoch_D}')):
            os.mkdir(os.path.join(args.save_path, 'discriminator', f'epoch_{epoch_D}'))
        torch.save(D.state_dict(), os.path.join(args.save_path, 'discriminator', f'epoch_{epoch_D}/Discriminator.pth'))

        print('Generator Training Start')
        loss_total_G, loss_total_dis, loss_total_adv = 0, 0, 0
        for e in range(epoch_G, epoch_G + 5):
            log = open(os.path.join(args.save_path, 'log.txt'), 'a')
            # E.eval()
            # D.eval()
            G.train()

            dataloader = InitDataLoader()
            for batch in tqdm(dataloader):
                real_images = torch.FloatTensor(batch['vil']).to(device)

                with torch.no_grad():
                    evo_result = E(real_images)
                # NowcastNet Generator
                fake_images = G(real_images, evo_result)

                # Loss_pooling
                real_pooled = max_pool(real_images[:, 9:, :, :, :].squeeze())
                fake_pooled = max_pool(fake_images.squeeze())
                w = torch.minimum(24 * torch.ones_like(real_pooled), 1 + real_pooled)
                weighted_diff = (real_pooled - fake_pooled) * w
                loss_dis = torch.norm(weighted_diff, p=1) / torch.sum(w)

                # Loss_adv
                d_outputs_fake = D(torch.cat((real_images[:, :9, :, :, :], fake_images), dim=1).squeeze())
                loss_adv = criterion(d_outputs_fake, real_labels)

                loss_G = 6 * loss_adv + 20 * loss_dis
                G_optimizer.zero_grad()
                loss_G.backward()
                G_optimizer.step()

                loss_total_G += loss_G.item()
                loss_total_adv += loss_adv.item()
                loss_total_dis += loss_dis.item()

            loss_total_G /= len(dataloader)
            loss_total_dis /= len(dataloader)
            loss_total_adv /= len(dataloader)
            loss_G_str = "{:.6f}".format(loss_total_G)
            loss_dis_str = "{:.6f}".format(loss_total_dis)
            loss_adv_str = "{:.6f}".format(loss_total_adv)

            print(f"Epoch_G:{e} LossG{loss_G_str} Loss_dis{loss_dis_str} loss_adv{loss_adv_str}\n ")
            log.write(f"Epoch_G:{e} LossG{loss_G_str} Loss_dis{loss_dis_str} loss_adv{loss_adv_str}\n ")
            log.close()
        # Save Generators
        epoch_G += 5
        if not os.path.exists(os.path.join(args.save_path, 'generator', f'epoch_{epoch_G}')):
            os.mkdir(os.path.join(args.save_path, 'generator', f'epoch_{epoch_G}'))
        torch.save(G.state_dict(), os.path.join(args.save_path, 'generator', f'epoch_{epoch_G}/Generator.pth'))

