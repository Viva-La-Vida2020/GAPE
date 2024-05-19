import os
import torch
import torch.nn as nn
from nowcasting.layers.generation.generative_network import Generative_Encoder, Generative_Decoder
from nowcasting.layers.generation.noise_projector import Noise_Projector


class Generator(nn.Module):
    def __init__(self, configs):
        super(Generator, self).__init__()
        self.configs = configs
        self.pred_length = self.configs.total_length - self.configs.input_length

        # self.evo_net = Evolution_Network(self.configs.input_length, self.pred_length, base_c=32)
        self.gen_enc = Generative_Encoder(self.configs.total_length, base_c=self.configs.ngf)
        self.gen_dec = Generative_Decoder(self.configs)
        self.proj = Noise_Projector(self.configs.ngf, configs)
        if configs.mrms_pretrained:
            self.load_mrms_ckpt()

    def load_mrms_ckpt(self):
        mrms_ckpt_path = '/mnt/ssd/liyiyao/projects/NowcastNet/checkpoints/mrms'
        self.gen_enc.load_state_dict(torch.load(os.path.join(mrms_ckpt_path, 'GenEnc.pth')))
        self.gen_dec.load_state_dict(torch.load(os.path.join(mrms_ckpt_path, 'GenDec.pth')))
        self.proj.load_state_dict(torch.load(os.path.join(mrms_ckpt_path, 'Proj.pth')))
        print("Load ckpt from pretrained mrms")


    def forward(self, all_frames, evo_result):
        all_frames = all_frames[:, :, :, :, :1]  # [1,29,512,512,1]
        frames = all_frames.permute(0, 1, 4, 2, 3)
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        # Input Frames
        input_frames = frames[:, :self.configs.input_length]
        input_frames = input_frames.reshape(batch, self.configs.input_length, height, width)  # [1,9,512,512]

        # Generative Network
        evo_feature = self.gen_enc(torch.cat([input_frames, evo_result], dim=1))  # [1,20,64,64]
        noise = torch.randn(batch, self.configs.ngf, height // 32, width // 32).cuda(evo_feature.device)
        noise_feature = self.proj(noise).reshape(batch, -1, 4, 4, 8, 8).permute(0, 1, 4, 5, 2, 3).reshape(batch, -1,
                                                                                                          height // 8,
                                                                                                          width // 8)
        feature = torch.cat([evo_feature, noise_feature], dim=1)  # [1,320,64,64]
        gen_result = self.gen_dec(feature, evo_result)  # [1,20,512,512]
        return gen_result.unsqueeze(-1)