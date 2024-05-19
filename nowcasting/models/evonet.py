import os
import torch
import torch.nn as nn
from nowcasting.layers.utils import warp, make_grid
from nowcasting.layers.evolution.evolution_network import Evolution_Network


class EvoNet(nn.Module):
    def __init__(self, configs):
        super(EvoNet, self).__init__()
        self.configs = configs
        self.pred_length = self.configs.total_length - self.configs.input_length
        self.evo_net = Evolution_Network(self.configs.input_length, self.pred_length, base_c=32)

        self.load_ckpt()

    def load_ckpt(self):
        self.evo_net.load_state_dict(torch.load(os.path.join(self.configs.evonet_ckpt_path, 'EvoNet.pth')))

    def forward(self, all_frames):
        all_frames = all_frames[:, :, :, :, :1]  # [1,29,512,512,1]

        frames = all_frames.permute(0, 1, 4, 2, 3)
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        # Input Frames
        input_frames = frames[:, :self.configs.input_length]
        input_frames = input_frames.reshape(batch, self.configs.input_length, height, width)  # [1,9,512,512]

        # Evolution Network
        # with torch.no_grad():
        intensity, motion = self.evo_net(input_frames)  # intensity:[1,20,512,512] motion:[1,40,512,512]
        motion_ = motion.reshape(batch, self.pred_length, 2, height, width)  # [1,20,2,512,512]
        intensity_ = intensity.reshape(batch, self.pred_length, 1, height, width)  # [1,20,1,512,512]
        series = []
        last_frames = all_frames[:, (self.configs.input_length - 1):self.configs.input_length, :, :, 0]
        sample_tensor = torch.zeros(1, 1, self.configs.img_height, self.configs.img_width)
        self.grid = make_grid(sample_tensor, last_frames.device)
        grid = self.grid.repeat(batch, 1, 1, 1).to(last_frames.device)
        for i in range(self.pred_length):
            '根据移动场中的每个点指示的方向和距离，调整雷达降水场中的每个点的位置'
            last_frames = warp(last_frames, motion_[:, i], grid, mode="nearest", padding_mode="border")
            '通过直接加上生消场intensity，更新雷达降水场的强度，完成对下一时刻雷达降水场的预测'
            last_frames = last_frames + intensity_[:, i]
            series.append(last_frames)
        evo_result = torch.cat(series, dim=1)
        evo_result = evo_result / 128

        return evo_result