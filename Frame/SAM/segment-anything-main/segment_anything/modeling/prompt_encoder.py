# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension                 prompt的嵌入维度
          image_embedding_size (tuple(int, int)): The spatial size of the   图像嵌入的空间大小
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input  作为输入的图像的填充大小
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for  用于编码输入掩码的隐藏通道数
            encoding input masks.
          activation (nn.Module): The activation to use when encoding 编码输入掩码时使用的激活函数
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim #prompt的嵌入维度
        self.input_image_size = input_image_size 
        self.image_embedding_size = image_embedding_size  #图像嵌入的空间大小，VIT处理后的图像大小？
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2) #创建位置编码
        #将输入的位置信息（如点击的点和框的角点）编码成特征向量，供模型使用
        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners   正/负点 + 2个框角，点嵌入的数量 
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)] #创建四个嵌入层
        self.point_embeddings = nn.ModuleList(point_embeddings) #将四个嵌入层放入ModuleList中
        self.not_a_point_embed = nn.Embedding(1, embed_dim) #创建一个嵌入层，输入不是点的嵌入

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1]) #输入掩码的大小
        self.mask_downscaling = nn.Sequential( #mask下采样
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim) #表示没有 mask prompt

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0) # 1 x C x H x W

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)#坐标与图像大小进行位置编码
        #增加相应可学习的嵌入向量
        point_embedding[labels == -1] = 0.0 #不是点
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight#负点
        point_embedding[labels == 1] += self.point_embeddings[1].weight#正点
        return point_embedding
    # boxes -> (batch_size, num_boxes, 4)：每个框有4个坐标，(x1, y1, x2, y2)
    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2) #2, 2 表示每个框有两个角点，每个角点有两个坐标（x, y）。  coords ->(batch_size * num_boxes, 2, 2) 
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings

#对坐标进行标准化，然后乘上一个可训练参数，再去sin和cos做编码
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """
    #num_pos_feats: 指定位置特征的数量
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None: 
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        #在模型中注册一个缓冲区，模型的一部分，但不会作为参数进行优化。
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)), #(2, num_pos_feats) 的高斯随机矩阵
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1].对归一化到 [0, 1] 范围内的点进行位置编码。"""
        # assuming coords are in [0, 1]^2(二维空间) square and have [d_1 , ... , d_n , 2] shape
        coords = 2 * coords - 1 #归一化到 [-1, 1] 范围内
        #coords -> [d_1 , ... , d_n , num_pos_feats] shape
        coords = coords @ self.positional_encoding_gaussian_matrix #与高斯矩阵(register_buffer)进行矩阵乘法，生成新的编码坐标。
        coords = 2 * np.pi * coords  #将坐标乘以 2π
        # outputs [d_1 , ... , d_n , C] shape  
        #对编码坐标分别应用正弦和余弦函数，最后一个维度上进行拼接，生成最终的编码结果。C是编码后的特征维度,C = num_pos_feats * 2
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    #用于生成指定大小的网格的位置信息编码。
    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32) #float32，全为1的矩阵
        y_embed = grid.cumsum(dim=0) - 0.5 #对行进行累加，减去0.5
        x_embed = grid.cumsum(dim=1) - 0.5 #对列进行累加，减去0.5
        #归一化到 [0, 1] 范围内，减去0.5是使得归一化的结果中心化，不然会偏向一侧，上方。
        y_embed = y_embed / h
        x_embed = x_embed / w
        #将x和y的位置编码进行最后一个维度拼接，然后进行位置编码。
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W
    #用于对未归一化到 [0,1] 范围内的点进行位置编码。
    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()#以避免对原始数据进行修改。
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]#归一化到 [0, 1] 范围内
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
