from typing import cast

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformers import AutoModel, AutoProcessor

from unite_detection.models import TemporalPositionalEncoding, ViTEncoder
from unite_detection.schemas import UNITEConfig, UNITEOutput
from unite_detection.utils import GPUSigLIPProcessor


class UNITE(nn.Module):
    def __init__(self, config: UNITEConfig | None = None):
        super().__init__()

        if config is None:
            config = UNITEConfig()

        self.config = config

        dtype = torch.bfloat16 if config.use_bfloat else torch.float16
        self.vis_encoder = AutoModel.from_pretrained(
            config.encoder_model,
            device_map="auto",
            dtype=dtype,
            attn_implementation="flash_attention_2",
        )
        self.embed_size = self.vis_encoder.config.vision_config.hidden_size
        processor = AutoProcessor.from_pretrained(config.encoder_model, use_fast=True)
        self.processor = GPUSigLIPProcessor(processor)

        for para in self.vis_encoder.parameters():
            para.requires_grad = False
        self.vis_encoder.eval()

        self.class_token: Float[Tensor, "embed"] = nn.Parameter(
            torch.randn((self.embed_size,)), requires_grad=True
        )

        self.pos_embedding = TemporalPositionalEncoding(
            self.embed_size, config.num_frames, config.dropout
        )
        self.first_encoder = ViTEncoder(
            self.embed_size, config.num_heads, config.dropout
        )
        self.encoders = nn.ModuleList(
            [
                ViTEncoder(self.embed_size, config.num_heads, config.dropout)
                for _ in range(3)
            ]
        )
        self.mlp_head = nn.Linear(self.embed_size, config.num_cls)

    def forward(
        self,
        x: Int[Tensor, "batch channel frame h w"],
        return_ad_param=False,
        return_embed=False,
    ) -> UNITEOutput:
        self.vis_encoder.eval()

        # Input: [batch, c, frame, h, w]
        b, _, f, *_ = x.shape

        with torch.no_grad():
            if self.config.cpu_preprocess:
                pixels_int: Int[Tensor, "bf channel h w"] = x.permute(
                    0, 2, 1, 3, 4
                ).flatten(0, 1)
                pixels: Float[Tensor, "bf channel h w"] = pixels_int.to(
                    self.vis_encoder.dtype
                )
            else:
                pixels: Float[Tensor, "bf channel h w"] = self.processor(x)
            encoded = cast(
                Float[Tensor, "bf token embed"],
                self.vis_encoder.vision_model(pixel_values=pixels).last_hidden_state,
            )
        encoded_reshape: Float[Tensor, "batch frame token embed"] = encoded.reshape(
            b, f, -1, self.embed_size
        )
        train_in: Float[Tensor, "batch frame token embed"] = self.pos_embedding(
            encoded_reshape
        )

        _, _, t, d = train_in.shape
        # Reshape for transformer
        # tot_token = frame * token
        train_reshape: Float[Tensor, "batch tot_token embed"] = train_in.reshape(
            b, t * f, d
        )
        cls_token: Float[Tensor, "batch 1 embed"] = self.class_token.view(
            1, 1, -1
        ).expand(b, -1, -1)
        # tot_w_cls = tot_token + 1
        transform_in: Float[Tensor, "batch tot_w_cls embed"] = torch.cat(
            [cls_token, train_reshape], dim=1
        )

        P: Float[Tensor, "batch head frame"] | None = None
        transformer_out: Float[Tensor, "batch tot_w_cls embed"]

        if return_ad_param:
            transformer_out, attn_output = cast(
                tuple[
                    Float[Tensor, "batch tot_w_cls embed"],
                    Float[Tensor, "batch tot_w_cls head head_dim"],
                ],
                self.first_encoder(transform_in, return_attn_output=True),
            )  # head_dim = embed // head

            attn_output: Float[Tensor, "batch head tot_token head_dim"]
            attn_output = attn_output.permute(0, 2, 1, 3)[:, :, 1:, :]

            train_in_reshape: Float[Tensor, "batch tot_token head head_dim"]
            train_in_reshape = train_in.reshape(
                b, -1, self.config.num_heads, d // self.config.num_heads
            )

            train_in_permute: Float[Tensor, "batch head tot_token head_dim"]
            train_in_permute = train_in_reshape.permute(0, 2, 1, 3)

            P_val: Float[Tensor, "batch head tot_token head_dim"] = (
                attn_output * train_in_permute
            )
            P_reshape: Float[Tensor, "batch head frame token head_dim"]
            P_reshape = P_val.reshape(b, self.config.num_heads, f, t, -1)
            P = P_reshape.mean(dim=(3, 4))
        else:
            transformer_out: Float[Tensor, "batch tot_w_cls embed"] = (
                self.first_encoder(x)
            )

        for encoder in self.encoders:
            transformer_out: Float[Tensor, "batch tot_w_cls embed"] = encoder(
                transformer_out
            )

        # Get only cls_token
        cls_token_out: Float[Tensor, "batch 1 embed"] = transformer_out[:, 0, :]
        embed: Float[Tensor, "batch embed"] = cls_token_out.reshape(b, -1)
        res: Float[Tensor, "batch cls"] = self.mlp_head(embed)
        ret = UNITEOutput(
            res=res,
            ad_param=P if return_ad_param else None,
            embed=embed if return_embed else None,
        )
        return ret
