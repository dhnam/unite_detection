from typing import Literal, cast, overload, override

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from unite_detection.schemas import ADLossConfig


class ADLoss(nn.Module):
    def __init__(self, config: ADLossConfig | None = None):
        super().__init__()
        if config is None:
            config = ADLossConfig()

        self.config: ADLossConfig = config
        # C shape: [num_classes, num_heads, max_len]
        # 논문 식(3)에 따라 센터를 각 클래스별로 유지해야 함
        # C = torch.zeros(num_cls, num_heads, max_len)
        # Try changing C to random normalization so that it might not collapse due to initial condition.
        c: Float[Tensor, "cls head frame"]
        c = torch.randn(config.num_cls, config.num_heads, config.num_frames)
        c = F.normalize(c, p=2, dim=2)
        self.register_buffer("C", c)

        delta_within: Float[Tensor, "cls"] = torch.tensor(
            config.delta_within
        )  # [0.01, -2.0] (True, Fake)
        self.register_buffer("delta_within", delta_within)

    @overload
    def forward(
        self,
        P: Float[Tensor, "batch head frame"],
        labels: Int[Tensor, "batch"],
        log_detail: Literal[False],
    ) -> Float[Tensor, ""]: ...

    @overload
    def forward(
        self,
        P: Float[Tensor, "batch head frame"],
        labels: Int[Tensor, "batch"],
        log_detail: Literal[True],
    ) -> tuple[
        Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]
    ]: ...

    @override
    def forward(
        self,
        P: Float[Tensor, "batch head frame"],
        labels: Int[Tensor, "batch"],
        log_detail: bool = False,
    ):
        """
        P: [batch, num_heads, max_len]
        labels: [batch] (Class indices)
        """
        self.C: Float[Tensor, "cls head frame"]  # pyright: ignore[reportUninitializedInstanceVariable]
        self.delta_within: Float[Tensor, "cls"]  # pyright: ignore[reportUninitializedInstanceVariable]
        device = P.device

        # 각 헤드의 특징 벡터(F)를 정규화
        P_norm: Float[Tensor, "batch head frame"] = F.normalize(P, p=2, dim=2)

        loss_between = torch.tensor(0.0, device=device)
        head_dist_mean = torch.tensor(0.0, device=device)
        valid_cls = 0

        # 1. 센터 업데이트 (현재 로직 유지하되 헤드별로 정규화 상태 유지)
        if self.training:
            for c in range(self.config.num_cls):
                mask: Bool[Tensor, "batch"] = labels == c
                if mask.any():
                    valid_cls += 1
                    batch_class_mean: Float[Tensor, "head frame"]
                    batch_class_mean = P_norm[mask].mean(dim=0)
                    with torch.no_grad():
                        # C[c]: [head frame]
                        self.C[c] = (1 - self.config.eta) * self.C[
                            c
                        ] + self.config.eta * batch_class_mean.detach()

                    # Calculating between class loss here (eq. 5)
                    # Note that in paper it is loss between C and C, but I edited it to P and P
                    # because C is detached from graph thus cannot have valid gradient.
                    # So assume center of P is almost C.
                    dist_matrix: Float[Tensor, "head head"]
                    dist_matrix = torch.cdist(batch_class_mean, batch_class_mean, p=2)

                    mask_triu: Bool[Tensor, "head head"] = torch.triu(
                        torch.ones(
                            self.config.num_heads, self.config.num_heads, device=device
                        ),
                        diagonal=1,
                    ).bool()
                    different_heads_dist: Float[Tensor, "n_pair"] = dist_matrix[
                        mask_triu
                    ]
                    head_dist_mean: Float[Tensor, ""]
                    head_dist_mean += different_heads_dist.mean()
                    loss_between: Float[Tensor, ""]
                    loss_between += torch.relu(
                        self.config.delta_between - different_heads_dist
                    ).mean()

        if valid_cls > 0:
            head_dist_mean /= valid_cls
            loss_between /= valid_cls

        # 센터 정규화 (거리 계산 전 필수)
        C_norm: Float[Tensor, "cls head frame"] = F.normalize(self.C, p=2, dim=2)

        # --- 2. Within-class Loss (식 4) ---
        # 각 샘플과 자기 클래스 센터 사이의 거리
        # C_norm[labels]: [H, F]
        diff_within: Float[Tensor, "batch head frame"] = P_norm - C_norm[labels]
        # L2 Norm 계산 (헤드와 프레임 차원에 대해)
        dist_within = cast(
            Float[Tensor, "batch"],
            torch.linalg.vector_norm(diff_within, ord=2, dim=(1, 2)),  # pyright: ignore[reportUnknownMemberType]
        )
        # 각 샘플별 delta 적용
        # loss_within = torch.relu(dist_within - self.delta_within[labels]).mean()

        # 샘플 수와 상관없이 일정한 스케일 유지
        loss_sum = torch.tensor(0.0, device=device)
        for c in range(self.config.num_cls):
            mask_cls: Bool[Tensor, "batch"] = labels == c
            if mask_cls.any():
                # 해당 클래스 샘플들만의 평균을 구함
                class_loss = Float[Tensor, ""]
                class_loss = torch.relu(
                    dist_within[mask_cls] - self.delta_within[c]
                ).mean()
                loss_sum += class_loss
        loss_within = loss_sum / self.config.num_cls  # 클래스당 기여도를 1/N로 고정

        if log_detail:
            return loss_within + loss_between, loss_within, loss_between, head_dist_mean

        return loss_within + loss_between
