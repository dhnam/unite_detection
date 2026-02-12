import math
from typing import cast, override

import torch
from jaxtyping import Float
from torch import Tensor, nn


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, num_frames: int = 32, dropout: float = 0.1):
        """
        Args:
            embed_dim: Feature dimension (ds = 768)
            num_frames: Maximum number of frames (nf = 32)
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout: nn.Module = nn.Dropout(p=dropout)

        # 1. Frame Index (j) 생성: 0 ~ max_len-1
        # 논문의 'j'에 해당합니다.
        position: Float[Tensor, "frame 1"] = (
            torch.arange(num_frames).unsqueeze(1).float()
        )

        # 2. Div Term 계산 (논문 Eq 1의 분모 부분)
        # 10000^(2i/ds) 부분을 로그 스케일로 계산
        div_term: Float[Tensor, "frame embed_half"] = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        # 3. PE Matrix 초기화 [max_len, d_model]
        pe_raw: Float[Tensor, "frame embed"] = torch.zeros(num_frames, embed_dim)

        # 4. 논문 수식 (Eq 1) 적용
        # PE(j, 2i) = sin(...) -> 짝수 인덱스
        pe_raw[:, 0::2] = torch.sin(position * div_term)
        # PE(j, 2i+1) = cos(...) -> 홀수 인덱스
        pe_raw[:, 1::2] = torch.cos(position * div_term)

        # 5. 차원 확장 (Broadcasting 준비)
        # PE는 [Frame, Dim] 정보를 담고 있습니다.
        # 입력 x: [Batch, Frames, Tokens, Dim]
        # PE  : [1,     Frames, 1,      Dim] 형태로 만들어야
        # Batch와 Tokens 차원으로 자동 확장(Broadcast)되어 더해집니다.
        pe: Float[Tensor, "1 frame 1 embed"] = pe_raw.unsqueeze(0).unsqueeze(2)

        self.register_buffer("pe", pe)

    @override
    def forward(
        self, x: Float[Tensor, "batch frame token embed"]
    ) -> Float[Tensor, "batch frame token embed"]:
        """
        Args:
            x: [Batch, Frames, Tokens, Dim]
               (예: [B, 32, 576, 768])
        Returns:
            x: Temporal PE가 더해진 텐서
        """
        self.pe: Float[Tensor, "1 frame 1 embed"]  # pyright: ignore[reportUninitializedInstanceVariable]
        # 입력된 비디오의 실제 프레임 수만큼만 PE를 잘라서 사용
        current_frames = x.size(1)

        # x에 PE를 더함.
        # pe[:, :current_frames, :, :]의 shape은 [1, F, 1, D]
        # x의 shape [B, F, T, D]에 맞춰서,
        # 같은 프레임(F)에 있는 모든 토큰(T)들에게 동일한 PE 벡터가 더해짐.
        x = x + self.pe[:, :current_frames, :, :]

        return cast(Float[Tensor, "batch froame token embed"], self.dropout(x))
