from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from torchmetrics.classification import MulticlassROC
from torchmetrics.functional.classification import multiclass_confusion_matrix

from unite_detection.schemas import PlotContext, VisualizationData


def plot_roc(data: VisualizationData, ctx: PlotContext):
    roc = MulticlassROC(len(ctx.class_names))
    roc.update(data.logits, data.labels)
    fig, _ = roc.plot()
    roc.reset()
    return fig


def plot_conf_mat(data: VisualizationData, ctx: PlotContext):
    conf_matrix = multiclass_confusion_matrix(
        data.preds, data.labels, len(ctx.class_names)
    )
    fig = plt.figure(figsize=(8, 6))
    _ = sns.heatmap(
        conf_matrix.numpy(),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ctx.class_names,
        yticklabels=ctx.class_names,
    )
    _ = plt.xlabel("Predicted")
    _ = plt.ylabel("True")
    _ = plt.title(f"{ctx.stage.capitalize()} Confusion Matrix")

    return fig


def plot_tsne(data: VisualizationData, ctx: PlotContext, max_samples:int=20000):
    # 데이터가 너무 많으면 t-SNE 속도가 매우 느려지므로 샘플링
    num_samples = data.embeds.shape[0]
    embeds = data.embeds
    labels = data.labels
    preds = data.preds
    if num_samples > max_samples:
        indices = torch.randperm(num_samples)[:max_samples]
        embeds = embeds[indices]
        labels = labels[indices]
        preds = preds[indices]

    # 텐서를 numpy로 변환
    X = embeds.numpy()
    y = labels.numpy()

    # t-SNE 계산
    # perplexity는 데이터 수보다 작아야 함 (보통 30~50)
    n_components = 2
    perplexity = min(30, num_samples - 1)
    if perplexity < 5:
        return  # 데이터가 너무 적으면 패스

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        max_iter=1000,
        random_state=42,
    )
    X_embedded = cast(np.ndarray, tsne.fit_transform(X))

    # DataFrame 생성 (시각화 용이성 위해)
    df_tsne = pd.DataFrame(X_embedded, columns=["x", "y"])
    df_tsne["label"] = [ctx.class_names[i] for i in y]

    style_col = None
    y_pred = preds
    df_tsne["prediction"] = [ctx.class_names[i] for i in y_pred]
    style_col = "prediction"

    # Plotting
    fig = plt.figure(figsize=(15, 12))
    sns.scatterplot(
        data=df_tsne,
        x="x",
        y="y",
        hue="label",
        style=style_col,
        palette=["blue", "red"],  # Real: Blue, Fake: Red
        markers={"Real": "o", "Fake": "X"},
        alpha=0.7,
        s=100,
    )
    plt.title(f"{ctx.stage.capitalize()} t-SNE Visualization")
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()

    return fig


def plot_encoder_tsne(data: VisualizationData, ctx: PlotContext, max_samples:int=400):
    # 데이터 준비
    tsne_embeds: list[np.ndarray] = []
    tsne_labels: list[str] = []  # Real/Fake 구분용
    tsne_heads: list[int] = []  # Head ID 구분용 (색상)
    tsne_types: list[Literal["Center", "Sample"]] = []  # Center vs Sample 구분용 (크기)
    assert data.cs is not None and data.ps is not None

    # [Centers 추출]
    for c_idx in range(len(ctx.class_names)):
        for h_idx in range(ctx.num_heads):
            c_vec = data.cs[c_idx, h_idx].numpy()
            tsne_embeds.append(c_vec)
            tsne_labels.append(ctx.class_names[c_idx])  # "Real" or "Fake"
            tsne_heads.append(h_idx)
            tsne_types.append("Center")

    # [Samples (P) 추출 및 샘플링]
    num_total_samples = data.ps.shape[0]
    sample_size = min(
        num_total_samples, max_samples
    )  # 브라우저 부하 방지를 위해 적절히 조절
    indices = torch.randperm(num_total_samples)[:sample_size]

    for i in indices:
        label_name = ctx.class_names[int(data.labels[i].item())]
        for h_idx in range(ctx.num_heads):
            p_vec = data.ps[i, h_idx].numpy()
            tsne_embeds.append(p_vec)
            tsne_labels.append(label_name)
            tsne_heads.append(h_idx)
            tsne_types.append("Sample")

    # t-SNE 실행
    tsne_embeds_arr = np.array(tsne_embeds)
    tsne_2d = cast(
        np.ndarray,
        TSNE(n_components=2, random_state=42).fit_transform(tsne_embeds_arr),
    )

    # 시각화 시작
    fig, ax = plt.subplots(figsize=(10, 8))

    # 설정: Head별 색상, Class별 마커
    cmap = sns.color_palette("husl", ctx.num_heads)
    marker_map = {
        ctx.class_names[0]: "o",
        ctx.class_names[1]: "X",
    }  # 0: Real(Circle), 1: Fake(X)

    for i in range(len(tsne_2d)):
        h_idx = tsne_heads[i]
        label = tsne_labels[i]
        m_type = tsne_types[i]

        color = cmap[h_idx]
        marker = marker_map.get(label, "d")
        size = 150 if m_type == "Center" else 50
        alpha = 1.0 if m_type == "Center" else 0.4
        edge = "black" if m_type == "Center" else "none"
        zorder = 3 if m_type == "Center" else 2

        _ = ax.scatter(
            tsne_2d[i, 0],
            tsne_2d[i, 1],
            c=[color],
            marker=marker,
            s=size,
            alpha=alpha,
            edgecolors=edge,
            linewidths=1.5,
            zorder=zorder,
        )

    # 범례 설정 (Custom Legend)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Real",
            markerfacecolor="gray",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            label="Fake",
            markerfacecolor="gray",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="Center",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=12,
        ),
    ]
    # Head별 색상 범례 추가
    for h in range(ctx.num_heads):
        legend_elements.append(Line2D([0], [0], color=cmap[h], lw=3, label=f"Head {h}"))

    _ = ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))
    _ = ax.set_title(f"t-SNE of P and C (Epoch {ctx.current_epoch})")
    plt.tight_layout()

    return fig
