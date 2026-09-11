"""
Helper pembuatan chart Plotly untuk dashboard, mengikuti palet yang sudah
divalidasi (lihat skill dataviz): kategori tetap dalam urutan tetap, warna
sekuensial satu hue, dan warna status yang tidak dipakai ulang untuk seri.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Palet tervalidasi -------------------------------------------------
BLUE = "#2a78d6"      # kategori slot 1 / sekuensial
ORANGE = "#eb6834"    # kategori slot 2
AQUA = "#1baf7a"      # kategori slot 3
RED = "#e34948"       # kategori slot 8 / diverging pole

GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

CATEGORICAL_ORDER = [BLUE, ORANGE, AQUA]

BASE_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=TEXT_PRIMARY),
    margin=dict(l=10, r=20, t=40, b=10),
)


def _apply_base_layout(fig: go.Figure, **kwargs) -> go.Figure:
    layout = dict(BASE_LAYOUT)
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


def local_shap_bar(
    labels: list[str],
    shap_values: np.ndarray,
    feature_values: list,
    top_n: int = 10,
) -> go.Figure:
    """Bar horizontal kontribusi SHAP untuk satu prediksi (lokal)."""
    order = np.argsort(-np.abs(shap_values))[:top_n]
    ordered_labels = [labels[i] for i in order][::-1]
    ordered_values = [shap_values[i] for i in order][::-1]
    ordered_feat_vals = [feature_values[i] for i in order][::-1]

    colors = [RED if v > 0 else BLUE for v in ordered_values]
    text = [f"{v:+.3f}" for v in ordered_values]
    hover = [
        f"{lbl}<br>Nilai fitur: {fv}<br>Kontribusi SHAP: {v:+.3f}"
        for lbl, fv, v in zip(ordered_labels, ordered_feat_vals, ordered_values)
    ]

    fig = go.Figure(
        go.Bar(
            x=ordered_values,
            y=ordered_labels,
            orientation="h",
            marker_color=colors,
            text=text,
            textposition="outside",
            hovertext=hover,
            hoverinfo="text",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color=BASELINE)
    max_abs = max(abs(v) for v in ordered_values) if ordered_values else 1.0
    padding = max_abs * 0.35 + 1e-6
    _apply_base_layout(
        fig,
        title="Kontribusi Faktor terhadap Probabilitas Keterlambatan",
        xaxis=dict(
            title="Kontribusi SHAP (mendorong risiko naik / turun)",
            gridcolor=GRIDLINE,
            zeroline=False,
            range=[-max_abs - padding, max_abs + padding],
        ),
        yaxis=dict(title=None, automargin=True),
        showlegend=False,
        height=max(320, 34 * len(ordered_labels)),
    )
    return fig


def global_importance_bar(labels: list[str], mean_abs_shap: np.ndarray, top_n: int = 12) -> go.Figure:
    order = np.argsort(mean_abs_shap)[-top_n:]
    ordered_labels = [labels[i] for i in order]
    ordered_values = [mean_abs_shap[i] for i in order]

    fig = go.Figure(
        go.Bar(
            x=ordered_values,
            y=ordered_labels,
            orientation="h",
            marker_color=BLUE,
            text=[f"{v:.3f}" for v in ordered_values],
            textposition="outside",
        )
    )
    max_val = max(ordered_values) if ordered_values else 1.0
    _apply_base_layout(
        fig,
        title="Rata-rata |Nilai SHAP| per Fitur (Kepentingan Global)",
        xaxis=dict(
            title="Rata-rata |SHAP value|",
            gridcolor=GRIDLINE,
            zeroline=False,
            range=[0, max_val * 1.2],
        ),
        yaxis=dict(title=None, automargin=True),
        showlegend=False,
        height=max(360, 32 * len(ordered_labels)),
    )
    return fig


def confusion_matrix_heatmap(cm: list[list[int]], class_labels: list[str]) -> go.Figure:
    cm_arr = np.array(cm)
    fig = go.Figure(
        go.Heatmap(
            z=cm_arr,
            x=[f"Prediksi: {c}" for c in class_labels],
            y=[f"Aktual: {c}" for c in class_labels],
            colorscale=[[0, SURFACE], [1, BLUE]],
            text=cm_arr,
            texttemplate="%{text}",
            textfont=dict(size=16, color=TEXT_PRIMARY),
            showscale=False,
        )
    )
    _apply_base_layout(
        fig,
        title="Confusion Matrix (Data Uji)",
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        height=380,
    )
    return fig


def probability_bar(probability: float) -> go.Figure:
    if probability < 0.35:
        color = GOOD
    elif probability < 0.65:
        color = WARNING
    else:
        color = CRITICAL

    fig = go.Figure(
        go.Bar(
            x=[probability * 100],
            y=["Probabilitas Terlambat"],
            orientation="h",
            marker_color=color,
            text=[f"{probability * 100:.1f}%"],
            textposition="outside",
            width=[0.5],
        )
    )
    fig.add_vline(x=35, line_dash="dot", line_color=MUTED, annotation_text="35%")
    fig.add_vline(x=65, line_dash="dot", line_color=MUTED, annotation_text="65%")
    _apply_base_layout(
        fig,
        xaxis=dict(title="Probabilitas Keterlambatan (%)", range=[0, 100], gridcolor=GRIDLINE),
        yaxis=dict(title=None),
        showlegend=False,
        height=180,
        margin=dict(l=10, r=20, t=10, b=30),
    )
    return fig


def feature_distribution(df: pd.DataFrame, feature: str, target_col: str) -> go.Figure:
    fig = go.Figure()
    status_colors = {0: GOOD, 1: CRITICAL}
    status_names = {0: "Tepat Waktu", 1: "Terlambat"}
    for value in sorted(df[target_col].unique()):
        subset = df[df[target_col] == value][feature]
        if pd.api.types.is_numeric_dtype(df[feature]):
            fig.add_trace(
                go.Histogram(
                    x=subset,
                    name=status_names.get(value, str(value)),
                    marker_color=status_colors.get(value, MUTED),
                    opacity=0.75,
                    nbinsx=25,
                )
            )
        else:
            counts = subset.value_counts()
            fig.add_trace(
                go.Bar(
                    x=counts.index,
                    y=counts.values,
                    name=status_names.get(value, str(value)),
                    marker_color=status_colors.get(value, MUTED),
                )
            )
    barmode = "overlay" if pd.api.types.is_numeric_dtype(df[feature]) else "group"
    _apply_base_layout(
        fig,
        title=f"Distribusi {feature} berdasarkan Status Proyek",
        barmode=barmode,
        xaxis=dict(title=feature, gridcolor=GRIDLINE),
        yaxis=dict(title="Jumlah Proyek", gridcolor=GRIDLINE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=360,
    )
    return fig


def category_share_pie(df: pd.DataFrame, category_col: str) -> go.Figure:
    counts = df[category_col].value_counts()

    # Pie hanya terbaca sampai ~8 kategori (batas palet kategorikal yang
    # tervalidasi) - untuk kategori lebih banyak, bar horizontal top-N +
    # "Lainnya" jauh lebih terbaca daripada pie dengan puluhan irisan tipis.
    max_pie_slices = 8
    if len(counts) > max_pie_slices:
        top = counts.iloc[: max_pie_slices - 1]
        other_total = counts.iloc[max_pie_slices - 1 :].sum()
        ordered = pd.concat([top, pd.Series({"Lainnya": other_total})]).sort_values()
        fig = go.Figure(
            go.Bar(
                x=ordered.values,
                y=ordered.index,
                orientation="h",
                marker_color=BLUE,
                text=ordered.values,
                textposition="outside",
            )
        )
        _apply_base_layout(
            fig,
            title=f"Distribusi {category_col} (top {max_pie_slices - 1} + Lainnya)",
            xaxis=dict(title="Jumlah", gridcolor=GRIDLINE, zeroline=False, range=[0, ordered.max() * 1.2]),
            yaxis=dict(title=None, automargin=True),
            showlegend=False,
            height=max(360, 28 * len(ordered)),
        )
        return fig

    colors = (CATEGORICAL_ORDER * (len(counts) // len(CATEGORICAL_ORDER) + 1))[: len(counts)]
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            marker_colors=colors,
            hole=0.45,
            textinfo="label+percent",
        )
    )
    _apply_base_layout(fig, title=f"Distribusi {category_col}", height=360)
    return fig
