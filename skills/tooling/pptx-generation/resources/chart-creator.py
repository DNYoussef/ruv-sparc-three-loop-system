#!/usr/bin/env python3
"""
Chart Creator - Data visualization for PowerPoint slides
Generates clean, accessible charts with matplotlib/plotly
Optimized for presentation context (high contrast, large fonts)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import base64
from io import BytesIO


class ChartType(Enum):
    """Supported chart types optimized for presentations"""
    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"


@dataclass
class ChartConfig:
    """Chart styling configuration for presentation context"""
    # Colors
    color_palette: List[str]
    background_color: str = '#FFFFFF'
    text_color: str = '#1F2937'
    grid_color: str = '#E5E7EB'

    # Typography
    title_font_size: int = 20
    label_font_size: int = 14
    tick_font_size: int = 12
    font_family: str = 'Arial'

    # Layout
    figure_width: float = 10.0  # inches
    figure_height: float = 6.0
    dpi: int = 150  # High DPI for crisp slides

    # Accessibility
    min_contrast_ratio: float = 4.5
    include_data_labels: bool = True

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration meets accessibility standards"""
        errors = []

        # Validate font sizes
        if self.tick_font_size < 12:
            errors.append("Tick font size below 12pt (readability issue)")
        if self.title_font_size < 18:
            errors.append("Title font size below 18pt (presentation minimum)")

        # Validate colors (would need full contrast check implementation)
        if len(self.color_palette) < 2:
            errors.append("Color palette needs at least 2 colors")

        return (len(errors) == 0, errors)


class ChartCreator:
    """
    Generate publication-quality charts for PowerPoint presentations
    Enforces accessibility and readability standards
    """

    def __init__(self, config: ChartConfig):
        self.config = config
        self._validate_config()

        # Set global matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({
            'font.family': self.config.font_family,
            'font.size': self.config.label_font_size,
            'axes.labelsize': self.config.label_font_size,
            'axes.titlesize': self.config.title_font_size,
            'xtick.labelsize': self.config.tick_font_size,
            'ytick.labelsize': self.config.tick_font_size,
            'legend.fontsize': self.config.label_font_size,
            'figure.titlesize': self.config.title_font_size,
            'figure.facecolor': self.config.background_color,
            'axes.facecolor': self.config.background_color,
            'axes.edgecolor': self.config.text_color,
            'axes.labelcolor': self.config.text_color,
            'text.color': self.config.text_color,
            'xtick.color': self.config.text_color,
            'ytick.color': self.config.text_color,
            'grid.color': self.config.grid_color,
            'grid.alpha': 0.3
        })

    def _validate_config(self):
        """Ensure configuration is valid before chart generation"""
        valid, errors = self.config.validate()
        if not valid:
            raise ValueError(f"Invalid chart configuration: {'; '.join(errors)}")

    def create_bar_chart(self,
                        data: Dict[str, float],
                        title: str,
                        x_label: str = "",
                        y_label: str = "",
                        horizontal: bool = False) -> Figure:
        """
        Create clean bar chart with data labels

        Args:
            data: {category: value} mapping
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            horizontal: True for horizontal bars

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height),
                              dpi=self.config.dpi)

        categories = list(data.keys())
        values = list(data.values())
        colors = self._get_colors(len(categories))

        if horizontal:
            bars = ax.barh(categories, values, color=colors, edgecolor=self.config.text_color, linewidth=1.5)

            if self.config.include_data_labels:
                for i, (bar, value) in enumerate(zip(bars, values)):
                    ax.text(value, i, f' {value:,.0f}',
                           va='center', ha='left',
                           fontsize=self.config.label_font_size,
                           fontweight='bold')

            ax.set_xlabel(y_label, fontweight='bold')
            ax.set_ylabel(x_label, fontweight='bold')
        else:
            bars = ax.bar(categories, values, color=colors, edgecolor=self.config.text_color, linewidth=1.5)

            if self.config.include_data_labels:
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:,.0f}',
                           ha='center', va='bottom',
                           fontsize=self.config.label_font_size,
                           fontweight='bold')

            ax.set_xlabel(x_label, fontweight='bold')
            ax.set_ylabel(y_label, fontweight='bold')
            plt.xticks(rotation=45, ha='right')

        ax.set_title(title, fontweight='bold', pad=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y' if not horizontal else 'x', alpha=0.3)

        plt.tight_layout()
        return fig

    def create_line_chart(self,
                         series_data: Dict[str, List[Tuple[float, float]]],
                         title: str,
                         x_label: str = "",
                         y_label: str = "") -> Figure:
        """
        Create multi-series line chart with markers

        Args:
            series_data: {series_name: [(x, y), ...]} mapping
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height),
                              dpi=self.config.dpi)

        colors = self._get_colors(len(series_data))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

        for idx, (name, data) in enumerate(series_data.items()):
            x_vals, y_vals = zip(*data)
            ax.plot(x_vals, y_vals,
                   label=name,
                   color=colors[idx],
                   marker=markers[idx % len(markers)],
                   markersize=8,
                   linewidth=3,
                   markeredgecolor=self.config.background_color,
                   markeredgewidth=2)

        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel(x_label, fontweight='bold')
        ax.set_ylabel(y_label, fontweight='bold')
        ax.legend(frameon=True, fancybox=False, shadow=False,
                 framealpha=0.9, edgecolor=self.config.text_color)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_pie_chart(self,
                        data: Dict[str, float],
                        title: str,
                        show_percentages: bool = True) -> Figure:
        """
        Create clean pie chart with clear labels

        Args:
            data: {category: value} mapping
            title: Chart title
            show_percentages: Include percentage labels

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height),
                              dpi=self.config.dpi)

        labels = list(data.keys())
        values = list(data.values())
        colors = self._get_colors(len(labels))

        # Create pie with white edge for separation
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%' if show_percentages else None,
            startangle=90,
            wedgeprops=dict(edgecolor=self.config.background_color, linewidth=3),
            textprops=dict(color=self.config.text_color, fontsize=self.config.label_font_size)
        )

        # Style percentage labels
        if show_percentages:
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(self.config.label_font_size)
                autotext.set_fontweight('bold')

        ax.set_title(title, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig

    def create_scatter_plot(self,
                           data: List[Tuple[float, float]],
                           title: str,
                           x_label: str = "",
                           y_label: str = "",
                           trend_line: bool = True) -> Figure:
        """
        Create scatter plot with optional trend line

        Args:
            data: [(x, y), ...] points
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            trend_line: Include linear regression line

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height),
                              dpi=self.config.dpi)

        x_vals, y_vals = zip(*data)

        # Scatter points
        ax.scatter(x_vals, y_vals,
                  s=100,
                  alpha=0.7,
                  color=self.config.color_palette[0],
                  edgecolors=self.config.text_color,
                  linewidth=1.5)

        # Trend line
        if trend_line:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            ax.plot(x_vals, p(x_vals),
                   "--",
                   color=self.config.color_palette[1] if len(self.config.color_palette) > 1 else self.config.text_color,
                   linewidth=2,
                   label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
            ax.legend()

        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel(x_label, fontweight='bold')
        ax.set_ylabel(y_label, fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _get_colors(self, count: int) -> List[str]:
        """
        Get color palette subset
        Cycles through palette if count exceeds available colors
        """
        if count <= len(self.config.color_palette):
            return self.config.color_palette[:count]

        # Repeat palette if needed
        multiplier = (count // len(self.config.color_palette)) + 1
        extended = self.config.color_palette * multiplier
        return extended[:count]

    def save_chart(self, fig: Figure, output_path: str, format: str = 'png'):
        """
        Save chart to file

        Args:
            fig: matplotlib Figure object
            output_path: Output file path
            format: Image format (png, svg, pdf)
        """
        fig.savefig(output_path,
                   format=format,
                   dpi=self.config.dpi,
                   bbox_inches='tight',
                   facecolor=self.config.background_color,
                   edgecolor='none')
        return output_path

    def to_base64(self, fig: Figure, format: str = 'png') -> str:
        """
        Convert chart to base64 string for HTML embedding

        Args:
            fig: matplotlib Figure object
            format: Image format

        Returns:
            Base64-encoded image string
        """
        buffer = BytesIO()
        fig.savefig(buffer,
                   format=format,
                   dpi=self.config.dpi,
                   bbox_inches='tight',
                   facecolor=self.config.background_color)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        return f"data:image/{format};base64,{img_base64}"


if __name__ == "__main__":
    # Example usage
    config = ChartConfig(
        color_palette=['#1E3A8A', '#3B82F6', '#60A5FA', '#93C5FD'],
        title_font_size=20,
        label_font_size=14
    )

    creator = ChartCreator(config)

    # Bar chart example
    revenue_data = {
        'Q1': 2.5,
        'Q2': 3.2,
        'Q3': 3.8,
        'Q4': 4.1
    }

    fig = creator.create_bar_chart(
        revenue_data,
        'Quarterly Revenue Growth',
        x_label='Quarter',
        y_label='Revenue ($M)'
    )

    creator.save_chart(fig, 'revenue_chart.png')
    print("Generated revenue_chart.png")
