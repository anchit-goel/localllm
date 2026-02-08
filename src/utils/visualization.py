"""
Visualization utilities for security analysis.
"""

import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class Visualizer:
    """Visualization utilities for analysis results."""
    
    def __init__(self):
        self.colors = {
            'safe': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'info': '#3498db',
        }
    
    def plot_spectrogram(
        self,
        mel_spectrogram_db: np.ndarray,
        title: str = "Mel Spectrogram",
        highlight_regions: Optional[List[Dict]] = None,
    ):
        """Plot mel spectrogram with optional highlighting."""
        try:
            import matplotlib.pyplot as plt
            import librosa.display
            
            fig, ax = plt.subplots(figsize=(12, 4))
            img = librosa.display.specshow(
                mel_spectrogram_db,
                x_axis='time',
                y_axis='mel',
                ax=ax
            )
            ax.set_title(title)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            
            # Highlight suspicious regions
            if highlight_regions:
                for region in highlight_regions:
                    ax.axvspan(
                        region['start'], region['end'],
                        alpha=0.3, color='red',
                        label=region.get('label', '')
                    )
            
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Error plotting spectrogram: {e}")
            return None
    
    def plot_risk_gauge(
        self,
        risk_score: float,
        title: str = "Risk Score",
    ):
        """Plot risk score as a gauge chart."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Wedge
            
            fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'aspect': 'equal'})
            
            # Background arc
            colors = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c']
            for i, color in enumerate(colors):
                theta1 = 180 - i * 45
                theta2 = 180 - (i + 1) * 45
                wedge = Wedge((0.5, 0), 0.4, theta2, theta1, width=0.15, color=color)
                ax.add_patch(wedge)
            
            # Needle
            angle = 180 - risk_score * 180
            needle_x = 0.5 + 0.35 * np.cos(np.radians(angle))
            needle_y = 0.35 * np.sin(np.radians(angle))
            ax.plot([0.5, needle_x], [0, needle_y], color='black', linewidth=3)
            ax.plot(0.5, 0, 'ko', markersize=10)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.1, 0.5)
            ax.axis('off')
            ax.set_title(f"{title}: {risk_score:.1%}")
            
            return fig
        except Exception as e:
            logger.error(f"Error plotting gauge: {e}")
            return None
    
    def plot_component_scores(
        self,
        scores: Dict[str, float],
        title: str = "Component Scores",
    ):
        """Plot component scores as horizontal bar chart."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            labels = list(scores.keys())
            values = list(scores.values())
            colors = [self.colors['danger'] if v > 0.5 else self.colors['safe'] for v in values]
            
            y_pos = np.arange(len(labels))
            bars = ax.barh(y_pos, values, color=colors, alpha=0.8)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Score')
            ax.set_title(title)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(value + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{value:.1%}', va='center')
            
            # Add threshold line
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Error plotting scores: {e}")
            return None
    
    def plot_feature_importance(
        self,
        importance: Dict[str, float],
        top_n: int = 15,
        title: str = "Feature Importance",
    ):
        """Plot feature importance scores."""
        try:
            import matplotlib.pyplot as plt
            
            # Sort and get top N
            sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            labels = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, values, color=self.colors['info'], alpha=0.8)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Importance')
            ax.set_title(title)
            ax.invert_yaxis()
            
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Error plotting importance: {e}")
            return None
    
    def create_annotated_spectrogram(
        self,
        mel_spectrogram_db: np.ndarray,
        anomaly_regions: List[Dict],
    ) -> np.ndarray:
        """Create spectrogram image with highlighted anomaly regions."""
        try:
            import cv2
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
            
            # Normalize spectrogram to 0-255
            norm = Normalize(vmin=mel_spectrogram_db.min(), vmax=mel_spectrogram_db.max())
            spec_normalized = (norm(mel_spectrogram_db) * 255).astype(np.uint8)
            
            # Apply colormap
            spec_colored = cv2.applyColorMap(spec_normalized, cv2.COLORMAP_MAGMA)
            
            # Highlight anomaly regions
            for region in anomaly_regions:
                start_x = region.get('start_frame', 0)
                end_x = region.get('end_frame', spec_colored.shape[1])
                cv2.rectangle(
                    spec_colored,
                    (start_x, 0),
                    (end_x, spec_colored.shape[0]),
                    (0, 0, 255),  # Red
                    2
                )
            
            return spec_colored
        except Exception as e:
            logger.error(f"Error creating annotated spectrogram: {e}")
            return mel_spectrogram_db
    
    def save_figure(self, fig, path: str, dpi: int = 150):
        """Save figure to file."""
        try:
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Figure saved to {path}")
        except Exception as e:
            logger.error(f"Error saving figure: {e}")
