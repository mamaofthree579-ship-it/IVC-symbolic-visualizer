"""
modules package
===============

This directory contains modular components for the IVC Symbolic Visualizer.
Each module provides specific functionality — analytics, visualization,
and data management — to support the main Streamlit app.

Submodules:
-----------
analytics : Functions for mathematical, linguistic, and symbolic analysis.
visuals   : Functions for rendering glyphs, vector maps, and symbol clusters.
data_loader : Helpers for reading and transforming structured datasets.

Usage:
------
from modules import analytics, visuals
or
from modules.analytics import find_resonant_clusters
"""

from . import analytics
from . import visuals

__all__ = [
    "analytics",
    "visuals"
]
