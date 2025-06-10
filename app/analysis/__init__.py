# Analysis Engine

from . import indicators
from . import dow_theory

# Import main components
from .indicators import (
    ATRIndicator, ZigZagIndicator, SwingDetector, MomentumAnalyzer,
    SwingPoint, SwingType, SwingDetectionMethod
)
from .dow_theory import (
    DowTheoryAnalyzer, TrendDirection, TrendStrength, 
    VolumeConfirmationAnalyzer, dow_theory_analyzer
)

__all__ = [
    # Submodules
    'indicators',
    'dow_theory',
    
    # Main classes
    'ATRIndicator',
    'ZigZagIndicator', 
    'SwingDetector',
    'MomentumAnalyzer',
    'DowTheoryAnalyzer',
    'VolumeConfirmationAnalyzer',
    
    # Enums and data structures
    'SwingPoint',
    'SwingType', 
    'SwingDetectionMethod',
    'TrendDirection',
    'TrendStrength',
    
    # Default instances
    'dow_theory_analyzer'
]