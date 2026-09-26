"""
Base Alpha Class — Common Interface

Every alpha family must implement this interface.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class BaseAlpha:
    """Base class for all alpha strategies.
    
    Common interface:
    - generate_signal() -> float in [-1, +1]
    - score() -> float confidence
    - explain() -> dict reason_codes
    - required_features() -> list of feature names
    - parameters() -> dict of current parameters
    - validate() -> dict of validation results
    - set_parameters() -> update parameters
    - regime_compatibility() -> str regime name
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self._parameters = parameters or self.default_parameters()
    
    def default_parameters(self) -> Dict[str, Any]:
        """Return default parameters for this alpha family."""
        return {}
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set parameters. Validates before setting."""
        self._parameters = self.validate(parameters)
    
    def parameters(self) -> Dict[str, Any]:
        """Return current parameters."""
        return self._parameters
    
    def required_features(self) -> List[str]:
        """Return list of feature names required by this alpha."""
        return []
    
    def generate_signal(self, features: Dict[str, Any]) -> float:
        """Generate alpha signal in range [-1, +1].
        
        Args:
            features: dict of feature name -> value
            
        Returns:
            signal: float in [-1, +1]
            -1 = strong SELL, 0 = HOLD, +1 = strong BUY
        """
        raise NotImplementedError("Subclasses must implement generate_signal()")
    
    def score(self, features: Dict[str, Any]) -> float:
        """Return confidence score for the generated signal.
        
        Args:
            features: dict of feature name -> value
            
        Returns:
            confidence: float in [0, 1], where 1 = highly confident
        """
        raise NotImplementedError("Subclasses must implement score()")
    
    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Provide explanation for the signal with reason codes.
        
        Returns:
            dict with:
            - reason_codes: list of strings explaining why
            - expected_horizon: expected holding period
            - expected_return: expected return
            - expected_risk: expected risk
            - regime_compatibility: which regimes this signal works in
        """
        raise NotImplementedError("Subclasses must implement explain()")
    
    def validate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters before setting.
        
        Returns:
            validated parameters dict
            
        Raises:
            ValueError: if parameters are invalid
        """
        # Default: accept all parameters
        return parameters
    
    def regime_compatibility(self) -> str:
        """Return which market regime this alpha works best in."""
        return "any"
    
    def expected_horizon(self, features: Dict[str, Any]) -> str:
        """Return expected holding period for the signal."""
        return "medium"
    
    def expected_return(self, features: Dict[str, Any]) -> float:
        """Return expected return for the signal."""
        return 0.0
    
    def expected_risk(self, features: Dict[str, Any]) -> float:
        """Return expected risk for the signal."""
        return 0.0