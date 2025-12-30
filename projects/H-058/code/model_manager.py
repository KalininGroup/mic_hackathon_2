"""
VantaScope Model Manager - Professional AI inference pipeline
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from vantascope.models.integrated_model import create_vantascope_model_with_uncertainty

class VantaScopeModelManager:
    """Professional model management for real-time inference."""
    
    def __init__(self, model_path="models/vantascope_fixed_epoch_3.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path
        self.is_model_loaded = False
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self):
        """Load the trained VantaScope model."""
        try:
            # Create model architecture
            self.model = create_vantascope_model_with_uncertainty()
            
            # Load trained weights
            if Path(self.model_path).exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Move to device and set eval mode
                self.model = self.model.to(self.device)
                self.model.eval()
                
                self.is_model_loaded = True
                
                # Warm-up inference
                self._warmup_model()
                
            else:
                raise FileNotFoundError(f"Model not found: {self.model_path}")
                
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.is_model_loaded = False
            raise
    
    def _warmup_model(self):
        """Warm up model with dummy input for faster first inference."""
        try:
            dummy_input = torch.randn(1, 1, 512, 512).to(self.device)
            with torch.no_grad():
                _ = self.model(dummy_input, return_graph=True)
        except Exception as e:
            print(f"Model warmup failed: {e}")
    
    def is_loaded(self):
        """Check if model is successfully loaded."""
        return self.is_model_loaded
    
    def get_param_count(self):
        """Get total number of model parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def analyze_sample(self, input_tensor):
        """
        Run complete analysis on input sample.
        
        Args:
            input_tensor: torch.Tensor of shape (1, 1, 512, 512)
            
        Returns:
            dict: Complete analysis results
        """
        if not self.is_model_loaded:
            raise RuntimeError("Model not loaded!")
        
        try:
            # Ensure tensor is on correct device
            if isinstance(input_tensor, np.ndarray):
                input_tensor = torch.from_numpy(input_tensor).float()
            
            if len(input_tensor.shape) == 2:
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            elif len(input_tensor.shape) == 3:
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dim
                
            input_tensor = input_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor, return_graph=True)
            
            # Post-process results for UI
            processed_results = self._post_process_results(outputs)
            
            return processed_results
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            raise
    
    def _post_process_results(self, raw_outputs):
        """Post-process raw model outputs for UI consumption."""
        results = {}
        
        # Move tensors to CPU for UI processing
        for key, value in raw_outputs.items():
            if isinstance(value, torch.Tensor):
                results[key] = value.cpu()
            else:
                results[key] = value
        
        # Add derived metrics
        results['analysis_summary'] = self._create_analysis_summary(results)
        
        return results
    
    def _create_analysis_summary(self, results):
        """Create high-level analysis summary."""
        summary = {}
        
        try:
            # Energy analysis
            energy_mean = results['energy_mean'].item()
            energy_std = results['energy_std'].item()
            summary['energy_confidence'] = 'High' if energy_std < 50 else 'Medium' if energy_std < 100 else 'Low'
            
            # Defect analysis
            defect_probs = results['patch_defect_probs'][0].numpy()
            defect_classes = np.argmax(defect_probs, axis=1)
            defect_distribution = np.bincount(defect_classes, minlength=4) / len(defect_classes)
            
            summary['primary_defect_type'] = ['Perfect', 'Vacancy', 'Interstitial', 'Grain Boundary'][np.argmax(defect_distribution)]
            summary['defect_confidence'] = np.max(defect_probs, axis=1).mean()
            
            # Crystallinity assessment
            if 'properties' in results and 'crystallinity' in results['properties']:
                crystallinity = results['properties']['crystallinity'].item()
                if crystallinity > 0.8:
                    summary['quality_assessment'] = 'Excellent'
                elif crystallinity > 0.6:
                    summary['quality_assessment'] = 'Good'
                elif crystallinity > 0.4:
                    summary['quality_assessment'] = 'Fair'
                else:
                    summary['quality_assessment'] = 'Poor'
            else:
                summary['quality_assessment'] = 'Unknown'
            
        except Exception as e:
            print(f"Summary creation failed: {e}")
            summary['error'] = str(e)
        
        return summary
