import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from vantascope.models.integrated_model import create_vantascope_model_with_uncertainty
from vantascope.data.dft_graphene_loader import DFTGrapheneDataset, collate_dft_batch
from torch.utils.data import DataLoader
import numpy as np

def evaluate_model(model_path):
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_vantascope_model_with_uncertainty()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = DFTGrapheneDataset(
        data_path="data/test",
        split='test', 
        max_samples=1000,  # Quick test
        grid_size=512
    )
    
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_dft_batch)
    
    energy_errors = []
    recon_errors = []
    
    print(f"🔬 Evaluating model: {model_path}")
    print(f"   Test samples: {len(test_dataset)}")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 10: break  # Quick test
            
            images = batch['image'].to(device)
            true_energy = batch['energy'].to(device)
            
            outputs = model(images, return_graph=True)
            
            # Energy error
            pred_energy = outputs['energy_mean']
            energy_error = torch.abs(pred_energy - true_energy).cpu().numpy()
            energy_errors.extend(energy_error)
            
            # Reconstruction error  
            recon_error = torch.nn.functional.mse_loss(outputs['reconstruction'], images).cpu().item()
            recon_errors.append(recon_error)
            
            if i == 0:
                print(f"   Sample 1 - True: {true_energy[0].item():.1f} eV, Pred: {pred_energy[0].item():.1f} eV")
    
    print(f"📊 Results:")
    print(f"   Energy MAE: {np.mean(energy_errors):.2f} eV")
    print(f"   Energy STD: {np.std(energy_errors):.2f} eV") 
    print(f"   Reconstruction MSE: {np.mean(recon_errors):.6f}")

if __name__ == "__main__":
    evaluate_model('models/vantascope_fixed_epoch_3.pth')
