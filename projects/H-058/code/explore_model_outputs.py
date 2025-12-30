import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import matplotlib.pyplot as plt
import numpy as np
from vantascope.models.integrated_model import create_vantascope_model_with_uncertainty
from vantascope.data.dft_graphene_loader import DFTGrapheneDataset, collate_dft_batch
from torch.utils.data import DataLoader

def explore_model_outputs():
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_vantascope_model_with_uncertainty()
    checkpoint = torch.load('models/vantascope_fixed_epoch_3.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load one sample
    test_dataset = DFTGrapheneDataset(
        data_path="data/test",
        split='test', 
        max_samples=10,
        grid_size=512
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_dft_batch)
    
    print("🔬 EXPLORING MODEL OUTPUTS")
    print("=" * 50)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 1: break  # Just one sample
            
            images = batch['image'].to(device)
            true_energy = batch['energy'].item()
            coords = batch['coordinates'][0].cpu().numpy()
            
            print(f"📊 INPUT DATA:")
            print(f"   Image shape: {images.shape}")
            print(f"   True energy: {true_energy:.1f} eV")
            print(f"   Coordinates shape: {coords.shape}")
            print(f"   Atom count: {len(coords)}")
            
            # Get model outputs
            outputs = model(images, return_graph=True)
            
            print(f"\n🧠 MODEL OUTPUTS:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.shape} | dtype: {value.dtype}")
                    if value.numel() == 1:
                        print(f"      Value: {value.item():.4f}")
                    elif len(value.shape) == 2 and value.shape[0] == 1:
                        print(f"      Sample values: {value[0][:5].cpu().numpy()}")
                else:
                    print(f"   {key}: {type(value)}")
            
            # Check disentangled latents
            if 'geometric_latent' in outputs:
                geo = outputs['geometric_latent'][0].cpu().numpy()
                topo = outputs['topological_latent'][0].cpu().numpy() 
                disorder = outputs['disorder_latent'][0].cpu().numpy()
                
                print(f"\n🔍 DISENTANGLED LATENTS:")
                print(f"   Geometric (16D): mean={geo.mean():.3f}, std={geo.std():.3f}")
                print(f"   Topological (32D): mean={topo.mean():.3f}, std={topo.std():.3f}")
                print(f"   Disorder (16D): mean={disorder.mean():.3f}, std={disorder.std():.3f}")
            
            # Check graph outputs
            if 'node_classifications' in outputs:
                node_class = outputs['node_classifications'][0].cpu().numpy()
                print(f"\n🕸️  GRAPH OUTPUTS:")
                print(f"   Node classifications shape: {node_class.shape}")
                print(f"   Class distribution: {np.bincount(np.argmax(node_class, axis=1))}")
            
            # Energy prediction
            pred_energy = outputs['energy_mean'].item()
            energy_error = abs(pred_energy - true_energy)
            print(f"\n⚡ ENERGY PREDICTION:")
            print(f"   True: {true_energy:.1f} eV")
            print(f"   Predicted: {pred_energy:.1f} eV") 
            print(f"   Error: {energy_error:.1f} eV ({100*energy_error/abs(true_energy):.1f}%)")
            
            # Reconstruction quality
            recon = outputs['reconstruction']
            recon_loss = torch.nn.functional.mse_loss(recon, images).item()
            print(f"\n🎨 RECONSTRUCTION:")
            print(f"   MSE Loss: {recon_loss:.6f}")
            print(f"   Input range: [{images.min().item():.3f}, {images.max().item():.3f}]")
            print(f"   Recon range: [{recon.min().item():.3f}, {recon.max().item():.3f}]")
            
            # Save visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(images[0, 0].cpu().numpy(), cmap='gray')
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(recon[0, 0].cpu().numpy(), cmap='gray')
            plt.title(f'Reconstruction (MSE: {recon_loss:.4f})')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            diff = torch.abs(images - recon)[0, 0].cpu().numpy()
            plt.imshow(diff, cmap='hot')
            plt.title('Difference Map')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('data/model_exploration.png', dpi=150, bbox_inches='tight')
            print(f"\n💾 Visualization saved to: data/model_exploration.png")
            
            return outputs

if __name__ == "__main__":
    outputs = explore_model_outputs()
