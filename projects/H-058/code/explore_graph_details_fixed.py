import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from vantascope.models.integrated_model import create_vantascope_model_with_uncertainty
from vantascope.data.dft_graphene_loader import DFTGrapheneDataset, collate_dft_batch
from torch.utils.data import DataLoader

def explore_graph_details():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_vantascope_model_with_uncertainty()
    checkpoint = torch.load('models/vantascope_fixed_epoch_3.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    test_dataset = DFTGrapheneDataset(data_path="data/test", split='test', max_samples=5, grid_size=512)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_dft_batch)
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 1: break
            
            images = batch['image'].to(device)
            outputs = model(images, return_graph=True)
            
            print("🕸️  GRAPH & DEFECT ANALYSIS")
            print("=" * 50)
            
            # Patch defect analysis
            defect_probs = outputs['patch_defect_probs'][0].cpu().numpy()
            defect_classes = np.argmax(defect_probs, axis=1)
            
            print(f"📊 PATCH DEFECT CLASSIFICATION:")
            print(f"   Total patches: {defect_probs.shape[0]}")
            print(f"   Defect classes: {defect_probs.shape[1]}")
            
            class_counts = np.bincount(defect_classes, minlength=4)
            class_names = ['Perfect', 'Vacancy', 'Interstitial', 'Grain Boundary']
            
            for i, (name, count) in enumerate(zip(class_names, class_counts)):
                percentage = 100 * count / len(defect_classes)
                avg_conf = defect_probs[defect_classes == i, i].mean() if count > 0 else 0
                print(f"   {name}: {count} patches ({percentage:.1f}%) | Avg confidence: {avg_conf:.3f}")
            
            # Graph outputs exploration
            graph_outputs = outputs['graph_outputs']
            print(f"\n�� GRAPH OUTPUTS STRUCTURE:")
            for key, value in graph_outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.shape}")
                else:
                    print(f"   {key}: {type(value)}")
            
            # Properties exploration
            properties = outputs['properties']
            print(f"\n🔬 PROPERTIES STRUCTURE:")
            for key, value in properties.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        print(f"   {key}: {value.shape} | Value: {value.item():.4f}")
                    else:
                        print(f"   {key}: {value.shape} | Values: {value.flatten()[:5].cpu().numpy()}")
                else:
                    print(f"   {key}: {type(value)} | {value}")
            
            print(f"\n🎯 ATTENTION MAP ANALYSIS:")
            attention = outputs['attention_maps'][0, 0].cpu().numpy()
            print(f"   Shape: {attention.shape}")
            print(f"   Range: [{attention.min():.4f}, {attention.max():.4f}]")
            print(f"   Mean: {attention.mean():.4f}")
            
            return outputs

if __name__ == "__main__":
    explore_graph_details()
