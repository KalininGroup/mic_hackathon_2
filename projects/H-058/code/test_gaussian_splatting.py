"""
Test Gaussian Splatting Module
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vantascope.data.gaussian_splatting import create_gaussian_splatter, create_defect_generator
from vantascope.utils.logging import logger

def create_test_graphene_structure(size: int = 20) -> np.ndarray:
    """Create a test graphene structure with hexagonal lattice."""
    
    # Graphene lattice parameters
    a = 2.46  # Lattice constant in Angstroms
    
    coordinates = []
    
    # Create hexagonal lattice
    for i in range(size):
        for j in range(size):
            # Two atoms per unit cell
            x1 = i * a * 3/2
            y1 = j * a * np.sqrt(3)/2
            
            x2 = x1 + a
            y2 = y1
            
            if j % 2 == 1:
                x1 += a * 3/4
                x2 += a * 3/4
            
            coordinates.append([x1, y1, 0.0])
            coordinates.append([x2, y2, 0.0])
    
    coords = np.array(coordinates)
    
    # Center the structure
    coords[:, :2] -= coords[:, :2].mean(axis=0)
    
    return coords

def test_gaussian_splatting():
    """Test the Gaussian splatting functionality."""
    logger.info("🧪 Testing Gaussian Splatting Module")
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Create splatter
    splatter = create_gaussian_splatter()
    
    # Create synthetic graphene coordinates (hexagonal lattice)
    coords = create_test_graphene_structure()
    logger.info(f"   Created test structure with {len(coords)} atoms")
    
    # Convert to image
    image = splatter.coordinates_to_image(coords)
    logger.info(f"   Generated image shape: {image.shape}")
    logger.info(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Save test image
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='hot', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title('Synthetic STEM Image from DFT Coordinates')
    plt.xlabel('Pixels')
    plt.ylabel('Pixels')
    plt.tight_layout()
    plt.savefig('outputs/test_gaussian_splatting.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("   ✅ Test image saved to outputs/test_gaussian_splatting.png")
    
    # Test batch processing
    coords_list = [coords, coords * 1.1, coords * 0.9]  # 3 variations
    batch_images = splatter.batch_process(coords_list)
    logger.info(f"   Batch processing: {batch_images.shape}")
    
    # Test defect generation
    defect_gen = create_defect_generator()
    
    # Create vacancy defect
    defect_coords = defect_gen.create_vacancy_defect(coords, (0.0, 0.0), vacancy_size=2)
    defect_image = splatter.coordinates_to_image(defect_coords)
    
    logger.info(f"   Defect structure: {len(defect_coords)} atoms (removed {len(coords) - len(defect_coords)})")
    
    # Save defect comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(image, cmap='hot', origin='lower')
    ax1.set_title('Pristine Graphene')
    ax1.set_xlabel('Pixels')
    ax1.set_ylabel('Pixels')
    
    ax2.imshow(defect_image, cmap='hot', origin='lower')
    ax2.set_title('Graphene with Vacancy Defects')
    ax2.set_xlabel('Pixels')
    ax2.set_ylabel('Pixels')
    
    plt.tight_layout()
    plt.savefig('outputs/test_defect_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("   ✅ Defect comparison saved to outputs/test_defect_comparison.png")
    
    # Test Stone-Wales defect
    sw_coords = defect_gen.create_stone_wales_defect(coords, (0.0, 0.0))
    sw_image = splatter.coordinates_to_image(sw_coords)
    
    # Save Stone-Wales comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(image, cmap='hot', origin='lower')
    ax1.set_title('Pristine Graphene')
    
    ax2.imshow(sw_image, cmap='hot', origin='lower')
    ax2.set_title('Graphene with Stone-Wales Defect')
    
    plt.tight_layout()
    plt.savefig('outputs/test_stone_wales_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("   ✅ Stone-Wales comparison saved to outputs/test_stone_wales_comparison.png")
    
    logger.info("🎉 Gaussian Splatting test completed successfully!")
    
    return {
        'pristine_image': image,
        'defect_image': defect_image,
        'stone_wales_image': sw_image,
        'batch_images': batch_images
    }

if __name__ == "__main__":
    test_gaussian_splatting()
