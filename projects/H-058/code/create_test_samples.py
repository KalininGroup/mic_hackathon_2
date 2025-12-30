import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Create directories
os.makedirs('data/examples', exist_ok=True)

def create_sample(pattern_type, filename, size=512):
    """Create synthetic microscopy sample and save as image file."""
    np.random.seed(hash(pattern_type) % 1000)
    x, y = np.meshgrid(np.linspace(-2, 2, size), np.linspace(-2, 2, size))
    
    if pattern_type == 'high_quality':
        # Clean honeycomb
        pattern = np.sin(3*x) * np.cos(3*y) + 0.1 * np.random.random((size, size))
    elif pattern_type == 'polycrystalline':
        # Multiple domains
        pattern = (np.sin(3*x) * np.cos(3*y) + 
                  0.5 * np.sin(5*x + 1) * np.cos(2*y + 1) + 
                  0.2 * np.random.random((size, size)))
    elif pattern_type == 'defective':
        # High defect density
        pattern = (0.3 * np.sin(3*x) * np.cos(3*y) + 
                  0.7 * np.random.random((size, size)))
    elif pattern_type == 'nanotube':
        # Tubular structure
        pattern = np.sin(8*x) + 0.1 * np.random.random((size, size))
    else:  # damaged
        # Very noisy
        pattern = 0.2 * np.sin(2*x) * np.cos(2*y) + 0.8 * np.random.random((size, size))
    
    # Normalize to 0-255 and save
    pattern_norm = ((pattern - pattern.min()) / (pattern.max() - pattern.min()) * 255).astype(np.uint8)
    img = Image.fromarray(pattern_norm, mode='L')
    img.save(filename)
    print(f"Created: {filename}")

# Generate all test samples
samples = [
    ('high_quality', 'data/examples/high_quality_graphene.tif'),
    ('polycrystalline', 'data/examples/polycrystalline_graphene.tif'),
    ('defective', 'data/examples/defective_graphene.tif'),
    ('nanotube', 'data/examples/pristine_carbon_nanotube.tif'),
    ('damaged', 'data/examples/damaged_graphene_oxide.tif')
]

for pattern_type, filepath in samples:
    create_sample(pattern_type, filepath)

print("\nAll test samples created!")
print("File paths:")
for _, filepath in samples:
    print(f"  {filepath}")
