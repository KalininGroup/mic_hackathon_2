import numpy as np

# Calculate RMSD for AFM y DMD using the first frame as reference
def calculate_rmsd(positions_ref, positions_target):
    diff = positions_ref - positions_target
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return rmsd

def compute_rmsd_series(frames, reference_frame):
    rmsd_series = []
    for i in range(len(frames)):
        max_val = np.max(frames[i])
        max_val_ref = np.max(reference_frame)
        positions_ref = np.column_stack(np.where(reference_frame >  0.1 * max_val_ref))  
        positions_target = np.column_stack(np.where(frames[i] >  0.1 * max_val))     
        min_size = min(len(positions_ref), len(positions_target))
        positions_ref, positions_target = positions_ref[:min_size], positions_target[:min_size]
        rmsd = calculate_rmsd(positions_ref, positions_target)
        rmsd_series.append(rmsd)
    return rmsd_series

def get_positions_from_frames(frames):
    """
    The positions (X,Y) are obtained and stores in a array for each frame, 
    asumming the intensities of the images as distributions of the positions coordinates
    """
    positions_per_frame = []
    for frame in frames:
        max_val = np.max(frame)
        y, x = np.where(frame > 0.1 * max_val)  
        positions = np.vstack([x, y]).T  
        positions_per_frame.append(positions)
    return positions_per_frame

# Function to calculate the mass centroid
def calculate_center_of_mass(positions):
    return np.mean(positions, axis=0)

# Función to calcualte RyG
def calculate_radius_of_gyration(positions):
    com = calculate_center_of_mass(positions)
    squared_distances = np.sum((positions - com) ** 2, axis=1)  
    rg = np.sqrt(np.mean(squared_distances))  
    return rg