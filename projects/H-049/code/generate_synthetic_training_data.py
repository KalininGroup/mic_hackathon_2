# how do I make sure that the parameters like lattice_angle_deg and atom_tilt_deg are included in the seed here? So that I can keep track of them depending on the seed used?
# I want to randomize them and have the seed keep track. Just like how the vacancy and interstitial positions are randomized
# What does background do in this code?
# how does this code draw the bounding boxes?

# A twin boundary changes the angle of atoms along a row in the lattice
# I made it into a double twin boundary by making it change the angle for just one line
# Adjust my code so that it makes a triple twin boundary by changing the angle back and forth for 3 lines in a row

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter  # for Gaussian blurring
from skimage.draw import disk               # to draw circular atoms/interstitials
import random
from PIL import Image
import pandas as pd
import math # For math.ceil in the bounding box widths

def generate_stem_image_with_continuous_gb(

    # parameters to keep constant
    img_size=768,              # Size of the square image (pixels)
    line_spacing=20,           # Spacing between lattice lines
    atom_spacing=20,           # Spacing between atoms along lines
    dumbbell_separation=4.0,   # Distance between the two "lobes" of a dumbbell atom
    interstitial_r = 4,         # Radius of interstitial atoms
    dumbbell_radius=4.0,       # Radius of each dumbbell lobe
    background=1.0,            # Base background intensity
    gb_misorientation_deg=0,
    seed=None                  # Random seed for reproducibility

):

    # Set random seed if provided
    # this ensures all randomized parameters are consistent according to seed #
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    def _clip_int(val, max_size):
        """Clip float to valid pixel index and cast to int."""
        return int(np.clip(val, 0, max_size - 1))

    # Function to rotate a point around a given line point by a specified angle
    def _rotate_about_line(point, line_point, angle):
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle),  np.cos(angle)]]) # 2D rotation matrix
        return line_point + R @ (point - line_point)

    # parameters to randomize (and save based on seed)
    atom_tilt_deg=random.uniform(20,70)                      # Atom tilt
    # atom_tilt_deg=random.uniform(0,360) 
    lattice_angle_deg=random.uniform(0,10)                  # lattice rotation
    # lattice_angle_deg=random.uniform(0,360)
    gb_angle = np.deg2rad(lattice_angle_deg)# random.uniform(0,360))             # GB angle
    gb_height = random.uniform(0,1)                          # Relative vertical position of GB (0=bottom, 1=top)
    atom_intensity=random.uniform(2,2.5)                       # Brightness of atoms
    interstitial_r=random.uniform(2.5, 4.5)                  # Radius of interstitial atoms
    interstitial_intensity=random.uniform(0.75, 0.9)         # Brightness relative to normal atoms
    noise_level=random.uniform(0.01, 0.1)                    # Gaussian noise added to final image
    gauss_sigma=random.uniform(3.5,4)                          # Gaussian blur applied to final image
    n_vacancies=random.randint(0,8)                          # Number of missing atoms
    n_interstitials=random.randint(0,8)                      # Number of extra interstitial atoms
    twin_gb=random.randint(0,1)                        # Randomize whether it's a twin boundary (0) or double twin boundary (1)
    
    # print(atom_tilt_deg)
    if twin_gb==0:
        print()
        print('single twin')
        print('num interstitials', n_interstitials)
        print('num vacancies', n_vacancies)
        print('atom_tilt_deg', atom_tilt_deg)
        print('lattice_angle_deg', lattice_angle_deg)
    if twin_gb==1:
        print()
        print('double twin')
        print('num interstitials', n_interstitials)
        print('num vacancies', n_vacancies)
        print('atom_tilt_deg', atom_tilt_deg)
        print('lattice_angle_deg', lattice_angle_deg)
        
    # calculate angle at which to rotate dumbell atoms across gb
    atom_tilt_refl_deg=2*np.rad2deg(gb_angle)-atom_tilt_deg

    bounding_boxes = [] # to store bounding boxes of defects in format (class, x1, y1, x2, y2, x3, y3, x4, y4)
    classes = {
        'gb': 0,
        'vacancy': 1,
        'interstitial': 2
    }

    # Create empty image with uniform background intensity
    img = np.full((img_size, img_size), background, dtype=float)

    # Grain boundary parameters (slope m and intercept b)
    m = np.tan(gb_angle)
    b = img_size * gb_height

    # Add grain boundary bounding box keeping coords within image bounds
    half_width = np.sqrt(line_spacing**2 + atom_spacing**2) * 1.5
    # if twin_gb==1: # increase width if making it a double twin gb
    #     half_width = half_width*1.9
    
    top_left_y = _clip_int(b + half_width, img_size)
    top_left_x = _clip_int((top_left_y - (b + half_width)) / m, img_size)

    bottom_left_y = _clip_int(b - half_width, img_size)
    bottom_left_x = _clip_int((bottom_left_y - (b - half_width)) / m, img_size)
        
    top_right_y = _clip_int(m * img_size + b + half_width, img_size)
    top_right_x = _clip_int((top_right_y - (b + half_width)) / m, img_size)

    bottom_right_y = _clip_int(m * img_size + b - half_width, img_size)
    bottom_right_x = _clip_int((bottom_right_y - (b - half_width)) / m, img_size)
    
    # Append bounding box for grain boundary to bounding box list
    bounding_boxes.append(
        (classes['gb'],
         bottom_left_x, bottom_left_y,
         bottom_right_x, bottom_right_y,
         top_right_x, top_right_y,
         top_left_x, top_left_y
        )
    )

    # Lattice directions
    lattice_angle = np.deg2rad(lattice_angle_deg)           # Convert lattice rotation to radians
    line_dir = np.array([np.cos(lattice_angle), np.sin(lattice_angle)])  # Direction along lines
    perp_dir = np.array([-line_dir[1], line_dir[0]])       # Perpendicular direction

    center = np.array([img_size // 2, img_size // 2])      # Image center
    n_lines = img_size // line_spacing                      # Number of lattice lines to draw

    atoms = []  # list to store atom positions and tilts

    # Generate atom positions on lattice
    for i in range(-n_lines, n_lines):
        for j in range(-img_size // atom_spacing,
                       img_size // atom_spacing):

            # Compute ideal atom position on lattice
            pos = center + j * atom_spacing * line_dir + i * line_spacing * perp_dir
            x, y = pos

            # # Small random perturbation to make lattice less perfect
            # array_offset=2 # amount to randomize by
            # x = x + random.uniform(-array_offset, array_offset)
            # y = y + random.uniform(-array_offset, array_offset)

            # Skip atoms outside image bounds
            if not (0 <= x < img_size and 0 <= y < img_size):
                continue

            # Project atom onto grain boundary line
            xg = (x + m * (y - b)) / (1 + m**2)
            yg = m * xg + b
            gb_point = np.array([xg, yg])

            # Determine which side of GB the atom is on
            if twin_gb==0:
                if y > (m * x + b):
                    tilt = np.deg2rad(atom_tilt_deg) # dumbbell tilt  
                else:
                    tilt = np.deg2rad(atom_tilt_refl_deg) # this makes sure atoms tilt relative to lattice
                    
            if twin_gb==1:
                if y > (m * x + b) + (atom_spacing-8) or y < (m * x + b) - (atom_spacing-8): # (atom_spacing-1) is a buffer zone
                    tilt = np.deg2rad(atom_tilt_deg) # dumbbell tilt
                else:
                    tilt = np.deg2rad(atom_tilt_refl_deg) # this makes sure atoms tilt relative to lattice

            # Rotate atom around its projected GB point
            new_pos = _rotate_about_line(pos, gb_point, 0)

            # Add small random jitter
            array_offset=1
            new_pos[0] = new_pos[0] + random.uniform(-array_offset, array_offset)
            new_pos[1] = new_pos[1] + random.uniform(-array_offset, array_offset)

            # Store atom position and tilt
            atoms.append((new_pos, tilt))

    # Randomly select indices for vacancies (missing atoms)
    vacancy_indices = random.sample(range(len(atoms)), n_vacancies)
    vacancy_set = set(vacancy_indices)
    # vacancy_set are all the vacancy positions

    # Draw dumbbell atoms
    for idx, (pos, tilt) in enumerate(atoms):
        if idx in vacancy_set:
            # bounding_boxes.append(
            #     (classes['vacancy'],
            #      pos[0] - line_spacing, pos[1] - line_spacing,
            #      pos[0] + line_spacing, pos[1] - line_spacing,
            #      pos[0] + line_spacing, pos[1] + line_spacing,
            #      pos[0] - line_spacing, pos[1] + line_spacing)
            # )
            # print('vacancy position', pos)
            bounding_boxes.append(
                (classes['vacancy'],
                 _clip_int((pos[0] - line_spacing), img_size), _clip_int((pos[1] - line_spacing), img_size),
                 _clip_int((pos[0] + line_spacing), img_size), _clip_int((pos[1] - line_spacing), img_size),
                 _clip_int((pos[0] + line_spacing), img_size), _clip_int((pos[1] + line_spacing), img_size),
                 _clip_int((pos[0] - line_spacing), img_size), _clip_int((pos[1] + line_spacing), img_size))
            )
            continue # skip vacancies

        cx, cy = pos
        dx = dumbbell_separation * np.cos(tilt)
        dy = dumbbell_separation * np.sin(tilt)

        # Each dumbbell has two lobes, draw both
        for sx, sy in [(cx + dx, cy + dy), (cx - dx, cy - dy)]:
            x, y = int(sx), int(sy)
            if 0 <= x < img_size and 0 <= y < img_size:
                rr, cc = disk((y, x), dumbbell_radius, shape=img.shape)
                img[rr, cc] += atom_intensity * 0.5  # add brightness to image

    # Interstitial atoms (extra atoms at midpoints)
    interstitial_sites = []
    for pos, _ in atoms:
        mid = pos + 0.5 * atom_spacing * line_dir + 0.5 * line_spacing * perp_dir
        x, y = map(int, mid)
        if 0 <= x < img_size and 0 <= y < img_size:
            interstitial_sites.append((x, y))

    # Remove duplicates and randomly select a subset to be used as interstitial locations
    interstitial_sites = list(set(interstitial_sites))
    chosen = random.sample(interstitial_sites,
                           min(n_interstitials, len(interstitial_sites)))

    # chosen = coordinates for interstitial sites
    # Draw interstitial atoms at chosen coordinates
    box_width = int(math.ceil(line_spacing / 2))
    for x, y in chosen:
        # print('interstitial position:',x,y)
        # bounding_boxes.append(
        #     (classes['interstitial'],
        #      x - box_width, y - box_width,
        #      x + box_width, y - box_width,
        #      x + box_width, y + box_width,
        #      x - box_width, y + box_width)
        # )
        bounding_boxes.append((classes['interstitial'],
             _clip_int((x - box_width), img_size), _clip_int((y - box_width), img_size),
             _clip_int((x + box_width), img_size), _clip_int((y - box_width), img_size),
             _clip_int((x + box_width), img_size), _clip_int((y + box_width), img_size),
             _clip_int((x - box_width), img_size), _clip_int((y + box_width), img_size)))

        rr, cc = disk((y, x), radius=interstitial_r, shape=img.shape)
        img[rr, cc] += atom_intensity * interstitial_intensity

    # Apply Gaussian blur to simulate microscope resolution
    img = gaussian_filter(img, sigma=gauss_sigma)
    # Add Gaussian noise
    img += np.random.normal(0, noise_level, img.shape)

    # # Normalize image to [0,1] range
    # # I don't know if this actually works
    img -= img.min()
    img /= img.max()

    metadata = {
            "seed": seed,
            "lattice_angle_deg": lattice_angle_deg,
            "atom_tilt_deg": atom_tilt_deg,
            "gb_angle_deg": np.rad2deg(gb_angle),
            "gb_misorientation_deg": gb_misorientation_deg,
            "gb_height": gb_height,
            "n_vacancies": n_vacancies,
            "n_interstitials": n_interstitials,
            "img_size": img_size,
            "twin_gb":twin_gb,
        }

    return img, bounding_boxes, metadata

def rotate_flip_image_and_bboxes(image, bounding_boxes, rot, refl):
    # Rotate image
    rotated_img = image.rotate(rot, expand=True)
    # Flip image
    if refl == 'horizontal' or refl == 'both':
        rotated_img = rotated_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if refl == 'vertical' or refl == 'both':
        rotated_img = rotated_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    transformed_bboxes = []

    for box in bounding_boxes:
        cls = box[0]
        coords = np.array(box[1:]).reshape((4, 2))  # reshape to 4 (x,y) points

        # Rotate points around image center
        center = np.array(image.size) / 2
        angle_rad = np.deg2rad(-rot)
        R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                      [np.sin(angle_rad),  np.cos(angle_rad)]])
        rotated_coords = np.dot(coords - center, R.T) + center

        # Apply reflections
        if refl == 'horizontal':
            rotated_coords[:, 0] = image.size[0] - rotated_coords[:, 0]
        elif refl == 'vertical':
            rotated_coords[:, 1] = image.size[1] - rotated_coords[:, 1]
        elif refl == 'both':
            rotated_coords[:, 0] = image.size[0] - rotated_coords[:, 0]
            rotated_coords[:, 1] = image.size[1] - rotated_coords[:, 1]

        transformed_bboxes.append(
            (cls,
             rotated_coords[0,0], rotated_coords[0,1],
             rotated_coords[1,0], rotated_coords[1,1],
             rotated_coords[2,0], rotated_coords[2,1],
             rotated_coords[3,0], rotated_coords[3,1])
        )

    return rotated_img, transformed_bboxes

def yolo_format_bounding_boxes(bounding_boxes, img_size):
    """
    Normalize polygon pixel coordinates for YOLO-style OBB / segmentation.

    Input:
        bounding_boxes: array-like of shape (N, 9)
            [class, x1, y1, x2, y2, x3, y3, x4, y4]  (pixel coords)

    Output:
        boxes_norm: (N, 9)
            [class, x1, y1, x2, y2, x3, y3, x4, y4]  (normalized to [0,1])
    """

    def _reorder_polygon_clockwise_numpy(pts):
        """
        pts: (4, 2) array of polygon points
        returns: (4, 2) array ordered clockwise
        """
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1],
                            pts[:, 0] - center[0])
        order = np.argsort(-angles)  # clockwise
        return pts[order]

    boxes = np.asarray(bounding_boxes, dtype=float)

    # Split class + polygon
    classes = boxes[:, 0].astype(int)
    polys = boxes[:, 1:].reshape(-1, 4, 2)

    # ---- reorder each polygon clockwise ----
    for i in range(polys.shape[0]):
        polys[i] = _reorder_polygon_clockwise_numpy(polys[i])

    # ---- normalize ----
    w = img_size
    h = img_size

    polys[..., 0] /= w  # x coords
    polys[..., 1] /= h  # y coords

    polys = polys.reshape(-1, 8)

    # ---- reassemble ----
    boxes_norm = np.concatenate(
        [classes[:, None], polys],
        axis=1
    )

    return boxes_norm


# Path to save generated images
savepath = r"C:\Users\proks\OneDrive\Documents\GitHub\2025_Hackathon\generated_test_data_3".replace("\\", "/")

seed = 23  # seed for reproducibility

if __name__ == "__main__":
    # Generate fake STEM image
    img_sz = 768
    img, bounding_boxes, metadata = generate_stem_image_with_continuous_gb(
                                    seed=seed, img_size=img_sz)
    yolo_bounding_boxes = yolo_format_bounding_boxes(bounding_boxes, img_sz)


    # Plot and save image
    figure, axis = plt.subplots(figsize=(5,5), dpi=300)
    axis.imshow(img, cmap="gray")
    
    ##%%
    
    # show the bounding boxes color coded by class
    for box in bounding_boxes:
        cls = box[0]
        x = list(box[1::2])
        y = list(box[2::2])
        if cls == 0:
            color = 'red'  # grain boundary
        elif cls == 1:
            color = 'blue' # vacancy
        elif cls == 2:
            color = 'green' # interstitial
        axis.plot(
            x + [x[0]],
            y + [y[0]],
            color=color, linewidth=.5)
    axis.axis("off")
    # figure.savefig(savepath+'/STEM_' + str(seed) + '.jpeg', bbox_inches="tight", dpi=300)
    plt.show()

    image = Image.fromarray((img * 255).astype(np.uint8))
    # image.save(savepath+'/STEM_' + str(seed) + '.png')

    # # Save bounding boxes to text file for yolo input
    # np.savetxt(
    #                 savepath+'/STEM_' + str(seed) + '_bboxes.txt',
    #                 yolo_bounding_boxes,
    #                 fmt='%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f'
    #             )

    # save metadata to CSV
    meta_df = pd.DataFrame([metadata])
    # meta_df.to_csv(savepath + f"/STEM_{seed}_metadata.csv", index=False)

    rotations = [0, 90, 180, 270]  # degrees
    reflections = ['none', 'horizontal', 'vertical', 'both']
#%%
    # --- Apply transformations and save images ---
    for rot in rotations:
        for refl in reflections:
            if rot == 0 and refl == 'none':
                continue  # skip original image
            transformed_img, transformed_bboxes = rotate_flip_image_and_bboxes(image, bounding_boxes, rot, refl)
            # import pdb; pdb.set_trace()
            yolo_transformed_bboxes = yolo_format_bounding_boxes(bounding_boxes, img_sz)

            # Save the transformed image
            transformed_img.save(savepath+'/STEM_' + str(seed) + f'_rot{rot}_refl{refl}.png')

            # Save transformed bounding boxes to txt
            np.savetxt(
                    savepath + f'/STEM_{seed}_rot{rot}_refl{refl}_bboxes.txt',
                    yolo_transformed_bboxes,
                    fmt='%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f'
                )

            # plot and save transformed image with bounding boxes
            figure, axis = plt.subplots(figsize=(5,5), dpi=300)
            axis.imshow(transformed_img, cmap="gray")
            for box in transformed_bboxes:
                cls = box[0]
                x = list(box[1::2])
                y = list(box[2::2])
                if cls == 0:
                    color = 'red'  # grain boundary
                elif cls == 1:
                    color = 'blue' # vacancy
                elif cls == 2:
                    color = 'green' # interstitial
                axis.plot(
                    x + [x[0]],
                    y + [y[0]],
                    color=color, linewidth=.5)
            axis.axis("off")
            figure.savefig(savepath+'/STEM_' + str(seed) + f'_rot{rot}_refl{refl}.jpeg', bbox_inches="tight", dpi=300)
