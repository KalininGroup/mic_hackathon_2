import numpy as np
from generate_synthetic_training_data import generate_stem_image_with_continuous_gb, rotate_flip_image_and_bboxes, yolo_format_bounding_boxes
from pathlib import Path
from PIL import Image
from tqdm import tqdm

training_data_dir = Path("./data/dataset_4")
img_size = 768
num_seeds = 80
rotations = [0, 90, 180, 270]
reflections = ['none', 'horizontal', 'vertical', 'both']

if __name__ == "__main__":
    training_data_dir.mkdir(exist_ok=True)

    images_dir = training_data_dir / "images"
    images_dir.mkdir(exist_ok=True)
    images_validation_dir = images_dir / "validation"
    images_validation_dir.mkdir(exist_ok=True)
    images_train_dir = images_dir / "train"
    images_train_dir.mkdir(exist_ok=True)


    labels_dir = training_data_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    labels_validation_dir = labels_dir / "validation"
    labels_validation_dir.mkdir(exist_ok=True)
    labels_train_dir = labels_dir / "train"
    labels_train_dir.mkdir(exist_ok=True)

    ## create training data yaml
    text = f"""
train: {images_train_dir.resolve()}
val: {images_validation_dir.resolve()}
nc: 3
names: ['grain_boundary', 'vacancy', 'interstitial']
    """
    with open(training_data_dir / "training_data.yaml", 'w') as f:
        f.write(text.strip())

    all_seeds = [s + 1 for s in range(num_seeds)]

    number_of_images = len(all_seeds) * len(rotations) * len(reflections)
    print(f"Generating {number_of_images} images...")

    training_split = 0.8

    # define random indices to be in the trianing set
    all_indices = np.arange(number_of_images)
    np.random.seed(0)
    np.random.shuffle(all_indices)
    split_index = int(training_split * number_of_images)
    training_indices = set(all_indices[:split_index])

    index = 0
    for seed in tqdm(all_seeds, total=num_seeds):
        stem_image, bboxes, metadata = generate_stem_image_with_continuous_gb(
            img_size=img_size,
            seed=seed
        )
        yolo_bounding_boxes = yolo_format_bounding_boxes(bboxes, img_size)


        pil_stem_image = Image.fromarray((stem_image * 255).astype(np.uint8))
        # Generate augmented images and save
        for rot in rotations:
            for refl in reflections:
                augmented_image, augmented_bboxes = rotate_flip_image_and_bboxes(
                    pil_stem_image,
                    bboxes,
                    rot,
                    refl
                )
                yolo_augmented_bboxes = yolo_format_bounding_boxes(augmented_bboxes, img_size)

                # Determine if this image is for training or validation
                if index in training_indices:
                    image_save_path = images_train_dir / f"image_seed{seed:04d}_rot{rot}_refl{refl}.png"
                    label_save_path = labels_train_dir / f"image_seed{seed:04d}_rot{rot}_refl{refl}.txt"
                else:
                    image_save_path = images_validation_dir / f"image_seed{seed:04d}_rot{rot}_refl{refl}.png"
                    label_save_path = labels_validation_dir / f"image_seed{seed:04d}_rot{rot}_refl{refl}.txt"

                # Save image
                augmented_image.save(image_save_path)

                 # Save bounding boxes to text file for yolo input
                np.savetxt(
                    label_save_path,
                    yolo_augmented_bboxes,
                    fmt='%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f'
                )

                index += 1