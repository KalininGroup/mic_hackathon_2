import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

def load_images(path_name):
    """Loads images from the specified path and returns them as a numpy array.

    Args:
        path_name (str): Path to the image file.
    Returns:
        numpy.ndarray: 2D array of shape (height, width) containing the images.
    """
    image = Image.open(path_name)
    image = np.asarray(image)
    if image.ndim == 3:
        # Convert to grayscale if RGB
        image = image[:, :, 0]
    return image

def chunk_image(image, chunk_size):
    """Chunks the input image into smaller patches of size chunk_size x chunk_size.

    Args:
        image (numpy.ndarray): 2D array representing the image to be chunked.
        chunk_size (int): Size of each chunk (both width and height).
    Returns:
        numpy.ndarray: 4D array of shape (num_chunks_y, num_chunks_x, chunk_size, chunk_size)
                       containing the image chunks.
    """
    img_height, img_width = image.shape
    chunks_y = img_height // chunk_size
    chunks_x = img_width // chunk_size

    if chunks_y == 0 or chunks_x == 0:
        raise ValueError("Chunk size is larger than the image dimensions.")

    if img_height % chunk_size != 0 or img_width % chunk_size != 0:
        print("Warning: Image dimensions are not perfectly divisible by chunk size. "
              "Trimming the image to fit.")
        # Trim the image to make it divisible by chunk_size
        image = image[:chunks_y * chunk_size, :chunks_x * chunk_size]

    # Reshape and transpose to get the chunks
    chunk_array = image.reshape(chunks_y, chunk_size, chunks_x, chunk_size)
    chunk_array = chunk_array.transpose(0, 2, 1, 3)

    return chunk_array

def _find_random_defect_boxes(image, num_defects=None):
    """Generates random bounding boxes to simulate defect detection.

    Args:
        image (numpy.ndarray): 2D array representing the image chunk.
        num_defects (int): Number of random defects to generate.
    Returns:
        list: List of bounding boxes in the format (x_min, y_min, width, height, [rotation]).
    """
    if num_defects is None:
        num_defects = np.random.randint(5, 10)
    height, width = image.shape
    boxes = []
    for _ in range(num_defects):
        box_width = np.random.randint(5, min(20, width // 2))
        box_height = np.random.randint(5, min(20, height // 2))
        x_min = np.random.randint(0, width - box_width)
        y_min = np.random.randint(0, height - box_height)
        rotation = np.random.uniform(0, 360)  # Random rotation angle
        # boxes are in format (class_id, x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4)
        class_id = np.random.randint(0, 3)  # Random class id (0, 1 or 2)
        box = (class_id,
               x_min, y_min,
               x_min + box_width * np.cos(np.radians(rotation)), y_min + box_width * np.sin(np.radians(rotation)),
               x_min + box_width * np.cos(np.radians(rotation)) - box_height * np.sin(np.radians(rotation)),
               y_min + box_width * np.sin(np.radians(rotation)) + box_height * np.cos(np.radians(rotation)),
               x_min - box_height * np.sin(np.radians(rotation)),
               y_min + box_height * np.cos(np.radians(rotation)))
        boxes.append(box)
    return boxes

def find_defects_yolo(model, image):
    """Detect defects in the image using a YOLO model.

    Args:
        model: YOLO model instance for inference.
        image (numpy.ndarray): 2D or 3D array representing the image.

    Returns:
        list: List of bounding boxes in the format (class_id, x1, y1, x2, y2, x3, y3, x4, y4).
              Each bounding box contains class_id followed by 4 corner points (x, y) of the oriented box.
    """
    # Convert grayscale to RGB if needed (YOLO expects 3 channels)
    if image.ndim == 2:
        # Stack grayscale image to create 3-channel RGB
        image = np.stack([image, image, image], axis=-1)

    # Run inference
    results = model(image, verbose=False, conf=0.1, iou=0.3)

    # Extract oriented bounding boxes
    bounding_boxes = []

    if len(results) > 0 and results[0].obb is not None:
        # Get OBB data
        obb_data = results[0].obb

        # Get class IDs and corner coordinates
        if obb_data.xyxyxyxy is not None and len(obb_data.xyxyxyxy) > 0:
            class_ids = obb_data.cls.cpu().numpy().astype(int)
            corners = obb_data.xyxyxyxy.cpu().numpy()

            # Convert to required format: (class_id, x1, y1, x2, y2, x3, y3, x4, y4)
            for cls_id, corner_points in zip(class_ids, corners):
                # corner_points shape is (4, 2) representing 4 (x, y) pairs
                bbox = (
                    int(cls_id),
                    float(corner_points[0, 0]), float(corner_points[0, 1]),  # x1, y1
                    float(corner_points[1, 0]), float(corner_points[1, 1]),  # x2, y2
                    float(corner_points[2, 0]), float(corner_points[2, 1]),  # x3, y3
                    float(corner_points[3, 0]), float(corner_points[3, 1])   # x4, y4
                )
                bounding_boxes.append(bbox)

    return bounding_boxes

def find_defects(image):
    """Located the defects in the image chunk.
    Args:
        image (numpy.ndarray): 2D array of shape (height, width) representing the image chunk.
    Returns:
        list of bounding boxes for the detected defects in the format (x_min, y_min, width, height, rotation).
    """

    #### Placeholder defect detection logic that defines random boxes ###
    defects = _find_random_defect_boxes(image)
    return defects

def display_bounding_boxes(image, bounding_boxes):
    """Displays the image with bounding boxes overlaid.

    Args:
        image (numpy.ndarray): 2D array representing the original image.
        bounding_boxes (list): List of bounding boxes in the format (x_min, y_min, width, height, [rotation]).
    """

    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')

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
        ax.plot(
            x + [x[0]],
            y + [y[0]],
            color=color, linewidth=.5)

    plt.show()

def main(path_name, model_path=None):
    """Main function to run defect detection.

    Args:
        path_name (str): Path to the image file.
        model_path (str, optional): Path to the YOLO model. If None, uses default path.
    """
    from pathlib import Path
    import os

    # Load YOLO model
    if model_path is None:
        model_path = Path(os.getcwd()) / "trained_models" / "yolov11_obb" / "model_1" / "weights" / "best.pt"

    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)

    # Load image
    image = load_images(path_name)

    # Run YOLO-based defect detection
    bounding_boxes = find_defects_yolo(model, image)

    # Display bounding boxes on original image
    display_bounding_boxes(image, bounding_boxes)
    return 0

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_image> [model_path]")
        sys.exit(1)

    path_name = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    main(path_name, model_path)