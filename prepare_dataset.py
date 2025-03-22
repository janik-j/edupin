import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import shutil
from pathlib import Path

def compute_dataset_stats(data_dir, split="train"):
    """
    Compute channel-wise mean and standard deviation over the original images
    
    Args:
        data_dir (str): Path to the dataset root directory
        split (str): Dataset split to compute statistics on (train, valid, or test)
        
    Returns:
        tuple: (mean, std) channel-wise statistics
    """
    img_dir = os.path.join(data_dir, split, "images")
    
    # Get list of image files
    img_files = [f for f in os.listdir(img_dir) 
                if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Initialize variables to accumulate values
    pixel_count = 0
    channel_sum = torch.zeros(3)
    channel_sum_squared = torch.zeros(3)
    
    print(f"Computing channel-wise statistics on {len(img_files)} images...")
    
    # Convert images to tensors and compute statistics
    to_tensor = transforms.ToTensor()  # Scales to 0-1 range
    
    for img_file in tqdm(img_files):
        img_path = os.path.join(img_dir, img_file)
        
        # Load original image without preprocessing
        img = Image.open(img_path).convert("RGB")
        
        # Convert to tensor (0-1 range)
        img_tensor = to_tensor(img)
        
        # Reshape to (3, -1) for channel-wise statistics
        img_tensor = img_tensor.view(3, -1)
        
        # Update sums
        channel_sum += img_tensor.sum(dim=1)
        channel_sum_squared += (img_tensor ** 2).sum(dim=1)
        
        # Update pixel count
        pixel_count += img_tensor.shape[1]
    
    # Calculate mean and std
    mean = channel_sum / pixel_count
    # Var[X] = E[X^2] - E[X]^2
    std = torch.sqrt((channel_sum_squared / pixel_count) - (mean ** 2))
    
    return mean.tolist(), std.tolist()

def prepare_images(data_dir, output_dir, mean, std, target_size=640, splits=None):
    """
    Prepare images for YOLOv8 m model:
    1. Normalize using channel-wise mean and std
    2. Resize keeping aspect ratio so larger dimension is target_size
    3. Zero pad the smaller dimension
    
    Args:
        data_dir (str): Path to original dataset
        output_dir (str): Path to save processed dataset
        mean (list): Channel-wise mean values [r, g, b]
        std (list): Channel-wise std values [r, g, b]
        target_size (int): Target image size (YOLOv8 m uses 640)
        splits (list): Dataset splits to process, e.g., ['train', 'valid', 'test']
    """
    if splits is None:
        splits = ['train', 'valid', 'test']
    
    # Create transforms for preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Process each split
    for split in splits:
        print(f"Processing {split} split...")
        
        # Setup directories
        src_img_dir = os.path.join(data_dir, split, "images")
        src_label_dir = os.path.join(data_dir, split, "labels")
        
        dst_img_dir = os.path.join(output_dir, split, "images")
        dst_label_dir = os.path.join(output_dir, split, "labels")
        
        # Create output directories
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_label_dir, exist_ok=True)
        
        # Get image files
        img_files = [f for f in os.listdir(src_img_dir) 
                    if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(img_files):
            # Process image
            img_path = os.path.join(src_img_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            
            # Original dimensions
            orig_w, orig_h = img.size
            
            # Calculate scaling to make larger dimension target_size
            scale = target_size / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            
            # Resize image
            resized_img = img.resize((new_w, new_h), Image.BILINEAR)
            
            # Create padded image (black padding)
            padded_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
            padded_img.paste(resized_img, (0, 0))
            
            # Apply normalization
            normalized_img = preprocess(padded_img)
            
            # Save as Torch tensor
            output_path = os.path.join(dst_img_dir, img_file.rsplit('.', 1)[0] + '.pt')
            torch.save(normalized_img, output_path)
            
            # Copy corresponding label file (if exists)
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label_path = os.path.join(src_label_dir, label_file)
            dst_label_path = os.path.join(dst_label_dir, label_file)
            
            if os.path.exists(src_label_path):
                # For YOLO format, we need to adjust the coordinates due to resizing/padding
                with open(src_label_path, 'r') as src_file:
                    with open(dst_label_path, 'w') as dst_file:
                        for line in src_file:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id = parts[0]
                                # YOLO format: class_id, x_center, y_center, width, height (normalized)
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                # Adjust for the new dimensions
                                new_x_center = (x_center * orig_w * scale) / target_size
                                new_y_center = (y_center * orig_h * scale) / target_size
                                new_width = (width * orig_w * scale) / target_size
                                new_height = (height * orig_h * scale) / target_size
                                
                                # Write adjusted coordinates
                                dst_file.write(f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n")
    
    # Copy data.yaml file
    shutil.copy(os.path.join(data_dir, "data.yaml"), os.path.join(output_dir, "data.yaml"))
    
    print(f"Dataset preparation complete. Processed data saved to {output_dir}")

def main():
    # Dataset paths
    data_dir = "data/pedestrian Traffic Light"
    output_dir = "data/prepared_pedestrian_traffic_light"
    
    # Compute dataset statistics
    mean, std = compute_dataset_stats(data_dir, split="train")
    
    print(f"Dataset channel-wise statistics:")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    
    # Prepare dataset for YOLOv8 m
    prepare_images(
        data_dir=data_dir,
        output_dir=output_dir,
        mean=mean,
        std=std,
        target_size=640,  # YOLOv8 m input dimension
        splits=['train', 'valid', 'test']
    )
    
    # Create a file with dataset statistics for future reference
    stats_file = os.path.join(output_dir, "dataset_stats.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Channel-wise mean: {mean}\n")
        f.write(f"Channel-wise std: {std}\n")
        f.write(f"Target size: 640\n")

if __name__ == "__main__":
    main() 