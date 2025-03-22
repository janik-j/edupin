import os
import torch
import numpy as np
from pathlib import Path
import yaml
import cv2
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class PedestrianTrafficLightDataset(Dataset):
    """
    PyTorch Dataset for the pedestrian traffic light dataset prepared for YOLOv8.
    Returns dictionary samples.
    """
    
    def __init__(
        self,
        data_root: str = "data/prepared_pedestrian_traffic_light",
        split: str = "train",
        transform: Optional[callable] = None,
        image_size: int = 640,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_root: Path to the root of the dataset
            split: Dataset split to use (train, valid, or test)
            transform: Optional additional transforms to apply
            image_size: Image size (height and width)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Load dataset config
        self.config = self._load_yaml(self.data_root / "data.yaml")
        self.num_classes = self.config.get("nc", 0)
        self.class_names = self.config.get("names", [])
        
        # Get paths to images and labels
        self.img_dir = self.data_root / split / "images"
        self.label_dir = self.data_root / split / "labels"
        
        # Get list of all image files
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.pt')])
        
        # Load dataset statistics
        stats_file = self.data_root / "dataset_stats.txt"
        self.mean, self.std = self._parse_stats(stats_file)
        
    def _load_yaml(self, yaml_file: Union[str, Path]) -> Dict:
        """Load YAML file."""
        with open(yaml_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _parse_stats(self, stats_file: Union[str, Path]) -> Tuple[List[float], List[float]]:
        """Parse dataset statistics from the stats file."""
        mean = [0.4038, 0.4144, 0.4064]  # Default values
        std = [0.3320, 0.3322, 0.3352]   # Default values
        
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                for line in f:
                    if line.startswith("Channel-wise mean:"):
                        # Extract mean values from the line
                        mean_str = line.split("[")[1].split("]")[0]
                        mean = [float(x.strip()) for x in mean_str.split(",")]
                    elif line.startswith("Channel-wise std:"):
                        # Extract std values from the line
                        std_str = line.split("[")[1].split("]")[0]
                        std = [float(x.strip()) for x in std_str.split(",")]
        
        return mean, std
    
    def _load_label(self, idx: int) -> torch.Tensor:
        """Load label file for an image."""
        img_file = self.img_files[idx]
        label_file = img_file.replace('.pt', '.txt')
        label_path = self.label_dir / label_file
        
        if not os.path.exists(label_path):
            return torch.zeros((0, 5))  # Return empty tensor if no labels
        
        # Load labels: class_id, x_center, y_center, width, height
        try:
            labels = np.loadtxt(label_path, delimiter=' ', ndmin=2).astype(np.float32)
            # Convert to tensor
            labels = torch.from_numpy(labels)
            return labels
        except Exception as e:
            print(f"Error loading label {label_path}: {e}")
            return torch.zeros((0, 5))
    
    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            dict: A dictionary containing:
                - 'image': normalized image tensor [C, H, W]
                - 'labels': tensor of labels [n_objects, 5] where each row is
                  [class_id, x_center, y_center, width, height]
                - 'image_id': image identifier
                - 'orig_shape': original image shape [h, w]
                - 'path': path to the image file
        """
        # Get image path
        img_file = self.img_files[idx]
        img_path = str(self.img_dir / img_file)
        
        # Load image tensor
        img = torch.load(img_path)
        
        # Load labels: [class_id, x_center, y_center, width, height]
        labels = self._load_label(idx)
        
        # Apply additional transforms if provided
        if self.transform is not None:
            img, labels = self.transform(img, labels)
        
        # Build sample dictionary
        sample = {
            'image': img,
            'labels': labels,
            'image_id': img_file,
            'orig_shape': (self.image_size, self.image_size),  # Images are already padded to square
            'path': img_path
        }
        
        return sample


# Define collate function outside of the main block so it can be pickled
def collate_fn(batch):
    """Custom collate function for DataLoader that handles variable-sized labels."""
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'labels': [item['labels'] for item in batch],
        'image_ids': [item['image_id'] for item in batch],
        'paths': [item['path'] for item in batch]
    }


def visualize_sample(sample, dataset, sample_idx=None, total_samples=None):
    """
    Visualize a sample with bounding boxes and class labels.
    
    Args:
        sample (dict): Sample dictionary from the dataset
        dataset (Dataset): Dataset instance (for class names)
        sample_idx (int, optional): Index of the sample for title
        total_samples (int, optional): Total number of samples for title
    """
    # Get image dimensions for scaling the normalized coordinates
    img = sample["image"].permute(1, 2, 0).numpy()
    h, w = img.shape[:2]
    
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img)
    
    # Print the number of labels
    num_labels = len(sample["labels"])
    if sample_idx is not None:
        print(f"Sample {sample_idx}: Image {sample['image_id']} - {num_labels} labels found")
    else:
        print(f"Image {sample['image_id']} - {num_labels} labels found")
    
    # Create a rectangle patch for each label
    for label in sample["labels"]:
        class_id = int(label[0])
        # Scale normalized coordinates to pixel values
        x_center, y_center = label[1] * w, label[2] * h
        width, height = label[3] * w, label[4] * h
        
        print(f"  Label: class={class_id}, center=({x_center:.1f}, {y_center:.1f}), size=({width:.1f}, {height:.1f})")
        
        # Create rectangle (x,y is top-left corner)
        rect = patches.Rectangle(
            (x_center - width/2, y_center - height/2),
            width, height, 
            linewidth=2, 
            edgecolor='r', 
            facecolor='none'
        )
        
        # Add rectangle to plot
        plt.gca().add_patch(rect)
        
        # Add class label
        class_name = dataset.class_names[class_id] if class_id < len(dataset.class_names) else f"Class {class_id}"
        plt.text(
            x_center - width/2, 
            y_center - height/2 - 5, 
            class_name,
            color='white', 
            fontsize=12, 
            backgroundcolor='red'
        )
    
    # Set title
    if sample_idx is not None and total_samples is not None:
        plt.title(f"Image: {sample['image_id']} ({sample_idx+1}/{total_samples})")
    else:
        plt.title(f"Image: {sample['image_id']}")
        
    plt.axis('off')
    plt.tight_layout()
    
    return fig


# Example usage:
if __name__ == "__main__":
    plt.style.use('dark_background')
    
    # Create dataset
    dataset = PedestrianTrafficLightDataset(
        data_root="data/prepared_pedestrian_traffic_light",
        split="train"
    )
    
    # Number of samples to visualize
    num_samples = 5
    
    # Visualize multiple samples
    for i in range(min(num_samples, len(dataset))):
        # Get sample and visualize
        sample = dataset[i]
        fig = visualize_sample(sample, dataset, i, num_samples)
        
        # Show plot and wait for it to be closed before continuing
        plt.show(block=True)
