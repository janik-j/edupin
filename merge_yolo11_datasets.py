from pathlib import Path
from typing import List, Dict, Optional, Set
import yaml
import sys
import shutil
import json

def merge_datasets(ds_dirs: List[Path], class_mapping_file: Optional[Path] = None):
    splits = ["train", "valid", "test"]

    # init structure for merged
    merged_root = ds_dirs[0].parent / "merged_yolov11"
    for split in splits:
        (merged_root / split / "images").mkdir(parents=True, exist_ok=True)
        (merged_root / split / "labels").mkdir(parents=True, exist_ok=True)

    # Load manual class mapping if provided
    manual_mappings = {}
    excluded_classes: Dict[str, Set[str]] = {}  # Format: {dataset_name: {class1, class2, ...}}
    
    if class_mapping_file and class_mapping_file.exists():
        with open(class_mapping_file, 'r') as f:
            file_ext = class_mapping_file.suffix.lower()
            if file_ext == '.json':
                manual_mappings = json.load(f)
            elif file_ext in ['.yaml', '.yml']:
                manual_mappings = yaml.safe_load(f)
            else:
                print(f"Unsupported mapping file format: {file_ext}")
                print("Proceeding without manual mappings")
        
        if manual_mappings:
            print(f"Loaded manual class mappings from {class_mapping_file}")
            
        # Process excluded classes if present in the config
        if manual_mappings and "excluded_classes" in manual_mappings:
            for exclude_entry in manual_mappings["excluded_classes"]:
                dataset = exclude_entry.get("dataset", "")
                classes = exclude_entry.get("classes", [])
                
                if dataset and classes:
                    if dataset not in excluded_classes:
                        excluded_classes[dataset] = set()
                    excluded_classes[dataset].update(classes)
                    print(f"Will exclude classes {classes} from dataset {dataset}")

    # iteratively add datasets
    unique_train_ids, unique_val_ids, unique_test_ids = set(), set(), set()
    all_classes = []
    class_mapping = {}  # Map original class ids to new global ids
    
    # First pass: collect all class names and create mapping
    for ds_idx, ds_dir in enumerate(ds_dirs):
        with open(ds_dir / "data.yaml", "r") as f:
            data = yaml.safe_load(f)
            
        class_names = data["names"]
        dataset_name = ds_dir.name
        print(f"Dataset {dataset_name} has {len(class_names)} classes: {class_names}")
        
        # Create mapping for this dataset's classes
        dataset_mapping = {}
        for i, class_name in enumerate(class_names):
            # Skip if class is in the exclusion list for this dataset
            if dataset_name in excluded_classes and class_name in excluded_classes[dataset_name]:
                dataset_mapping[i] = -1  # Mark as excluded
                print(f"Excluding class '{class_name}' from dataset '{dataset_name}'")
                continue
                
            # Check if this class has a manual mapping
            mapped_name = class_name
            if manual_mappings and "class_mappings" in manual_mappings:
                for mapping in manual_mappings["class_mappings"]:
                    for source in mapping.get("sources", []):
                        if source.get("dataset") == dataset_name and source.get("class") == class_name:
                            mapped_name = mapping["target"]
                            print(f"Mapping '{dataset_name}.{class_name}' to '{mapped_name}'")
                            break
            
            if mapped_name not in all_classes:
                all_classes.append(mapped_name)
                new_id = len(all_classes) - 1
            else:
                new_id = all_classes.index(mapped_name)
            dataset_mapping[i] = new_id
        
        class_mapping[dataset_name] = dataset_mapping
    
    print(f"Merged dataset will have {len(all_classes)} classes: {all_classes}")
    
    # Second pass: copy files and update labels
    for ds_idx, ds_dir in enumerate(ds_dirs):
        dataset_name = ds_dir.name
        dataset_mapping = class_mapping[dataset_name]
        
        for split in splits:
            images_dir = ds_dir / split / "images"
            labels_dir = ds_dir / split / "labels"
            
            if not images_dir.exists() or not labels_dir.exists():
                print(f"Skipping {split} split for {dataset_name} - directories not found")
                continue
                
            # Get unique IDs set for this split
            if split == "train":
                unique_ids_set = unique_train_ids
            elif split == "valid":
                unique_ids_set = unique_val_ids
            else:  # test
                unique_ids_set = unique_test_ids
            
            # Process all images and labels
            for img_path in images_dir.glob("*.*"):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                    
                img_id = img_path.stem
                new_id = f"{dataset_name}_{img_id}"
                
                # Skip if we've seen this ID before (shouldn't happen, but just in case)
                if new_id in unique_ids_set:
                    print(f"Warning: Duplicate ID {new_id} found in {split} split, skipping")
                    continue
                
                # Update and process label file if it exists
                label_path = labels_dir / f"{img_id}.txt"
                has_valid_labels = False
                updated_labels = []
                
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        labels = f.readlines()
                        
                    for label in labels:
                        parts = label.strip().split()
                        if not parts:
                            continue
                            
                        # Map the class ID to new global ID
                        old_class_id = int(parts[0])
                        new_class_id = dataset_mapping.get(old_class_id, -1)
                        
                        # Skip excluded classes
                        if new_class_id == -1:
                            continue
                        
                        # Update class ID and keep other values (bbox coordinates)
                        parts[0] = str(new_class_id)
                        updated_labels.append(" ".join(parts) + "\n")
                        has_valid_labels = True
                
                # Skip samples that have no valid labels after exclusion
                if not has_valid_labels and label_path.exists():
                    print(f"Skipping {new_id} - no valid labels after exclusion")
                    continue
                
                unique_ids_set.add(new_id)
                
                # Copy image with new name
                dest_img_path = merged_root / split / "images" / f"{new_id}{img_path.suffix}"
                shutil.copy(img_path, dest_img_path)
                
                # Write updated labels if there are any
                if updated_labels:
                    dest_label_path = merged_root / split / "labels" / f"{new_id}.txt"
                    with open(dest_label_path, 'w') as f:
                        f.writelines(updated_labels)
    
    # Create merged data.yaml
    merged_data = {
        "train": "../train/images",
        "val": "../valid/images", 
        "test": "../test/images",
        "nc": len(all_classes),
        "names": all_classes
    }
    
    with open(merged_root / "data.yaml", 'w') as f:
        yaml.dump(merged_data, f, sort_keys=False)
        
    print(f"Merged dataset created at {merged_root}")
    print(f"Train samples: {len(unique_train_ids)}")
    print(f"Validation samples: {len(unique_val_ids)}")
    print(f"Test samples: {len(unique_test_ids)}")

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python merge_yolo11_datasets.py <datasets_dir> [class_mapping_file]")
        sys.exit(1)
    
    root_dir = Path(sys.argv[1])
    class_mapping_file = Path(sys.argv[2]) if len(sys.argv) == 3 else None
    
    ds_dirs = [p for p in root_dir.glob("*") if p.is_dir() and "yolov11" in p.name]
    merge_datasets(ds_dirs, class_mapping_file)

if __name__ == "__main__":
    main()