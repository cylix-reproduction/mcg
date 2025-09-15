import os
import shutil
import tarfile

import scipy.io as sio
from tqdm import tqdm

# Paths - Modify these as needed
SOURCE_DIR = os.path.expanduser("~/autodl-pub/ImageNet/ILSVRC2012")
TARGET_DIR = "./.data"
DEVKIT_PATH = os.path.join(SOURCE_DIR, "ILSVRC2012_devkit_t12.tar.gz")
TRAIN_TAR_PATH = os.path.join(SOURCE_DIR, "ILSVRC2012_img_train.tar")
VAL_TAR_PATH = os.path.join(SOURCE_DIR, "ILSVRC2012_img_val.tar")


def extract_tar_file(tar_path, extract_to):
    """Extract a tar file to specified directory"""
    if not os.path.exists(tar_path):
        print(f"File not found: {tar_path}")
        return False

    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=f"Extracting {os.path.basename(tar_path)}"):
            tar.extract(member, path=extract_to)
    print(f"Extracted to {extract_to}")
    return True


def setup_train_data():
    """Setup training data in folder structure required by ImageFolder"""
    train_source = os.path.join(TARGET_DIR, "Data", "CLS-LOC", "train")
    train_target = os.path.join(TARGET_DIR, "train")

    # Check if we need to extract train data
    if not os.path.exists(train_source):
        print("Extracting train tar file...")
        # Extract to a writable temporary location
        temp_train_dir = os.path.join(TARGET_DIR, "temp_train")
        os.makedirs(temp_train_dir, exist_ok=True)

        if not extract_tar_file(TRAIN_TAR_PATH, temp_train_dir):
            return False

        # Move extracted content to correct location
        extracted_content = os.path.join(temp_train_dir, "Data", "CLS-LOC", "train")
        if os.path.exists(extracted_content):
            shutil.copytree(extracted_content, train_source)
            # Clean up temp directory
            shutil.rmtree(temp_train_dir, ignore_errors=True)

        # For each class, extract the tar file
        print("Extracting individual class tar files...")
        class_tars = [f for f in os.listdir(train_source) if f.endswith(".tar")]

        for class_tar in tqdm(class_tars, desc="Extracting class tars"):
            class_name = class_tar[:-4]  # Remove .tar extension
            class_dir = os.path.join(train_source, class_name)
            os.makedirs(class_dir, exist_ok=True)

            with tarfile.open(os.path.join(train_source, class_tar), "r") as tar:
                members = tar.getmembers()
                for member in members:
                    tar.extract(member, path=class_dir)

            # Remove the tar file after extraction
            os.remove(os.path.join(train_source, class_tar))

    # Create target directory
    os.makedirs(train_target, exist_ok=True)

    # Move class folders to target directory
    class_folders = [
        f
        for f in os.listdir(train_source)
        if os.path.isdir(os.path.join(train_source, f))
    ]

    print("Moving training data to target structure...")
    for class_folder in tqdm(class_folders, desc="Moving training classes"):
        src = os.path.join(train_source, class_folder)
        dst = os.path.join(train_target, class_folder)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
        elif os.path.isdir(src) and os.path.exists(dst):
            # If destination exists, merge contents
            for file in os.listdir(src):
                shutil.copy2(os.path.join(src, file), os.path.join(dst, file))

    print("Training data setup complete")
    return True


def setup_val_data():
    """Setup validation data in folder structure required by ImageFolder"""
    val_source = os.path.join(TARGET_DIR, "Data", "CLS-LOC", "val")
    val_target = os.path.join(TARGET_DIR, "validation")

    # Check if we need to extract validation data
    if not os.path.exists(val_source):
        print("Extracting validation tar file...")
        # Extract to a writable temporary location
        temp_val_dir = os.path.join(TARGET_DIR, "temp_val")
        os.makedirs(temp_val_dir, exist_ok=True)

        if not extract_tar_file(VAL_TAR_PATH, temp_val_dir):
            return False

        # Create val_source directory and move all JPEG files there
        os.makedirs(val_source, exist_ok=True)
        # Move all extracted JPEG files to val_source
        for file in os.listdir(temp_val_dir):
            if file.endswith(".JPEG"):
                src = os.path.join(temp_val_dir, file)
                dst = os.path.join(val_source, file)
                shutil.copy2(src, dst)
        # Clean up temp directory
        shutil.rmtree(temp_val_dir, ignore_errors=True)

    # Extract devkit to get validation ground truth
    devkit_dir = os.path.join(TARGET_DIR, "devkit")
    if not os.path.exists(devkit_dir):
        # Extract to a writable temporary location
        temp_devkit_dir = os.path.join(TARGET_DIR, "temp_devkit")
        os.makedirs(temp_devkit_dir, exist_ok=True)

        if not extract_tar_file(DEVKIT_PATH, temp_devkit_dir):
            return False

        # Move extracted content to correct location using copytree instead of move
        extracted_content = os.path.join(temp_devkit_dir, "ILSVRC2012_devkit_t12")
        if os.path.exists(extracted_content):
            shutil.copytree(extracted_content, devkit_dir)
            # Clean up temp directory
            shutil.rmtree(temp_devkit_dir, ignore_errors=True)

    # Load validation ground truth
    val_meta_path = os.path.join(
        devkit_dir, "data", "ILSVRC2012_validation_ground_truth.txt"
    )
    if not os.path.exists(val_meta_path):
        # Try alternative path
        val_meta_path = os.path.join(
            TARGET_DIR,
            "ILSVRC2012_devkit_t12",
            "data",
            "ILSVRC2012_validation_ground_truth.txt",
        )
        if not os.path.exists(val_meta_path):
            print("Validation ground truth file not found")
            return False

    with open(val_meta_path, "r") as f:
        val_labels = [int(line.strip()) for line in f.readlines()]

    # Load synset words to map class indices to WordNet IDs
    meta_path = os.path.join(devkit_dir, "data", "meta.mat")
    if not os.path.exists(meta_path):
        # Try alternative path
        meta_path = os.path.join(
            TARGET_DIR, "ILSVRC2012_devkit_t12", "data", "meta.mat"
        )
        if not os.path.exists(meta_path):
            print("Meta file not found")
            return False

    meta = sio.loadmat(meta_path, squeeze_me=True)["synsets"]
    # Extract WordNet IDs (class names) from meta data - element at index 1 is the WordNet ID
    ilsvrc_classes = [str(meta[i][1]) for i in range(1000)]  # First 1000 classes

    # Create target directory
    os.makedirs(val_target, exist_ok=True)

    # Create class directories
    for class_id in tqdm(ilsvrc_classes, desc="Creating validation class directories"):
        class_dir = os.path.join(val_target, class_id)
        os.makedirs(class_dir, exist_ok=True)

    # Move validation images to corresponding class folders
    val_images = sorted([f for f in os.listdir(val_source) if f.endswith(".JPEG")])

    print("Organizing validation data by classes...")
    for i, img_name in enumerate(tqdm(val_images, desc="Moving validation images")):
        class_index = val_labels[i] - 1  # Convert to 0-based indexing
        if 0 <= class_index < len(ilsvrc_classes):  # Safety check
            class_id = ilsvrc_classes[class_index]
            src = os.path.join(val_source, img_name)
            dst = os.path.join(val_target, class_id, img_name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    print("Validation data setup complete")
    return True


def main():
    """Main function to convert ImageNet data to required format"""
    print("Starting ImageNet data conversion...")

    # Create target directory
    os.makedirs(TARGET_DIR, exist_ok=True)

    # # Setup training data
    # print("Setting up training data...")
    # if not setup_train_data():
    #     print("Training data setup failed")
    #     return

    # Setup validation data
    print("Setting up validation data...")
    if not setup_val_data():
        print("Validation data setup failed")
        return

    print("ImageNet data conversion completed!")
    print(f"Data is available at: {TARGET_DIR}")


if __name__ == "__main__":
    main()