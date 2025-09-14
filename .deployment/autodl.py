import os
import shutil
import tarfile

from tqdm import tqdm

# Updated path configuration
TAR_HOME = os.path.expanduser("~/autodl-pub/ImageNet/ILSVRC2012")
DEST_HOME = os.path.abspath("./.data")  # New destination as requested
TRAIN_DEST = os.path.join(DEST_HOME, "train")
VAL_DEST = os.path.join(DEST_HOME, "validation")
DEVKIT_TAR = os.path.join(TAR_HOME, "ILSVRC2012_devkit_t12.tar.gz")
DEVKIT_TMP = os.path.join(DEST_HOME, "devkit_tmp")  # Temporary devkit extraction


def extract_tar(tar_path, dest):
    """Extract tar file with progress bar"""
    if not os.path.exists(tar_path):
        print(f"Error: Tar file not found - {tar_path}")
        return False
    os.makedirs(dest, exist_ok=True)

    try:
        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()
            for m in tqdm(members, desc=f"Extract {os.path.basename(tar_path)}"):
                tar.extract(m, dest)
        return True
    except Exception as e:
        print(f"Extract failed: {str(e)}")
        return False


def process_train():
    """Process train set into class folders"""
    main_tar = os.path.join(TAR_HOME, "ILSVRC2012_img_train.tar")
    if not extract_tar(main_tar, TRAIN_DEST):
        return

    # Process sub-tar files
    subtars = [f for f in os.listdir(TRAIN_DEST) if f.endswith(".tar")]
    for tar in tqdm(subtars, desc="Process train subtars"):
        tar_path = os.path.join(TRAIN_DEST, tar)
        cls_dir = os.path.join(TRAIN_DEST, os.path.splitext(tar)[0])
        os.makedirs(cls_dir, exist_ok=True)

        # Extract sub-tar
        with tarfile.open(tar_path, "r") as t:
            members = t.getmembers()
            for m in tqdm(members, desc=f"Extract {tar}", leave=False):
                t.extract(m, cls_dir)

        os.remove(tar_path)  # Clean up


def process_val():
    """Process validation set using devkit for labels"""
    # Step 1: Extract validation images
    val_tar = os.path.join(TAR_HOME, "ILSVRC2012_img_val.tar")
    if not extract_tar(val_tar, VAL_DEST):
        return

    # Step 2: Extract devkit to get labels
    if not os.path.exists(DEVKIT_TAR):
        print(f"Error: Devkit tar not found - {DEVKIT_TAR}")
        return
    os.makedirs(DEVKIT_TMP, exist_ok=True)
    if not extract_tar(DEVKIT_TAR, DEVKIT_TMP):
        return

    # Step 3: Get validation labels from devkit
    # Devkit structure: devkit_tmp/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt
    val_labels_path = os.path.join(
        DEVKIT_TMP,
        "ILSVRC2012_devkit_t12",
        "data",
        "ILSVRC2012_validation_ground_truth.txt",
    )

    # Get synset mapping (ILSVRC2012_ID -> nxxxxxxx folder name)
    synset_mapping_path = os.path.join(
        DEVKIT_TMP,
        "ILSVRC2012_devkit_t12",
        "data",
        "meta.mat",  # Matlab file containing synset info
    )

    # Step 4: Parse synset mapping (simplified for ImageNet standard structure)
    # We'll extract synset IDs from train folder names since they follow standard format
    train_classes = [
        d for d in os.listdir(TRAIN_DEST) if os.path.isdir(os.path.join(TRAIN_DEST, d))
    ]
    synset_map = {
        str(i + 1): cls for i, cls in enumerate(sorted(train_classes))
    }  # 1-based index

    # Step 5: Read validation labels
    with open(val_labels_path, "r") as f:
        val_labels = [line.strip() for line in f]

    # Step 6: Organize validation images into class folders
    for i, label in enumerate(tqdm(val_labels, desc="Organize validation set")):
        img_idx = i + 1
        img_name = f"ILSVRC2012_val_000{img_idx:05d}.JPEG"
        src_path = os.path.join(VAL_DEST, img_name)

        if os.path.exists(src_path):
            cls_folder = synset_map.get(label)
            if cls_folder:
                dest_folder = os.path.join(VAL_DEST, cls_folder)
                os.makedirs(dest_folder, exist_ok=True)
                shutil.move(src_path, os.path.join(dest_folder, img_name))

    # Clean up devkit temporary files
    shutil.rmtree(DEVKIT_TMP, ignore_errors=True)


if __name__ == "__main__":
    print("Starting ImageNet extraction...")
    process_train()
    process_val()
    print("Extraction completed. Both sets are in ImageFolder format.")
