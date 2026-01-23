import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_negative_label(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    return label_path.read_text(encoding="utf-8", errors="ignore").strip() == ""


def collect_samples(images_dir: Path, labels_dir: Path):
    positives, negatives = [], []

    for img_path in images_dir.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        if is_negative_label(label_path):
            negatives.append((img_path, label_path))
        else:
            positives.append((img_path, label_path))

    return positives, negatives


def copy_pairs(pairs, out_images: Path, out_labels: Path):
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in pairs:
        shutil.copy2(img_path, out_images / img_path.name)
        shutil.copy2(label_path, out_labels / label_path.name)


def make_small_yolo_dataset(
    src_root: Path,
    dst_root: Path,
    train_split: str = "train2017",
    val_split: str = "val2017",
    n_train_pos: int = 3000,
    n_train_neg: int = 3000,
    n_val_pos: int = 390,
    n_val_neg: int = 390,
    seed: int = 42,
    overwrite: bool = False,
):
    random.seed(seed)

    if overwrite and dst_root.exists():
        shutil.rmtree(dst_root)

    src_train_images = src_root / "images" / train_split
    src_train_labels = src_root / "labels" / train_split
    src_val_images = src_root / "images" / val_split
    src_val_labels = src_root / "labels" / val_split

    for p in [src_train_images, src_train_labels, src_val_images, src_val_labels]:
        if not p.exists():
            raise FileNotFoundError(f"Missing expected folder: {p}")

    train_pos, train_neg = collect_samples(src_train_images, src_train_labels)
    val_pos, val_neg = collect_samples(src_val_images, src_val_labels)

    print(f"Found train: {len(train_pos)} positives, {len(train_neg)} negatives")
    print(f"Found val:   {len(val_pos)} positives, {len(val_neg)} negatives")

    if len(train_pos) < n_train_pos:
        raise ValueError(
            f"Not enough TRAIN positives: requested {n_train_pos}, available {len(train_pos)}"
        )
    if len(train_neg) < n_train_neg:
        raise ValueError(
            f"Not enough TRAIN negatives: requested {n_train_neg}, available {len(train_neg)}"
        )


    if n_val_pos > len(val_pos):
        print(f"[val] Capping positives from {n_val_pos} to {len(val_pos)} (availability)")
        n_val_pos = len(val_pos)
    if n_val_neg > len(val_neg):
        print(f"[val] Capping negatives from {n_val_neg} to {len(val_neg)} (availability)")
        n_val_neg = len(val_neg)

    train_pos_s = random.sample(train_pos, n_train_pos)
    train_neg_s = random.sample(train_neg, n_train_neg)
    val_pos_s = random.sample(val_pos, n_val_pos) if n_val_pos > 0 else []
    val_neg_s = random.sample(val_neg, n_val_neg) if n_val_neg > 0 else []

    dst_train_images = dst_root / "images" / train_split
    dst_train_labels = dst_root / "labels" / train_split
    dst_val_images = dst_root / "images" / val_split
    dst_val_labels = dst_root / "labels" / val_split

    copy_pairs(train_pos_s + train_neg_s, dst_train_images, dst_train_labels)
    copy_pairs(val_pos_s + val_neg_s, dst_val_images, dst_val_labels)

    print(f"Small dataset written to: {dst_root}")
    print(f"Train total: {n_train_pos + n_train_neg} images")
    print(f"Val total:   {n_val_pos + n_val_neg} images")
    print("Done.")


if __name__ == "__main__":
    PROJECT_ROOT = Path(r"D:\an4sem1\PRS\smart-object-detection")

    SRC = PROJECT_ROOT / "cup_coco_yolo"

    DST = PROJECT_ROOT / "cup_coco_yolo_small"

    make_small_yolo_dataset(
        src_root=SRC,
        dst_root=DST,
        n_train_pos=3000,
        n_train_neg=3000,
        n_val_pos=390,
        n_val_neg=390,
        seed=42,
        overwrite=True,
    )
