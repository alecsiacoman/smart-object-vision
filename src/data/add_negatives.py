import shutil
import random
from pathlib import Path
from pycocotools.coco import COCO

def add_negative_images_no_cup(
    split_dir: Path,
    ann_path: Path,
    out_dir: Path,
    cat_name: str = "cup",
    max_negatives: int | None = None,
    seed: int = 42,
):
    """
    Copy images that do NOT contain `cat_name` into YOLO structure and
    create empty label files for them.

    Parameters
    ----------
    split_dir : Path
        Path to COCO images directory for the split (e.g. coco/train2017/train2017).
    ann_path : Path
        Path to COCO instances annotation json for the split.
    out_dir : Path
        Output root (e.g. mug_coco_yolo).
    cat_name : str
        Category name in COCO (default "cup").
    max_negatives : int | None
        If set, randomly sample up to this many negative images.
    seed : int
        Random seed for reproducibility.
    """
    coco = COCO(str(ann_path))

    cat_ids = coco.getCatIds(catNms=[cat_name])
    if not cat_ids:
        raise ValueError(f"Category '{cat_name}' not found in annotations: {ann_path}")
    cup_id = cat_ids[0]

    # Images that contain at least one cup annotation
    cup_img_ids = set(coco.getImgIds(catIds=[cup_id]))

    # All images in this COCO split (from annotations)
    all_img_ids = coco.getImgIds()
    neg_img_ids = [img_id for img_id in all_img_ids if img_id not in cup_img_ids]

    # Only keep negatives that actually exist in the given split_dir
    neg_infos = coco.loadImgs(neg_img_ids)
    existing_neg_infos = []
    for info in neg_infos:
        src = split_dir / info["file_name"]
        if src.exists():
            existing_neg_infos.append(info)

    # Sample if requested
    if max_negatives is not None and len(existing_neg_infos) > max_negatives:
        random.seed(seed)
        existing_neg_infos = random.sample(existing_neg_infos, k=max_negatives)

    img_out = out_dir / "images" / split_dir.name
    lab_out = out_dir / "labels" / split_dir.name
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_already_present = 0

    for info in existing_neg_infos:
        fname = info["file_name"]
        src = split_dir / fname
        dst_img = img_out / fname
        dst_lab = lab_out / (Path(fname).stem + ".txt")

        # Do not overwrite if already there
        if dst_img.exists() and dst_lab.exists():
            skipped_already_present += 1
            continue

        shutil.copy2(src, dst_img)
        # Create empty label file
        dst_lab.write_text("")
        copied += 1

    print(f"[{split_dir.name}] negatives found in split: {len(existing_neg_infos)}")
    print(f"[{split_dir.name}] copied new negatives: {copied}")
    print(f"[{split_dir.name}] skipped already present: {skipped_already_present}")


def main():
    root = Path("coco")
    out = Path("mug_coco_yolo")

    # You said you have ~6500 positives overall; for 1:1 you can do ~6500 negatives total.
    # You can split that across train/val in proportion, or keep val as-is.
    #
    # A simple approach:
    # - Add negatives ONLY to train split (recommended), keep val curated for reporting consistency, OR
    # - Add to both train and val to reflect real-world background.
    #
    # Below: add to both, with caps you can adjust.

    add_negative_images_no_cup(
        split_dir=root / "train2017" / "train2017",
        ann_path=root / "annotations" / "annotations" / "instances_train2017.json",
        out_dir=out,
        cat_name="cup",
        max_negatives=6500,   # set None to take all available negatives
        seed=42,
    )

    add_negative_images_no_cup(
        split_dir=root / "val2017" / "val2017",
        ann_path=root / "annotations" / "annotations" / "instances_val2017.json",
        out_dir=out,
        cat_name="cup",
        max_negatives=1000,   # optional; you can set 0/None depending on how you want val
        seed=42,
    )


if __name__ == "__main__":
    main()
