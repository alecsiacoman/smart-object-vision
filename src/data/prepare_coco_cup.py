import os, shutil, json
from pathlib import Path
from pycocotools.coco import COCO

def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    xc = (x + w/2) / img_w
    yc = (y + h/2) / img_h
    return f"0 {xc:.6f} {yc:.6f} {(w/img_w):.6f} {(h/img_h):.6f}\n"  # class 0 = mug

def extract_cup_split(split_dir, ann_path, out_dir):
    coco = COCO(ann_path)
    cup_id = coco.getCatIds(catNms=['cup'])[0]
    img_ids = coco.getImgIds(catIds=[cup_id])

    img_out = Path(out_dir, 'images', split_dir.name); img_out.mkdir(parents=True, exist_ok=True)
    lab_out = Path(out_dir, 'labels', split_dir.name); lab_out.mkdir(parents=True, exist_ok=True)

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        fname = img_info['file_name']
        src = split_dir / fname
        if not src.exists():  # skip if image is in the other split
            continue
        shutil.copy2(src, img_out / fname)

        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cup_id])
        anns = coco.loadAnns(ann_ids)
        with open(lab_out / (Path(fname).stem + '.txt'), 'w') as f:
            for ann in anns:
                f.write(coco_to_yolo_bbox(ann['bbox'], img_info['width'], img_info['height']))

def main():
    root = Path('coco')
    out = Path('mug_coco_yolo')

    extract_cup_split(
        root / 'train2017' / 'train2017',
        root / 'annotations' / 'annotations' / 'instances_train2017.json',
        out
    )

    extract_cup_split(
        root / 'val2017' / 'val2017',
        root / 'annotations' / 'annotations' / 'instances_val2017.json',
        out
    )


if __name__ == "__main__":
    main()
