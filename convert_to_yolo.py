"""
Pascal VOC XML → YOLOv8 Format Converter
==========================================
يحوّل ملفات XML (Pascal VOC) إلى صيغة YOLOv8 (YOLO txt).

البنية الناتجة:
    yolo_dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml

الكلاسات الأربع:
    0: at_plastic
    1: ap_plastic
    2: at_metal
    3: ap_metal

صيغة YOLOv8 لكل كائن:
    <class_id> <x_center> <y_center> <width> <height>
    (جميع القيم منسوبة للصورة بين 0.0 و 1.0)
"""

import os
import xml.etree.ElementTree as ET
import shutil
import random
import yaml
from pathlib import Path

# ─── الإعدادات ───────────────────────────────────────────────────────────────
DATASET_ROOT = Path("/home/hamzah/Desktop/beykoz/proje/MAYIN TARLASI/landmine_final - Copy")
OUTPUT_DIR   = Path("/home/hamzah/Desktop/beykoz/proje/MAYIN TARLASI/yolo_dataset")

# نسب التقسيم
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

RANDOM_SEED = 42

# ─── خريطة الكلاسات ──────────────────────────────────────────────────────────
CLASS_MAP = {
    "at_plastic": 0,
    "ap_plastic": 1,
    "at_metal":   2,
    "ap_metal":   3,
}

# ─── إنشاء مجلدات الخرج ──────────────────────────────────────────────────────
def create_output_dirs():
    for split in ("train", "val", "test"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"✅ تم إنشاء مجلدات الخرج في: {OUTPUT_DIR}")

# ─── تحويل XML واحد إلى صيغة YOLO ───────────────────────────────────────────
def convert_xml_to_yolo(xml_path: Path):
    """
    يقرأ ملف XML بصيغة Pascal VOC ويرجع:
      - img_path: مسار الصورة المقابلة
      - yolo_lines: قائمة نصوص YOLO (سطر لكل كائن)
    أو None إذا فشل التحويل.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # ─ أبعاد الصورة
        size   = root.find("size")
        img_w  = int(size.find("width").text)
        img_h  = int(size.find("height").text)

        if img_w == 0 or img_h == 0:
            print(f"⚠️  صفر أبعاد في: {xml_path}")
            return None, None

        # ─ الصورة المقابلة
        filename = root.find("filename").text
        img_path = xml_path.parent / filename
        if not img_path.exists():
            # بعض الملفات قد لا يكون لها امتداد
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = xml_path.with_suffix(ext)
                if candidate.exists():
                    img_path = candidate
                    break
            else:
                print(f"⚠️  لم توجد الصورة: {img_path}")
                return None, None

        # ─ تعليقات الكائنات
        yolo_lines = []
        unknown_classes = set()

        for obj in root.findall("object"):
            name = obj.find("name").text.strip()

            if name not in CLASS_MAP:
                unknown_classes.add(name)
                continue

            class_id = CLASS_MAP[name]
            bndbox   = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # تأكد من الحدود
            xmin = max(0.0, xmin)
            ymin = max(0.0, ymin)
            xmax = min(img_w, xmax)
            ymax = min(img_h, ymax)

            if xmax <= xmin or ymax <= ymin:
                continue

            # تحويل إلى YOLO (نسبي ومركزي)
            x_center = ((xmin + xmax) / 2.0) / img_w
            y_center = ((ymin + ymax) / 2.0) / img_h
            width    = (xmax - xmin) / img_w
            height   = (ymax - ymin) / img_h

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        if unknown_classes:
            print(f"⚠️  كلاسات غير معروفة في {xml_path.name}: {unknown_classes}")

        return img_path, yolo_lines

    except Exception as e:
        print(f"❌ خطأ في تحليل {xml_path}: {e}")
        return None, None


# ─── جمع كل أزواج (xml, img) ─────────────────────────────────────────────────
def collect_all_pairs():
    """يبحث عن جميع ملفات XML ويطابقها مع الصور."""
    pairs = []
    xml_files = list(DATASET_ROOT.rglob("*.xml"))
    print(f"🔍 وُجد {len(xml_files)} ملف XML")

    for xml_path in xml_files:
        img_path, yolo_lines = convert_xml_to_yolo(xml_path)
        if img_path is not None and yolo_lines is not None:
            pairs.append((img_path, yolo_lines))

    print(f"✅ تم تحويل {len(pairs)} زوج (صورة + تعليق)")
    return pairs


# ─── التقسيم وحفظ الملفات ────────────────────────────────────────────────────
def split_and_save(pairs):
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    n        = len(pairs)
    n_train  = int(n * TRAIN_RATIO)
    n_val    = int(n * VAL_RATIO)

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train : n_train + n_val],
        "test":  pairs[n_train + n_val :],
    }

    stats = {}
    for split_name, split_pairs in splits.items():
        img_dir = OUTPUT_DIR / "images" / split_name
        lbl_dir = OUTPUT_DIR / "labels" / split_name

        for img_path, yolo_lines in split_pairs:
            # نسخ الصورة
            dst_img = img_dir / img_path.name
            # تجنب تضارب الأسماء بإضافة مجلد الأب
            if dst_img.exists():
                parent_prefix = img_path.parent.name
                dst_img = img_dir / f"{parent_prefix}_{img_path.name}"

            shutil.copy2(img_path, dst_img)

            # كتابة ملف YOLO txt
            dst_lbl = lbl_dir / (dst_img.stem + ".txt")
            dst_lbl.write_text("\n".join(yolo_lines))

        stats[split_name] = len(split_pairs)

    return stats


# ─── إنشاء data.yaml ─────────────────────────────────────────────────────────
def create_yaml(stats):
    yaml_content = {
        "path":  str(OUTPUT_DIR),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CLASS_MAP),
        "names": list(CLASS_MAP.keys()),
    }

    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✅ تم إنشاء data.yaml في: {yaml_path}")
    return yaml_path


# ─── التقرير النهائي ──────────────────────────────────────────────────────────
def print_report(stats, yaml_path):
    total = sum(stats.values())
    print("\n" + "=" * 55)
    print("         📊 تقرير التحويل النهائي")
    print("=" * 55)
    print(f"  المجموع الكلي  : {total:,} صورة")
    print(f"  Train          : {stats['train']:,}  ({stats['train']/total*100:.1f}%)")
    print(f"  Val            : {stats['val']:,}  ({stats['val']/total*100:.1f}%)")
    print(f"  Test           : {stats['test']:,}  ({stats['test']/total*100:.1f}%)")
    print("=" * 55)
    print(f"\n  الكلاسات ({len(CLASS_MAP)}):")
    for name, idx in CLASS_MAP.items():
        print(f"    {idx}: {name}")
    print(f"\n  ملف الإعداد: {yaml_path}")
    print("\n  لبدء التدريب:")
    print("    yolo train data=yolo_dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640")
    print("=" * 55)


# ─── نقطة الدخول ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 بدء تحويل Pascal VOC XML → YOLOv8\n")
    create_output_dirs()
    pairs = collect_all_pairs()

    if not pairs:
        print("❌ لم يُعثر على بيانات صالحة للتحويل!")
        exit(1)

    stats = split_and_save(pairs)
    yaml_path = create_yaml(stats)
    print_report(stats, yaml_path)

    print("\n✅ اكتمل التحويل بنجاح!")
