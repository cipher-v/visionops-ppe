"""Generate deterministic visual artifacts for the Construction-PPE audit."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from audit_dataset import AuditResult, audit_dataset, image_files, load_dataset_config


def save_bar_chart(values: dict[str, int], title: str, ylabel: str, path: Path) -> None:
    """Save a labeled bar chart using Matplotlib's default color cycle."""
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(values.keys(), values.values())
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_audit_distributions(result: AuditResult, output_dir: Path) -> None:
    """Create the four required aggregate audit charts."""
    summary = result.summary
    save_bar_chart(
        summary["box_count_per_class"],
        "Bounding boxes per class",
        "Bounding boxes",
        output_dir / "class_distribution.png",
    )
    save_bar_chart(
        summary["images_containing_class"],
        "Images containing each class",
        "Images",
        output_dir / "images_per_class.png",
    )

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(result.box_areas, bins=50)
    axis.set_title("Bounding-box relative area distribution")
    axis.set_xlabel("Box area / image area")
    axis.set_ylabel("Bounding boxes")
    figure.tight_layout()
    figure.savefig(output_dir / "box_area_distribution.png", dpi=150)
    plt.close(figure)

    widths = [width for width, _ in result.image_sizes]
    heights = [height for _, height in result.image_sizes]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.scatter(widths, heights, alpha=0.5)
    axis.set_title("Image size distribution")
    axis.set_xlabel("Width (pixels)")
    axis.set_ylabel("Height (pixels)")
    figure.tight_layout()
    figure.savefig(output_dir / "image_size_distribution.png", dpi=150)
    plt.close(figure)


def annotate_image(
    image_path: Path,
    label_path: Path,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Render one image with its YOLO annotations."""
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    height, width = image.shape[:2]
    figure, axis = plt.subplots(figsize=(12, 8))
    axis.imshow(image)
    axis.axis("off")
    colors = plt.get_cmap("tab20")

    if label_path.is_file():
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            fields = raw_line.split()
            if len(fields) != 5:
                continue
            try:
                class_value, x_center, y_center, box_width, box_height = map(
                    float, fields
                )
                class_id = int(class_value)
            except ValueError:
                continue
            if class_value != class_id or not 0 <= class_id < len(class_names):
                continue
            left = (x_center - box_width / 2) * width
            top = (y_center - box_height / 2) * height
            color = colors(class_id % colors.N)
            axis.add_patch(
                Rectangle(
                    (left, top),
                    box_width * width,
                    box_height * height,
                    fill=False,
                    edgecolor=color,
                    linewidth=2,
                )
            )
            axis.text(
                left,
                max(0, top),
                class_names[class_id],
                color="white",
                fontsize=8,
                bbox={"facecolor": color, "alpha": 0.8, "pad": 2},
            )

    figure.tight_layout(pad=0)
    figure.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def create_samples(
    dataset_root: Path,
    output_dir: Path,
    sample_count: int,
    seed: int,
) -> None:
    """Select and annotate a deterministic training-image sample."""
    _, class_names = load_dataset_config(dataset_root)
    train_images = image_files(dataset_root / "images" / "train")
    if sample_count > len(train_images):
        raise ValueError(
            f"Requested {sample_count} samples but found {len(train_images)} images"
        )
    selected = random.Random(seed).sample(train_images, sample_count)
    for index, image_path in enumerate(selected, start=1):
        relative = image_path.relative_to(dataset_root / "images" / "train")
        label_path = dataset_root / "labels" / "train" / relative.with_suffix(".txt")
        output_name = f"sample_{index:02d}_{image_path.stem}.png"
        annotate_image(image_path, label_path, class_names, output_dir / output_name)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/construction-ppe")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/dataset_audit")
    )
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the audit and generate plots and annotated samples."""
    args = parse_args()
    result = audit_dataset(args.dataset_root, args.output_dir)
    plot_audit_distributions(result, args.output_dir)
    create_samples(
        args.dataset_root.resolve(),
        args.output_dir,
        args.sample_count,
        args.seed,
    )
    print(f"Generated dataset audit artifacts in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
