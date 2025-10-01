"""Download PRBench from Hugging Face and restore the local benchmark layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from tqdm import tqdm

DEFAULT_REPO_ID = "yzweak/PRBench"
DEFAULT_OUTPUT_DIR = Path("eval") / "data"


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    base_dir = output_dir.resolve().parent.parent
    input_dir = base_dir / "input_dir"
    fine_dir = output_dir / "Fine_grained_evaluation"
    twitter_dir = output_dir / "twitter_figure"
    xhs_dir = output_dir / "xhs_figure"

    for directory in (output_dir, input_dir, fine_dir, twitter_dir, xhs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": base_dir,
        "input_dir": input_dir,
        "fine_dir": fine_dir,
        "twitter_dir": twitter_dir,
        "xhs_dir": xhs_dir,
    }


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)


def _write_text(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(data)


def reconstruct(repo_id: str, subset: str, output_dir: Path) -> None:
    print(f"Loading dataset from {repo_id} (split={subset})...")
    dataset = load_dataset(repo_id, split="full", trust_remote_code=True)
    print(f"Loaded {len(dataset)} records.")

    dirs = ensure_dirs(output_dir)

    # Prepare metadata aggregation for JSON dumps.
    full_records: List[Dict] = []
    figure_written = set()
    pdf_written = set()
    processed = 0

    for record in tqdm(dataset, desc="Rebuilding assets"):
        origin_data_raw = record.get("origin_data") or "{}"
        try:
            origin_data = json.loads(origin_data_raw)
        except json.JSONDecodeError:
            origin_data = {}

        metadata = {
            "title": record.get("title", ""),
            "arxiv_id": record.get("arxiv_id", ""),
            "PDF_path": record.get("PDF_path", ""),
            "platform_source": record.get("platform_source", ""),
            "id": record.get("id", ""),
            "figure_path": record.get("figure_path", []),
            "markdown_content": record.get("markdown_content", ""),
            "origin_data": origin_data,
            "is_core": bool(record.get("is_core")),
            "paper_pdf_path": record.get("paper_pdf_filename", ""),
            "checklist_yaml_path": record.get("checklist_filename", ""),
        }
        full_records.append(metadata)

        if subset == "core" and not metadata["is_core"]:
            continue

        processed += 1

        arxiv_id = metadata["arxiv_id"]
        post_id = metadata["id"]
        platform = (metadata["platform_source"] or "").upper()

        # Restore PDFs + checklists under Fine_grained_evaluation.
        if arxiv_id:
            paper_dir = dirs["fine_dir"] / arxiv_id
            pdf_filename = record.get("paper_pdf_filename") or f"{arxiv_id}.pdf"
            pdf_bytes: bytes = record.get("paper_pdf") or b""
            if pdf_bytes and arxiv_id not in pdf_written:
                _write_bytes(paper_dir / pdf_filename, pdf_bytes)
                pdf_written.add(arxiv_id)

            checklist_filename = record.get("checklist_filename") or "checklist.yaml"
            checklist_text: str = record.get("checklist_yaml") or ""
            if checklist_text:
                _write_text(paper_dir / "Factual_accuracy" / checklist_filename, checklist_text)

            # Populate input_dir entry for this promotion.
            if pdf_bytes:
                promo_dir = dirs["input_dir"] / str(post_id)
                target_pdf = promo_dir / pdf_filename
                if not target_pdf.exists():
                    _write_bytes(target_pdf, pdf_bytes)

        # Restore figures.
        figure_paths = metadata["figure_path"] or []
        figures = record.get("figures") or []
        for rel_path, figure_entry in zip(figure_paths, figures):
            dest_rel = Path(rel_path)
            if dest_rel.parts and dest_rel.parts[0] == "files":
                dest_rel = Path(*dest_rel.parts[1:])
            if dest_rel.parts and dest_rel.parts[0] == "figures":
                dest_rel = Path(*dest_rel.parts[1:])

            dest_root = dirs["twitter_dir"] if platform == "TWITTER" else dirs["xhs_dir"]
            dest_path = dest_root / dest_rel

            if str(dest_path) in figure_written:
                continue

            if isinstance(figure_entry, dict):
                if figure_entry.get("bytes"):
                    data = figure_entry["bytes"]
                elif figure_entry.get("path"):
                    data = Path(figure_entry["path"]).read_bytes()
                else:
                    continue
            elif isinstance(figure_entry, (str, Path)):
                data = Path(figure_entry).read_bytes()
            else:
                continue

            _write_bytes(dest_path, data)
            figure_written.add(str(dest_path))

    # Write metadata files.
    academic_path = output_dir / "academic_promotion_data.json"
    _write_text(academic_path, json.dumps(full_records, ensure_ascii=False, indent=2))

    core_subset = [record for record in full_records if record.get("is_core")]
    _write_text(output_dir / "academic_promotion_data_core.json", json.dumps(core_subset, ensure_ascii=False, indent=2))

    print(f"Reconstructed {processed} records for subset '{subset}'.")
    print("\nReconstruction complete!")
    print(f"Metadata directory: {output_dir}")
    print(f"Input directory: {dirs['input_dir']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and reconstruct the PRBench dataset.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repository ID.")
    parser.add_argument("--subset", default="full", choices=["full", "core"], help="Dataset split to download.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path, help="Destination directory (default: eval/data).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reconstruct(args.repo_id, args.subset, args.output_dir)


if __name__ == "__main__":
    main()
