"""Download PRBench from Hugging Face and restore the local benchmark layout."""
from __future__ import annotations

import argparse
import base64
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import hf_hub_download
from datasets import load_dataset
from tqdm import tqdm

DEFAULT_REPO_ID = "yzweak/PRBench"
DEFAULT_OUTPUT_DIR = Path("eval_test") / "data"


def _json_default(value: Any) -> Any:
    """Fallback serializer for objects not handled by the stdlib JSON encoder."""
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(value).decode("utf-8")
    return str(value)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default)


def ensure_assets(repo_id: str, base_dir: Path) -> Path:
    """Download and extract the large asset bundle if not already present."""
    assets_root = base_dir / ".prbench_assets"
    files_root = assets_root / "files"
    if files_root.exists():
        return files_root

    print("Downloading asset archive from Hugging Face (this may take a while)...")
    archive_path = hf_hub_download(repo_id, filename="assets.tar.gz", repo_type="dataset")

    assets_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting assets from {archive_path}...")
    with tarfile.open(archive_path, "r:gz") as bundle:
        bundle.extractall(assets_root)

    return files_root


def _resolve_asset(files_root: Path, relative_path: str | None) -> Path | None:
    if not relative_path:
        return None

    path_str = relative_path.strip().lstrip("/").replace("\\", "/")
    if not path_str:
        return None

    sanitized = Path(path_str)
    if sanitized.parts and sanitized.parts[0] == "files":
        sanitized = Path(*sanitized.parts[1:])

    candidate = files_root / sanitized
    return candidate if candidate.exists() else None


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
    dataset = load_dataset(repo_id, split="full")
    print(f"Loaded {len(dataset)} records.")

    dirs = ensure_dirs(output_dir)
    files_root = ensure_assets(repo_id, dirs["base_dir"])

    # Prepare metadata aggregation for JSON dumps.
    full_records: List[Dict] = []
    figure_written = set()
    pdf_written = set()
    processed = 0

    for record in tqdm(dataset, desc="Rebuilding assets"):
        origin_data_field = record.get("origin_data")
        if isinstance(origin_data_field, str):
            if not origin_data_field.strip():
                origin_data = {}
            else:
                try:
                    origin_data = json.loads(origin_data_field)
                except json.JSONDecodeError:
                    origin_data = {}
        elif isinstance(origin_data_field, dict):
            origin_data = origin_data_field
        else:
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
            "paper_pdf_path": record.get("paper_pdf_path", ""),
            "checklist_yaml_path": record.get("checklist_yaml_path", ""),
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
            pdf_path = _resolve_asset(files_root, record.get("paper_pdf_path") or metadata["paper_pdf_path"])
            if pdf_path and arxiv_id not in pdf_written:
                _write_bytes(paper_dir / pdf_filename, pdf_path.read_bytes())
                pdf_written.add(arxiv_id)

            checklist_filename = record.get("checklist_filename") or "checklist.yaml"
            checklist_path = _resolve_asset(files_root, record.get("checklist_yaml_path") or metadata["checklist_yaml_path"])
            if checklist_path:
                _write_text(paper_dir / "Factual_accuracy" / checklist_filename, checklist_path.read_text(encoding="utf-8", errors="ignore"))

            # Populate input_dir entry for this promotion.
            if pdf_path:
                promo_dir = dirs["input_dir"] / str(post_id)
                target_pdf = promo_dir / pdf_filename
                if not target_pdf.exists():
                    _write_bytes(target_pdf, pdf_path.read_bytes())

        # Restore figures.
        figure_paths = metadata["figure_path"] or []
        for rel_path in figure_paths:
            dest_rel = Path(rel_path)
            if dest_rel.parts and dest_rel.parts[0] == "files":
                dest_rel = Path(*dest_rel.parts[1:])
            if dest_rel.parts and dest_rel.parts[0] == "figures":
                dest_rel = Path(*dest_rel.parts[1:])

            dest_root = dirs["twitter_dir"] if platform == "TWITTER" else dirs["xhs_dir"]
            dest_path = dest_root / dest_rel

            if str(dest_path) in figure_written:
                continue

            source = _resolve_asset(files_root, rel_path)
            if not source:
                continue

            _write_bytes(dest_path, source.read_bytes())
            figure_written.add(str(dest_path))

    # Write metadata files.
    academic_path = output_dir / "academic_promotion_data.json"
    _write_text(academic_path, _json_dumps(full_records))

    core_subset = [record for record in full_records if record.get("is_core")]
    _write_text(output_dir / "academic_promotion_data_core.json", _json_dumps(core_subset))

    print(f"Reconstructed {processed} records for subset '{subset}'.")
    print("\nReconstruction complete!")
    print(f"Metadata directory: {output_dir}")
    print(f"Input directory: {dirs['input_dir']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and reconstruct the PRBench dataset.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repository ID.")
    parser.add_argument("--subset", default="core", choices=["full", "core"], help="Dataset split to download.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path, help="Destination directory (default: eval/data).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reconstruct(args.repo_id, args.subset, args.output_dir)


if __name__ == "__main__":
    main()
