
'''
This script downloads the PRBench dataset from the Hugging Face Hub and reconstructs the original
file structure required by the project.

It supports downloading the full dataset or a 'core' subset and can merge data from multiple runs.

Usage:
python download_and_reconstruct.py [--subset core]
'''
import os
import json
import shutil
import argparse
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

# The repository ID is now hardcoded.
REPO_ID = "yzweak/PRbench"

def reconstruct_dataset(repo_id, subset, output_dir):
    print(f"Starting dataset reconstruction...")
    print(f"  - Repo ID: {repo_id}")
    print(f"  - Subset: {subset}")
    print(f"  - Output Dir: {output_dir}")

    # --- Step 1: Setup directories ---
    # Directories are now created with exist_ok=True to support merging.
    base_dir = os.path.abspath(os.path.join(output_dir, os.pardir, os.pardir))
    input_dir = os.path.join(base_dir, "input_dir")
    
    fine_grained_dir = os.path.join(output_dir, "Fine_grained_evaluation")
    twitter_dir = os.path.join(output_dir, "twitter_figure")
    xhs_dir = os.path.join(output_dir, "xhs_figure")

    os.makedirs(fine_grained_dir, exist_ok=True)
    os.makedirs(twitter_dir, exist_ok=True)
    os.makedirs(xhs_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    print(f"Ensured output directories exist.")
    print(f"Generated input directory at: {input_dir}")

    # --- Step 2: Download and filter metadata ---
    print("Downloading metadata (data.jsonl)...")
    try:
        jsonl_path = hf_hub_download(repo_id=repo_id, filename="data.jsonl", repo_type="dataset")
    except Exception as e:
        print(f"Error downloading data.jsonl: {e}")
        print("Please ensure the repository ID is correct and you have the necessary permissions.")
        return

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        all_records = [json.loads(line) for line in f]

    if subset == 'core':
        print("Filtering for 'core' subset...")
        records_to_process = [r for r in all_records if r.get('is_core', False)]
    else:
        records_to_process = all_records

    print(f"Processing {len(records_to_process)} records.")

    # --- Step 3: Download and place files ---
    
    required_arxiv_ids = {r['arxiv_id'] for r in records_to_process if r.get('arxiv_id')}
    required_figure_ids = {r['id'] for r in records_to_process if r.get('platform_source') in ["TWITTER", "XHS_NOTE"] and r.get('id')}

    print(f"Downloading {len(required_arxiv_ids)} unique papers and checklists...")
    for arxiv_id in tqdm(required_arxiv_ids, desc="Papers/Checklists"):
        paper_dir = os.path.join(fine_grained_dir, arxiv_id)
        os.makedirs(paper_dir, exist_ok=True)

        pdf_local_path = os.path.join(paper_dir, f"{arxiv_id}.pdf")
        if not os.path.exists(pdf_local_path):
            try:
                pdf_path_on_hub = f"files/papers/{arxiv_id}.pdf"
                hf_hub_download(
                    repo_id=repo_id, 
                    filename=pdf_path_on_hub, 
                    repo_type="dataset",
                    local_dir=paper_dir,
                    local_dir_use_symlinks=False
                )
                os.rename(os.path.join(paper_dir, pdf_path_on_hub), pdf_local_path)
            except Exception:
                pass

        checklist_local_path = os.path.join(paper_dir, "Factual_accuracy", "checklist.yaml")
        if not os.path.exists(checklist_local_path):
            try:
                checklist_path_on_hub = f"files/checklists/{arxiv_id}.yaml"
                checklist_dest_dir = os.path.join(paper_dir, "Factual_accuracy")
                os.makedirs(checklist_dest_dir, exist_ok=True)
                hf_hub_download(
                    repo_id=repo_id, 
                    filename=checklist_path_on_hub, 
                    repo_type="dataset",
                    local_dir=checklist_dest_dir,
                    local_dir_use_symlinks=False
                )
                os.rename(os.path.join(checklist_dest_dir, checklist_path_on_hub), checklist_local_path)
            except Exception:
                pass
        
        # Populate input_dir
        if os.path.exists(pdf_local_path):
            associated_records = [r for r in records_to_process if r.get('arxiv_id') == arxiv_id and r.get('id')]
            for record in associated_records:
                input_subdir = os.path.join(input_dir, record['id'])
                os.makedirs(input_subdir, exist_ok=True)
                shutil.copy(pdf_local_path, os.path.join(input_subdir, f"{arxiv_id}.pdf"))

    print(f"Downloading {len(required_figure_ids)} unique figure sets...")
    figure_records = [r for r in records_to_process if r.get('id') in required_figure_ids]
    
    for record in tqdm(figure_records, desc="Figures"):
        figure_id = record['id']
        platform = record['platform_source']
        dest_dir = twitter_dir if platform == "TWITTER" else xhs_dir
        figure_dest_path = os.path.join(dest_dir, figure_id)

        if not os.path.exists(figure_dest_path):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    allow_patterns=f"files/figures/{figure_id}/*",
                    local_dir=figure_dest_path,
                    local_dir_use_symlinks=False,
                    ignore_patterns=["*.md", ".gitattributes"],
                )
                nested_fig_dir = os.path.join(figure_dest_path, "files", "figures", figure_id)
                if os.path.exists(nested_fig_dir):
                    for f_name in os.listdir(nested_fig_dir):
                        shutil.move(os.path.join(nested_fig_dir, f_name), os.path.join(figure_dest_path, f_name))
                    shutil.rmtree(os.path.join(figure_dest_path, "files"))
            except Exception:
                pass

    # --- Step 4: Recreate promotion JSON files ---
    print("Recreating promotion JSON files...")
    with open(os.path.join(output_dir, "academic_promotion_data.json"), 'w', encoding='utf-8') as f_full:
        for record in all_records:
            f_full.write(json.dumps(record) + '\n')

    core_records = [r for r in all_records if r.get('is_core', False)]
    with open(os.path.join(output_dir, "academic_promotion_data_core.json"), 'w', encoding='utf-8') as f_core:
        for record in core_records:
            f_core.write(json.dumps(record) + '\n')

    print("\nDataset reconstruction complete!")
    print(f"Data is available at: {output_dir}")
    print(f"Input directory generated at: {input_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and reconstruct the PRBench dataset from Hugging Face Hub.")
    parser.add_argument("--subset", type=str, default="full", choices=["full", "core"], help="The subset to download ('full' or 'core'). Defaults to 'full'.")
    parser.add_argument("--output_dir", type=str, default=os.path.join("eval", "data"), help="The local directory to reconstruct the data in. Defaults to 'eval/data'.")
    
    args = parser.parse_args()
    
    reconstruct_dataset(REPO_ID, args.subset, args.output_dir)
