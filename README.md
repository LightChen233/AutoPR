<p align="center">
<h1 align="center"> 🎉 AutoPR: Let's Automate Your Academic Promotion!</h1>
</p>
<p align="center">
  	<a href="https://github.com/LightChen233/AutoPR">
      <img alt="version" src="https://img.shields.io/badge/version-v0.0.1-blue?color=FF8000?color=009922" />
    </a>
    <a href="ttps://github.com/LightChen233/AutoPR">
       <img alt="PRs-Welcome" src="https://img.shields.io/badge/PRs-Welcome-blue" />
  	</a>
   	<a href="https://github.com/LightChen233/AutoPR/stargazers">
       <img alt="stars" src="https://img.shields.io/github/stars/LightChen233/AutoPR" />
  	</a>
  	<a href="https://github.com/LightChen233/AutoPR/network/members">
       <img alt="FORK" src="https://img.shields.io/github/forks/LightChen233/AutoPR?color=FF8000" />
  	</a>
    <a href="https://github.com/LightChen233/AutoPR/issues">
      <img alt="Issues" src="https://img.shields.io/github/issues/LightChen233/AutoPR?color=0088ff"/>
    </a>
    <br />
    
</p>

<p align="center">
  	<b>
    | [<a href="https://arxiv.org/abs/XXX">📝 ArXiv</a>] | [<a href="https://yzweak.github.io/autopr.github.io/">📚 Project Website</a>] | [<a href="https://huggingface.co/datasets/yzweak/PRBench">🤗 PRBench</a>] | [<a href="https://huggingface.co/spaces/yzweak/AutoPR">🔥 PRAgent Demo</a>] |
    </b>
    <br />
</p>


This is the official implementation for **"AUTOPR: LET'S AUTOMATE YOUR ACADEMIC PROMOTION!**".

![](assets/images/title.png)

## 👀 1. Overview
As the volume of peer-reviewed research surges, scholars increasingly rely on social platforms for discovery, while authors invest significant effort in promotion to sustain visibility and citations. This project aims to address that challenge.



We formalize **AutoPR (Automatic Promotion)**, a new task to automatically translate research papers into faithful, engaging, and well-timed public-facing content. To accomplish this, we developed **PRAgent**, a modular agentic framework for automatically transforming research papers into promotional posts optimized for specific social media platforms.

![](assets/images/intro.png)

-----

## 🔥 2. News
- **[2025-10-08]** Our 🔥🔥 **PRAgent** 🔥🔥 and 🔥🔥 **PRBench** 🔥🔥 benchmark is released! You can download the dataset from here.


-----

## 🏅 3. Leaderboard

### 3.1 PRBench-Core
![](assets/images/prbench-core.png)
![](assets/images/prbench-core-1.png)

### 3.2 PRBench-Full
![](assets/images/prbench-full.png)

-----

## 🛠️ 4. Installation & Configuration

### 4.1 Environment Installation

1.  Create and activate a Conda environment (recommended):

    ```bash
    conda create -n autopr python=3.11
    conda activate autopr
    ```

2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```


### 4.2 Configuration

Before running the code, you need to configure your Large Language Model (LLM) API keys and endpoints.

First, copy the example `.env.example` file to a new `.env` file:

```bash
cp .env.example .env
```

Then, edit the `.env` file with your API credentials:

```python
# Main API Base URL for text and vision models (e.g., OpenAI, Qwen, etc.)
OPENAI_API_BASE="https://api.openai.com/v1"
# Your API Key
OPENAI_API_KEY="sk-..."
```

The scripts will automatically load these environment variables.

-----

## ⚡ 5. PRBench Evaluation

The entire workflow, from generation to evaluation, is managed through simple shell scripts.

### 5.1 Step 1: Preparation

Download the PRBench dataset from Hugging Face Hub. You can choose to download the full dataset or the core subset.

```bash
python download_and_reconstruct_prbench.py \
    --repo-id yzweak/PRBench \
    --subset core \ # or "full"
    --output-dir eval
```

You also need to download the [DocLayout-YOLO](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/blob/main/doclayout_yolo_docstructbench_imgsz1024.pt) model. You can specify the path to the model using the `--model-path` argument in the generation script.

### 5.2 Step 2: Evaluate Post Quality

After generation, use the evaluation script to assess the quality of the posts in your output directory.

```bash
chmod +x scripts/run_eval.sh
./scripts/run_eval.sh
```

### 5.3 Step 3: Calculate and View Metrics

Finally, run the calculation script to aggregate the raw evaluation data into a formatted results table.

```bash
chmod +x scripts/calc_results.sh
./scripts/calc_results.sh
```

## 🕹️ 6. PRAgent Generation

![](assets/images/pragent.png)
### 6.1 Step 1: Preparation

You need to download the [DocLayout-YOLO](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/blob/main/doclayout_yolo_docstructbench_imgsz1024.pt) model. When running the generation script, you can specify the path to the model using the `--model-path` argument.

for example:
```bash
python3 pragent/run.py --model-path /path/to/your/model.pt ...
```

### 6.2 Step 2: Generate Promotional Posts (PRAgent)



First, prepare your input directory. The script automatically determines the target platform based on the **folder name**:

* **Numeric** folder name -\> **Twitter (English)**
* **Alphanumeric** folder name -\> **Xiaohongshu (Chinese)**

<!-- end list -->

```python
/path/to/your/papers/
├── 12345/               # Numeric -> will generate a Twitter-style post in English
│   └── paper.pdf
└── some_paper_name/     # Alphanumeric -> will generate a Xiaohongshu-style post in Chinese
    └── paper.pdf
```

If you have run ``download_and_reconstruct.py``, you can use the ``papers`` folder as input

Next, configure and run the generation script.
```bash
chmod +x scripts/run_pragent.sh
./script/run_generation.sh
```


### PRAgent Case
**Baseline:**
![](assets/images/case-1.png)

**PRAgent:**
![](assets/images/case-2.png)

## ☎️ Contact
If interested in our work, please contact us at:
- Qiguang Chen: charleschen2333@gmail.com
- Zheng Yan: xxx

## 🎁 Citation
```
@misc{chen2025autopr,
      title={AutoPR: Let's Automate Your Academic Promotion!}, 
      author={Qiguang Chen and Zheng Yan and Mingda Yang and Libo Qin and Yixin Yuan and Hanjing Li and Jinhao Liu and Yiyan Ji and Dengyun Peng and Jiannan Guan and Mengkang Hu and Yantao Du and Wanxiang Che},
      year={2025},
}
```
