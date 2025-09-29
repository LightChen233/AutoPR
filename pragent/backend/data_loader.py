# data_loader.py
import asyncio
import aiofiles
from pathlib import Path
import re
from typing import List, Dict
from tqdm.asyncio import tqdm
async def load_plain_text(txt_path: str) -> str:
    """异步地从 .txt 文件加载纯文本内容。"""
    try:
        async with aiofiles.open(txt_path, mode='r', encoding='utf-8') as f:
            return await f.read()
    except Exception as e:
        tqdm.write(f"[!] 读取文本文件 '{txt_path}' 时出错: {e}")
        return ""

def load_paired_image_paths(base_dir: Path) -> List[Dict]:
    """
    递归地扫描 'paired_*' 文件夹，并加载主图和其标题图的路径。
    """
    items = []
    if not base_dir.is_dir():
        tqdm.write(f"[!] 错误: 找不到配对结果的基础文件夹: {base_dir}")
        return items

    tqdm.write(f"[*] 正在从 {base_dir} 递归加载图文对...")
    
    item_dirs = sorted(
        [d for d in base_dir.rglob('paired_*') if d.is_dir()],
        key=lambda p: p.name  
    )

    for item_dir in item_dirs:
        item_files = list(item_dir.glob('*.jpg'))
        if len(item_files) < 2:
            continue

        main_item_path, caption_path = None, None
        for f in item_files:
            if "caption" in f.name:
                caption_path = f
            else:
                main_item_path = f
        
        if main_item_path and caption_path:
            items.append({
                "type": "figure" if "figure" in item_dir.name else "table",
                "item_path": str(main_item_path.resolve()),
                "caption_path": str(caption_path.resolve()),
            })
            
    tqdm.write(f"[*] 加载完成，共找到 {len(items)} 个图文对。")
    return items