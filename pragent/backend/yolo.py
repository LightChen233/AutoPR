# yolo.py
import os
from PIL import Image
from doclayout_yolo import YOLOv10
from tqdm.asyncio import tqdm
CLASS_NAMES = {
    0: "title",
    1: "plain_text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption_above",
    7: "table_caption_below",
    8: "formula",
    9: "formula_caption",
}

def extract_and_save_layout_components(image_path, model_path, save_base_dir="./cropped_results", imgsz=1024, conf=0.2, device="cuda:0"):
    """
    从图像中提取文档布局组件，并按类别保存截图。

    Args:
        image_path (str): 输入图像路径
        model_path (str): 模型权重路径（.pt）
        save_base_dir (str): 保存截图的根目录
        imgsz (int): 输入图像的尺寸（会缩放到这个大小）
        conf (float): 检测框的置信度阈值
        device (str): 使用的计算设备，比如 'cuda:0' 或 'cpu'
    """
    model = YOLOv10(model_path)
    image = Image.open(image_path)
    det_results = model.predict(image_path, imgsz=imgsz, conf=conf, device=device)

    result = det_results[0]
    boxes = result.boxes.xyxy.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()

    for idx, (box, cls_id, score) in enumerate(zip(boxes, classes, scores)):
        cls_id = int(cls_id)
        class_name = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
        save_dir = os.path.join(save_base_dir, class_name)
        os.makedirs(save_dir, exist_ok=True)
        x1, y1, x2, y2 = map(int, box)
        cropped = image.crop((x1, y1, x2, y2))
        if cropped.mode == 'RGBA':
            cropped = cropped.convert('RGB')
        save_path = os.path.join(save_dir, f"{class_name}_{idx}_score{score:.2f}.jpg")
        cropped.save(save_path)
    tqdm.write(f"共保存 {len(boxes)} 张截图，按类别分别保存在 {save_base_dir}/")