import base64
import aiofiles
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image

QUALITY_SETTINGS = {
    'high': {'max_size': (2048, 2048), 'jpeg_quality': 95},
    'medium': {'max_size': (1024, 1024), 'jpeg_quality': 85},
    'low': {'max_size': (768, 768), 'jpeg_quality': 75},
    'very_low': {'max_size': (512, 512), 'jpeg_quality': 70}
}

async def read_and_preprocess_image_as_base64(image_path: str, quality: str = 'high') -> str | None:
    """
    异步读取、根据指定的质量等级预处理图片，并返回Base64编码的字符串。

    Args:
        image_path: 图片文件的路径。
        quality: 质量等级 ('high', 'medium', 'low')。
    
    Returns:
        经过预处理和Base64编码的图片字符串，或在出错时返回 None。
    """
    if quality not in QUALITY_SETTINGS:
        raise ValueError(f"Invalid quality setting: {quality}. Must be one of {list(QUALITY_SETTINGS.keys())}")

    settings = QUALITY_SETTINGS[quality]
    max_size = settings['max_size']
    jpeg_quality = settings['jpeg_quality']

    try:
        async with aiofiles.open(image_path, "rb") as image_file:
            image_data = await image_file.read()
            
            with Image.open(BytesIO(image_data)) as img:
                img.thumbnail(max_size)
                
                buffer = BytesIO()
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                img.save(buffer, format="JPEG", quality=jpeg_quality)
                
                encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return encoded_string
                
    except FileNotFoundError:
        print(f"ERROR: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"ERROR: Could not read or preprocess image at {image_path}: {e}")
        return None

# --- NEW ---
async def extract_text_from_pdf(pdf_path: str) -> str | None:
    """Extracts all text from a PDF file."""
    if not pdf_path:
        return None
    try:
        text = ""
        # PyMuPDF is not async, run in a thread to avoid blocking if necessary,
        # but for this script, direct call is acceptable.
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
                break  # Remove this line if you want to extract text from all pages
        return text
    except FileNotFoundError:
        print(f"ERROR: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to extract text from PDF {pdf_path}: {e}")
        return None