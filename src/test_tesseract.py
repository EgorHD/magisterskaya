from pathlib import Path
from core.io.loader import load_document
from core.ocr.factory import get_ocr_engine

doc = load_document(Path("C:\Users\egork\OneDrive\Рабочий стол\тестик.pdf"), pdf_dpi=300)
engine = get_ocr_engine()
res = engine.recognize_document(doc)
print(res.text()[:2000])