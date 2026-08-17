# tesseract_kurulum.py
"""
Tesseract çalıştırılabilir dosyasının yolunu ayarlar. Bu dosyayı import
eden HER modül (sadece `import tesseract_kurulum` yazmak yeterli, başka
bir şey kullanmaya gerek yok) Tesseract'ın nerede olduğunu öğrenmiş olur.
pytesseract.pytesseract.tesseract_cmd, pytesseract modülü üzerinde global
bir ayar olduğu için tek bir yerde ayarlanması yeterli.
"""
import os
import pytesseract

tesseract_yolu = os.environ.get('TESSERACT_CMD')
if tesseract_yolu:
    pytesseract.pytesseract.tesseract_cmd = tesseract_yolu
else:
    # Windows'taki varsayılan Tesseract kurulum yolu
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'