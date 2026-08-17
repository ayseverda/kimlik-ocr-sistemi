import os

os.environ["FLAGS_use_mkldnn"] = "0"

import cv2
from paddleocr import PaddleOCR


print("OCR oluşturuluyor...")

ocr = PaddleOCR(
    lang="tr",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

print("OCR oluşturuldu.")

resim = cv2.imread("referans_kimlik.jpg")

if resim is None:
    print("HATA: referans_kimlik.jpg okunamadı.")
    raise SystemExit

print("Resim okundu:", resim.shape)

print("OCR başlıyor...")

sonuclar = ocr.predict(resim)

print("OCR tamamlandı.")
print("Sonuç sayısı:", len(sonuclar))

for i, sonuc in enumerate(sonuclar):

    print("\n============================")
    print("RESULT:", i)
    print("============================")

    try:
        print(sonuc.json)

    except Exception as e:
        print("JSON alınamadı:", e)
        print(sonuc)