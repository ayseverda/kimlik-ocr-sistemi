import cv2
import re
import time
import easyocr
import numpy as np

from difflib import SequenceMatcher

from kimlikler.kimlikler_tc import tc_bul
from kimlikler.kimlikler_eski_tc import eski_tc_bilgilerini_bul
from kimlikler.kimlikler_gocmen import gocmen_bilgilerini_bul


# =========================================================
# EASYOCR — LAZY SINGLETON
# =========================================================
#
# ÖNEMLİ: 'reader = easyocr.Reader(...)' MODÜL SEVİYESİNDE çalıştırılırsa,
# bu modül import edilir edilmez (yani web.py açılır açılmaz, kullanıcı daha
# hiçbir dosya yüklemeden) model indirmeye başlar. İlk deploy'da bu indirme
# birkaç dakika sürebiliyor; Streamlit Cloud script tamamen çalışıp bir sayfa
# döndürene kadar health-check bekliyor, bu da health-check'i zaman aşımına
# uğratıp "Oh no. Error running app." hatasına yol açıyor (gerçekte yaşandı).
# Bu yüzden reader SADECE ilk gerçek OCR çağrısında (kullanıcı bir dosya
# işlettiğinde) oluşturuluyor; app hemen açılıyor, indirme yalnızca o an ve
# yalnızca bir kez gerçekleşiyor (cache'lenir).

_READER = None


def reader_getir():
    global _READER
    if _READER is None:
        _READER = easyocr.Reader(["tr", "en"], gpu=False)
    return _READER


# =========================================================
# NORMALIZE
# =========================================================

def normalize_text(text):
    text = str(text).upper().strip()
    text = text.translate(str.maketrans({
        "İ": "I", "Ş": "S", "Ğ": "G", "Ü": "U", "Ö": "O", "Ç": "C",
    }))
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def isim_temizle(text):
    if not text:
        return ""
    text = str(text).upper().strip()
    text = re.sub(r"[^A-ZÇĞİIÖŞÜ\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def benzerlik(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


# =========================================================
# OCR
# =========================================================

def easyocr_oku(resim, offset_x=0, offset_y=0):
    baslangic = time.perf_counter()
    bulunanlar = []

    try:
        sonuclar = reader_getir().readtext(resim, detail=1, paragraph=False, decoder="greedy")
    except Exception as e:
        print("EasyOCR hatası:", repr(e))
        return [], 0.0

    for sonuc in sonuclar:
        try:
            box, text, conf = sonuc
        except Exception:
            continue

        text = str(text).strip()
        if not text:
            continue

        box = np.asarray(box, dtype=np.float32)
        bulunanlar.append({
            "text": text, "norm": normalize_text(text), "conf": float(conf),
            "x1": int(box[:, 0].min()) + offset_x, "y1": int(box[:, 1].min()) + offset_y,
            "x2": int(box[:, 0].max()) + offset_x, "y2": int(box[:, 1].max()) + offset_y,
        })

    return bulunanlar, time.perf_counter() - baslangic


# =========================================================
# ESKİ TC — ÖZEL OCR
# =========================================================

def _roi_ocr_oku(roi, offset_x, offset_y, **kwargs):
    """easyocr_oku_eski_tc / _fallback ortak gövdesi: roi üzerinde OCR
    çalıştırır, kutuları orijinal kart koordinatına taşır."""
    try:
        sonuclar = reader_getir().readtext(roi, detail=1, paragraph=False, decoder="greedy", **kwargs)
    except Exception as e:
        print("Eski TC EasyOCR hatası:", repr(e))
        return []

    bulunanlar = []
    for sonuc in sonuclar:
        try:
            box, text_ocr, conf = sonuc
        except Exception:
            continue

        text_ocr = str(text_ocr).strip()
        if not text_ocr:
            continue

        box = np.asarray(box, dtype=np.float32)
        bulunanlar.append({
            "text": text_ocr, "norm": normalize_text(text_ocr), "conf": float(conf),
            "x1": int(box[:, 0].min()) + offset_x, "y1": int(box[:, 1].min()) + offset_y,
            "x2": int(box[:, 0].max()) + offset_x, "y2": int(box[:, 1].max()) + offset_y,
        })
    return bulunanlar


def easyocr_oku_eski_tc(resim):
    """HIZLI ESKİ TC OCR: ilk aşamada SADECE TC + SOYADI + ADI bölgesini okur.
    İkinci/geniş OCR yalnızca parser gerçekten bir alanı bulamazsa
    eski_tc_bilgilerini_oku() içinde çağrılır."""
    baslangic = time.perf_counter()
    if resim is None:
        return [], 0.0

    h, w = resim.shape[:2]
    x1, x2 = int(w * 0.025), int(w * 0.985)
    # Perspektif sonrası eski nüfus cüzdanında TC + SOYADI + ADI alanlarını kapsayan bölüm.
    y1, y2 = int(h * 0.47), int(h * 0.82)

    roi = resim[y1:y2, x1:x2]
    if roi.size == 0:
        return [], 0.0

    bulunanlar = _roi_ocr_oku(
        roi, x1, y1,
        text_threshold=0.48, low_text=0.28, link_threshold=0.32,
        min_size=8, width_ths=0.75, ycenter_ths=0.5, add_margin=0.05,
    )
    return bulunanlar, time.perf_counter() - baslangic


def easyocr_oku_eski_tc_fallback(resim):
    """SADECE GEREKİRSE çalışan geniş eski-TC OCR: ilk hızlı ROI sonucunda
    TC/AD/SOYAD alanlarından biri bulunamazsa çağrılır."""
    baslangic = time.perf_counter()
    if resim is None:
        return [], 0.0

    h, w = resim.shape[:2]
    x1, x2 = int(w * 0.015), int(w * 0.99)
    # Biraz daha geniş alan: farklı basım/kırpma varyasyonlarına karşı fallback.
    y1, y2 = int(h * 0.40), int(h * 0.90)

    roi = resim[y1:y2, x1:x2]
    if roi.size == 0:
        return [], 0.0

    olcek = 1.10  # sadece fallback'te hafif büyütme
    roi_ocr = cv2.resize(roi, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_LINEAR)

    try:
        sonuclar = reader_getir().readtext(
            roi_ocr, detail=1, paragraph=False, decoder="greedy",
            text_threshold=0.45, low_text=0.25, link_threshold=0.30,
            min_size=8, width_ths=0.75, ycenter_ths=0.5, add_margin=0.06,
        )
    except Exception as e:
        print("Eski TC fallback EasyOCR hatası:", repr(e))
        return [], 0.0

    bulunanlar = []
    for sonuc in sonuclar:
        try:
            box, text_ocr, conf = sonuc
        except Exception:
            continue

        text_ocr = str(text_ocr).strip()
        if not text_ocr:
            continue

        box = np.asarray(box, dtype=np.float32)
        bulunanlar.append({
            "text": text_ocr, "norm": normalize_text(text_ocr), "conf": float(conf),
            "x1": int(box[:, 0].min() / olcek) + x1, "y1": int(box[:, 1].min() / olcek) + y1,
            "x2": int(box[:, 0].max() / olcek) + x1, "y2": int(box[:, 1].max() / olcek) + y1,
        })

    return bulunanlar, time.perf_counter() - baslangic


# =========================================================
# TÜRKÇE SECOND PASS
# =========================================================

def turkce_second_pass_gerekli_mi(text):
    if not text:
        return False
    norm = normalize_text(text)
    return any(harf in norm for harf in ["S", "I", "U", "O", "C", "G"])


def isim_kutusunu_hafif_genislet(kart, item):
    if item is None:
        return None

    h, w = kart.shape[:2]
    x1, y1 = max(0, item["x1"] - 3), max(0, item["y1"] - 6)
    x2, y2 = min(w, item["x2"] + 3), min(h, item["y2"] + 4)

    roi = kart[y1:y2, x1:x2]
    return roi if roi.size else None


def turkce_harf_iyilestir(kart, item):
    if item is None:
        return None, 0.0, 0.0

    ilk_text = isim_temizle(item.get("text", ""))
    ilk_conf = float(item.get("conf", 0.0))

    if not ilk_text:
        return None, ilk_conf, 0.0
    if not turkce_second_pass_gerekli_mi(ilk_text):
        return ilk_text, ilk_conf, 0.0

    baslangic = time.perf_counter()
    roi = isim_kutusunu_hafif_genislet(kart, item)
    if roi is None:
        return ilk_text, ilk_conf, 0.0

    # Küçük value crop'ında recognition için yeterli.
    roi = cv2.resize(roi, None, fx=1.65, fy=1.65, interpolation=cv2.INTER_LINEAR)

    try:
        # HIZ KRİTİK: kutu zaten belli olduğu için readtext() ile tekrar text
        # detector çalıştırmıyoruz. reader.recognize() yalnızca verilen
        # yatay kutuyu recognize eder — çok daha hızlı.
        roi_h, roi_w = roi.shape[:2]
        sonuclar = reader_getir().recognize(
            roi, horizontal_list=[[0, roi_w, 0, roi_h]], free_list=[],
            decoder="greedy", beamWidth=5, batch_size=1, workers=0,
            allowlist=None, blocklist=None, detail=1, rotation_info=None,
            paragraph=False, contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003,
        )
    except Exception:
        try:
            sonuclar = reader_getir().readtext(roi, detail=1, paragraph=False, decoder="greedy")
        except Exception:
            return ilk_text, ilk_conf, time.perf_counter() - baslangic

    sure = time.perf_counter() - baslangic
    if not sonuclar:
        return ilk_text, ilk_conf, sure

    parcalar = []
    for sonuc in sonuclar:
        try:
            _, ikinci_text, ikinci_conf = sonuc
        except Exception:
            continue
        ikinci_text = isim_temizle(ikinci_text)
        if ikinci_text:
            parcalar.append((ikinci_text, float(ikinci_conf)))

    if not parcalar:
        return ilk_text, ilk_conf, sure

    ikinci_text = " ".join(x[0] for x in parcalar)
    ikinci_conf = sum(x[1] for x in parcalar) / len(parcalar)

    # Ana OCR ile aynı normalize edilmiş kelimeyse, Türkçe karakter açısından
    # daha iyi olanı seç.
    if normalize_text(ikinci_text) == normalize_text(ilk_text):
        ilk_tr = len(re.findall(r"[ÇĞİÖŞÜ]", ilk_text))
        ikinci_tr = len(re.findall(r"[ÇĞİÖŞÜ]", ikinci_text))

        if ikinci_tr > ilk_tr:
            return ikinci_text, ikinci_conf, sure
        # Aynı yazım ailesindeyse confidence ciddi yüksekse ikinci sonucu da kullanabiliriz.
        if ikinci_conf > ilk_conf + 0.12:
            return ikinci_text, ikinci_conf, sure

    return ilk_text, ilk_conf, sure


# =========================================================
# YENİ TC
# =========================================================

SOYAD_LABEL_HEDEFLERI = ["SOYADI SURNAME", "SURNAME", "SOYADI"]
AD_LABEL_HEDEFLERI = [
    "ADI GIVEN NAME S", "ADI GIVEN NAMES", "ADI GIVEN NAME",
    "GIVEN NAME S", "GIVEN NAMES", "GIVEN NAME",
]


def fuzzy_label_bul(ocr_sonuclari, hedefler, esik=0.40):
    en_iyi_item, en_iyi_skor = None, 0.0
    for item in ocr_sonuclari:
        for hedef in hedefler:
            skor = benzerlik(item["text"], hedef)
            if skor > en_iyi_skor:
                en_iyi_skor, en_iyi_item = skor, item
    return en_iyi_item if en_iyi_skor >= esik else None


def buyuk_harf_orani(text):
    harfler = [c for c in str(text) if c.isalpha()]
    if not harfler:
        return 0.0
    return sum(1 for c in harfler if c.isupper()) / len(harfler)


def tc_isim_adayi_mi(item):
    text = str(item.get("text", ""))
    if any(c.isdigit() for c in text):
        return False

    temiz = isim_temizle(text)
    if len(re.sub(r"[^A-ZÇĞİÖŞÜ]", "", temiz)) < 2:
        return False

    norm = normalize_text(text)
    yasaklar = [
        "SOYADI", "SURNAME", "GIVEN", "NAME", "KIMLIK", "IDENTITY",
        "BIRTH", "GENDER", "NATIONALITY", "DOCUMENT",
    ]
    if any(x in norm for x in yasaklar):
        return False

    return buyuk_harf_orani(text) >= 0.75


def label_altindaki_degeri_bul(ocr_sonuclari, label, sonraki_label=None):
    if label is None:
        return None

    adaylar = []
    for item in ocr_sonuclari:
        if item is label or not tc_isim_adayi_mi(item):
            continue

        dy = item["y1"] - label["y2"]
        if dy < -12 or dy > 150:
            continue
        if sonraki_label is not None and item["y1"] >= sonraki_label["y1"]:
            continue

        dx = abs(item["x1"] - label["x1"])
        if dx > 240:
            continue

        skor = -dy * 2 - dx * 0.2 + float(item.get("conf", 0)) * 20
        adaylar.append((skor, item))

    if not adaylar:
        return None
    return max(adaylar, key=lambda x: x[0])[1]


# =========================================================
# DEBUG
# =========================================================

def debug_resmi_olustur(kart, alanlar):
    debug = kart.copy()
    for item, etiket, renk in alanlar:
        if item is None:
            continue
        cv2.rectangle(debug, (item["x1"], item["y1"]), (item["x2"], item["y2"]), renk, 4)
        cv2.putText(
            debug, etiket, (item["x1"], max(25, item["y1"] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2,
        )
    return debug


# =========================================================
# TC OKU
# =========================================================

def tc_bilgilerini_oku(kart, sayfa_no=None, debug=False):
    h, w = kart.shape[:2]
    x1, x2 = int(w * 0.02), int(w * 0.70)
    y1, y2 = int(h * 0.15), int(h * 0.70)
    roi = kart[y1:y2, x1:x2]

    ocr, sure = easyocr_oku(roi, x1, y1)
    tc_no, tc_item = tc_bul(ocr)

    soyad_label = fuzzy_label_bul(ocr, SOYAD_LABEL_HEDEFLERI)
    ad_label = fuzzy_label_bul(ocr, AD_LABEL_HEDEFLERI)

    soyad_item = label_altindaki_degeri_bul(ocr, soyad_label, ad_label)
    ad_item = label_altindaki_degeri_bul(ocr, ad_label)

    ad = isim_temizle(ad_item["text"]) if ad_item else "Bulunamadi"
    soyad = isim_temizle(soyad_item["text"]) if soyad_item else "Bulunamadi"
    ad_conf = float(ad_item.get("conf", 0)) if ad_item else 0
    soyad_conf = float(soyad_item.get("conf", 0)) if soyad_item else 0

    debug_resmi = None
    if debug:
        debug_resmi = debug_resmi_olustur(kart, [
            (tc_item, "TC", (0, 255, 0)),
            (soyad_item, "SOYAD", (0, 255, 255)),
            (ad_item, "AD", (255, 255, 0)),
        ])

    return {
        "sayfa_no": sayfa_no, "belge_tipi": "tc", "tc_no": tc_no, "ad": ad, "soyad": soyad,
        "ad_conf": ad_conf, "soyad_conf": soyad_conf, "belge_gecerli": None, "bitis_tarihi": "",
        "debug_resmi": debug_resmi, "tum_ocr": ocr if debug else [], "ocr_suresi": sure,
    }


# =========================================================
# ESKİ TC
# =========================================================

def eski_tc_bilgilerini_oku(kart, sayfa_no=None, debug=False):
    # ---- 1. Hızlı OCR ---------------------------------------------------
    ocr, ilk_ocr_suresi = easyocr_oku_eski_tc(kart)
    sonuc = eski_tc_bilgilerini_bul(ocr)

    # ---- 2. Gerçek parser sonucuna göre fallback (hepsi bulunduysa asla çalışmaz) ----
    ilk_tam = (
        sonuc.get("tc_no", "Bulunamadi") != "Bulunamadi"
        and sonuc.get("ad", "Bulunamadi") != "Bulunamadi"
        and sonuc.get("soyad", "Bulunamadi") != "Bulunamadi"
    )
    fallback_ocr_suresi = 0.0

    if not ilk_tam:
        ocr_fallback, fallback_ocr_suresi = easyocr_oku_eski_tc_fallback(kart)

        if ocr_fallback:
            fallback_sonuc = eski_tc_bilgilerini_bul(ocr_fallback)

            def alan_sayisi(x):
                return sum([
                    x.get("tc_no", "Bulunamadi") != "Bulunamadi",
                    x.get("ad", "Bulunamadi") != "Bulunamadi",
                    x.get("soyad", "Bulunamadi") != "Bulunamadi",
                ])

            if alan_sayisi(fallback_sonuc) > alan_sayisi(sonuc):
                sonuc, ocr = fallback_sonuc, ocr_fallback
            elif alan_sayisi(fallback_sonuc) == alan_sayisi(sonuc):
                # Aynı sayıda alan varsa ortalama confidence daha iyi olan sonucu kullan.
                ilk_conf = float(sonuc.get("ad_conf", 0.0)) + float(sonuc.get("soyad_conf", 0.0))
                fallback_conf = (
                    float(fallback_sonuc.get("ad_conf", 0.0)) + float(fallback_sonuc.get("soyad_conf", 0.0))
                )
                if fallback_conf > ilk_conf + 0.08:
                    sonuc, ocr = fallback_sonuc, ocr_fallback

    # ---- 3. Second pass — sadece düşük conf + şüpheli harf --------------
    ad_item, soyad_item = sonuc.get("ad_item"), sonuc.get("soyad_item")
    ad_ham, soyad_ham = sonuc.get("ad", "Bulunamadi"), sonuc.get("soyad", "Bulunamadi")
    ad_ham_conf = float(sonuc.get("ad_conf", 0.0))
    soyad_ham_conf = float(sonuc.get("soyad_conf", 0.0))

    if ad_item is not None and ad_ham != "Bulunamadi" and ad_ham_conf < 0.58 and turkce_second_pass_gerekli_mi(ad_ham):
        ad_second, ad_second_conf, ad_second_sure = turkce_harf_iyilestir(kart, ad_item)
    else:
        ad_second, ad_second_conf, ad_second_sure = ad_ham, ad_ham_conf, 0.0

    if (
        soyad_item is not None and soyad_ham != "Bulunamadi"
        and soyad_ham_conf < 0.58 and turkce_second_pass_gerekli_mi(soyad_ham)
    ):
        soyad_second, soyad_second_conf, soyad_second_sure = turkce_harf_iyilestir(kart, soyad_item)
    else:
        soyad_second, soyad_second_conf, soyad_second_sure = soyad_ham, soyad_ham_conf, 0.0

    second_pass_suresi = ad_second_sure + soyad_second_sure

    # ---- 4. Son değerler --------------------------------------------------
    ad = ad_second if ad_second else ad_ham
    soyad = soyad_second if soyad_second else soyad_ham
    ad_conf = float(ad_second_conf if ad_second else ad_ham_conf)
    soyad_conf = float(soyad_second_conf if soyad_second else soyad_ham_conf)
    tc_no = sonuc.get("tc_no", "Bulunamadi")

    bulunan = sum([tc_no != "Bulunamadi", ad != "Bulunamadi", soyad != "Bulunamadi"])
    if bulunan == 3:
        guven = "yuksek" if (ad_conf >= 0.50 and soyad_conf >= 0.50) else "orta"
    elif bulunan > 0:
        guven = "orta"
    else:
        guven = "dusuk"

    # ---- 5. Debug -----------------------------------------------------
    debug_resmi = None
    if debug:
        debug_resmi = debug_resmi_olustur(kart, [
            (sonuc.get("tc_item"), "TC", (0, 255, 0)),
            (sonuc.get("soyad_item"), "SOYAD", (0, 255, 255)),
            (sonuc.get("ad_item"), "AD", (255, 255, 0)),
        ])

    ana_ocr_suresi = ilk_ocr_suresi + fallback_ocr_suresi

    return {
        "sayfa_no": sayfa_no, "belge_tipi": "eski_tc", "tc_no": tc_no, "ad": ad, "soyad": soyad,
        "guven": guven, "ad_conf": ad_conf, "soyad_conf": soyad_conf,
        "baslangic_tarihi": "", "bitis_tarihi": "", "belge_gecerli": None, "gecerlilik_durumu": None,
        "ocr_suresi": ana_ocr_suresi + second_pass_suresi, "ana_ocr_suresi": ana_ocr_suresi,
        "ilk_ocr_suresi": ilk_ocr_suresi, "fallback_ocr_suresi": fallback_ocr_suresi,
        "second_pass_suresi": second_pass_suresi,
        "debug_resmi": debug_resmi, "tum_ocr": ocr if debug else [],
    }


# =========================================================
# GÖÇMEN
# =========================================================

def gocmen_bilgilerini_oku(kart, sayfa_no=None, debug=False):
    ocr, sure = easyocr_oku(kart)
    sonuc = gocmen_bilgilerini_bul(ocr)

    debug_resmi = None
    if debug:
        alanlar = [
            (sonuc.get("kimlik_no_item"), "YKN", (0, 255, 0)),
            (sonuc.get("soyad_item"), "SOYAD", (0, 255, 255)),
            (sonuc.get("ad_item"), "AD", (255, 255, 0)),
        ]
        tarih_items = sonuc.get("gecerlilik_items", [])
        if len(tarih_items) >= 1:
            alanlar.append((tarih_items[0], "BASLANGIC", (255, 0, 255)))
        if len(tarih_items) >= 2:
            alanlar.append((tarih_items[1], "BITIS", (180, 0, 255)))
        debug_resmi = debug_resmi_olustur(kart, alanlar)

    return {
        "sayfa_no": sayfa_no, "belge_tipi": "gocmen", "tc_no": sonuc["kimlik_no"],
        "ad": sonuc["ad"], "soyad": sonuc["soyad"],
        "ad_conf": sonuc["ad_conf"], "soyad_conf": sonuc["soyad_conf"],
        "baslangic_tarihi": sonuc["baslangic_tarihi"], "bitis_tarihi": sonuc["bitis_tarihi"],
        "belge_gecerli": sonuc["belge_gecerli"], "gecerlilik_durumu": sonuc["gecerlilik_durumu"],
        "debug_resmi": debug_resmi, "tum_ocr": ocr if debug else [], "ocr_suresi": sure,
    }


# =========================================================
# ANA
# =========================================================

def bilgileri_cimbizla(orijinal_kart, sayfa_no=None, debug=False, belge_tipi="tc"):
    if orijinal_kart is None:
        return {
            "sayfa_no": sayfa_no, "belge_tipi": belge_tipi, "tc_no": "Bulunamadi",
            "ad": "Bulunamadi", "soyad": "Bulunamadi", "ad_conf": 0.0, "soyad_conf": 0.0,
            "belge_gecerli": None, "bitis_tarihi": "", "debug_resmi": None, "tum_ocr": [],
        }

    if belge_tipi == "gocmen":
        return gocmen_bilgilerini_oku(orijinal_kart, sayfa_no, debug)
    if belge_tipi == "eski_tc":
        return eski_tc_bilgilerini_oku(orijinal_kart, sayfa_no, debug)
    return tc_bilgilerini_oku(orijinal_kart, sayfa_no, debug)