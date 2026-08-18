import cv2
import re
import time
import easyocr
import numpy as np

from difflib import SequenceMatcher


# =========================================================
# EASYOCR
# =========================================================

reader = easyocr.Reader(["tr", "en"], gpu=False)


# =========================================================
# METİN YARDIMCILARI
# =========================================================

def normalize_text(text):
    text = str(text).upper().strip()
    text = text.translate(str.maketrans({
        "İ": "I", "Ş": "S", "Ğ": "G", "Ü": "U", "Ö": "O", "Ç": "C"
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
# BÜYÜK HARF ORANI
# =========================================================

def buyuk_harf_orani(text):
    harfler = [c for c in str(text) if c.isalpha()]
    if not harfler:
        return 0.0
    buyuk = sum(1 for c in harfler if c.isupper())
    return buyuk / len(harfler)


# =========================================================
# KUTU YARDIMCILARI
# =========================================================

def kutu_yuksekligi(item):
    return max(0, item["y2"] - item["y1"]) if item is not None else 0


def kutu_genisligi(item):
    return max(0, item["x2"] - item["x1"]) if item is not None else 0


def kutu_merkez_x(item):
    return (item["x1"] + item["x2"]) / 2 if item is not None else 0


# =========================================================
# TC
# =========================================================

def tc_kimlik_gecerli_mi(no):
    if not no or len(no) != 11 or not no.isdigit() or no[0] == "0":
        return False
    d = [int(x) for x in no]
    d10 = (sum(d[0:9:2]) * 7 - sum(d[1:8:2])) % 10
    d11 = sum(d[:10]) % 10
    return d[9] == d10 and d[10] == d11


def tc_metni_duzelt(text):
    donusum = {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "İ": "1", "L": "1",
        "Z": "2", "S": "5", "G": "6", "B": "8",
    }
    sonuc = ""
    for karakter in str(text).upper():
        if karakter.isdigit():
            sonuc += karakter
        elif karakter in donusum:
            sonuc += donusum[karakter]
    return sonuc


def tc_bul(ocr_sonuclari):
    # 1. Direkt TC
    for item in ocr_sonuclari:
        rakamlar = re.sub(r"\D", "", item["text"])
        if len(rakamlar) == 11 and tc_kimlik_gecerli_mi(rakamlar):
            return rakamlar, item

    # 2. OCR düzeltmeli TC
    for item in ocr_sonuclari:
        aday = tc_metni_duzelt(item["text"])
        if len(aday) == 11 and tc_kimlik_gecerli_mi(aday):
            return aday, item

    return "Bulunamadi", None


# =========================================================
# ANA EASYOCR
# =========================================================

def easyocr_oku(resim, offset_x=0, offset_y=0):
    baslangic = time.perf_counter()
    bulunanlar = []

    try:
        sonuclar = reader.readtext(resim, detail=1, paragraph=False, decoder="greedy")
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
        x1 = int(box[:, 0].min()) + offset_x
        y1 = int(box[:, 1].min()) + offset_y
        x2 = int(box[:, 0].max()) + offset_x
        y2 = int(box[:, 1].max()) + offset_y

        bulunanlar.append({
            "text": text, "norm": normalize_text(text), "conf": float(conf),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    sure = time.perf_counter() - baslangic
    return bulunanlar, sure


# =========================================================
# LABEL HEDEFLERİ
# =========================================================

SOYAD_LABEL_HEDEFLERI = ["SOYADI SURNAME", "SURNAME", "SOYADI"]

AD_LABEL_HEDEFLERI = [
    "ADI GIVEN NAME S", "ADI GIVEN NAMES", "ADI GIVEN NAME",
    "GIVEN NAME S", "GIVEN NAMES", "GIVEN NAME",
]


# =========================================================
# LABEL BUL
# =========================================================

def fuzzy_label_bul(ocr_sonuclari, hedefler, esik=0.40):
    en_iyi_item = None
    en_iyi_skor = 0.0

    for item in ocr_sonuclari:
        for hedef in hedefler:
            skor = benzerlik(item["text"], hedef)
            if skor > en_iyi_skor:
                en_iyi_skor = skor
                en_iyi_item = item

    return en_iyi_item if en_iyi_skor >= esik else None


# =========================================================
# LABEL METNİ Mİ?
# =========================================================

def label_metni_mi(text):
    norm = normalize_text(text)
    if not norm:
        return False

    sabitler = [
        "TURKIYE", "REPUBLIC", "IDENTITY", "KIMLIK", "CARD",
        "SOYADI", "SURNAME", "ADI", "GIVEN", "NAME",
        "DATE", "BIRTH", "GENDER", "CINSIYET",
        "DOCUMENT", "SERI", "NATIONALITY", "UYRUGU",
        "VALID", "GECERLILIK", "SIGNATURE", "IMZASI",
    ]

    return any(kelime in norm for kelime in sabitler)


# =========================================================
# CİNSİYET DEĞERİ Mİ?
# =========================================================

def cinsiyet_degeri_mi(text):
    norm = normalize_text(text)
    yasak = {"K", "F", "E", "M", "K F", "F K", "E M", "M E", "M F", "F M"}
    return norm in yasak


# =========================================================
# VALUE ADAYI MI?
# =========================================================

def isim_value_adayi_mi(item, label_item=None):
    if item is None:
        return False

    text = str(item["text"]).strip()
    if not text:
        return False

    norm = normalize_text(text)

    # Kesin yasaklar
    if norm in {"TC", "T C", "TR", "TUR"}:
        return False

    # Cinsiyet / label / sayı-sembol / rakam
    if cinsiyet_degeri_mi(text):
        return False
    if label_metni_mi(text):
        return False
    if re.fullmatch(r"[\d\W]+", text):
        return False
    if any(c.isdigit() for c in text):
        return False

    # Temizle
    temiz = isim_temizle(text)
    harfler = re.sub(r"[^A-ZÇĞİÖŞÜ]", "", temiz)
    if len(harfler) < 2:
        return False

    # Büyük harf oranı
    if buyuk_harf_orani(text) < 0.80:
        return False

    # Value label'dan aşırı küçük olmasın
    if label_item is not None:
        value_h = kutu_yuksekligi(item)
        label_h = kutu_yuksekligi(label_item)
        if label_h > 0 and value_h < label_h * 0.75:
            return False

    return True


# =========================================================
# LABEL ALTINDAKİ VALUE
# =========================================================

def label_degerini_bul(ocr_sonuclari, label_item, sonraki_label=None, maksimum_dikey=140, maksimum_yatay=220):
    if label_item is None:
        return None

    adaylar = []
    label_merkez_x = kutu_merkez_x(label_item)
    label_w = max(1, kutu_genisligi(label_item))

    for item in ocr_sonuclari:
        if item is label_item:
            continue
        if not isim_value_adayi_mi(item, label_item):
            continue

        # Label'ın altında
        dikey_fark = item["y1"] - label_item["y2"]
        if dikey_fark < -12 or dikey_fark > maksimum_dikey:
            continue

        # Soyad ad label'ının altına geçmesin
        if sonraki_label is not None and item["y1"] >= sonraki_label["y1"]:
            continue

        # Aynı sütun
        yatay_fark = abs(item["x1"] - label_item["x1"])
        if yatay_fark > maksimum_yatay:
            continue

        # Merkez kontrolü
        item_merkez_x = kutu_merkez_x(item)
        merkez_farki = abs(item_merkez_x - label_merkez_x)
        izinli_merkez_farki = max(120, label_w * 0.85)
        if merkez_farki > izinli_merkez_farki:
            continue

        # Aşırı sağa kaçmasın
        if item["x1"] > label_item["x2"] + 90:
            continue

        # Score
        value_h = kutu_yuksekligi(item)
        label_h = max(1, kutu_yuksekligi(label_item))
        boyut_orani = value_h / label_h

        skor = 0.0
        skor -= max(0, dikey_fark) * 2.5
        skor -= yatay_fark * 0.20
        skor -= merkez_farki * 0.12
        skor += buyuk_harf_orani(item["text"]) * 35
        skor += min(boyut_orani, 2.0) * 12
        skor += item["conf"] * 15
        if 0 <= dikey_fark <= 65:
            skor += 40
        if yatay_fark <= 70:
            skor += 20

        adaylar.append((skor, item))

    if not adaylar:
        return None
    return max(adaylar, key=lambda x: x[0])[1]


# =========================================================
# FALLBACK
# =========================================================

def fallback_ad_soyad_bul(ocr_sonuclari, tc_item):
    if tc_item is None:
        return None, None

    try:
        tc_index = ocr_sonuclari.index(tc_item)
    except ValueError:
        return None, None

    adaylar = [
        item for item in ocr_sonuclari[tc_index + 1:tc_index + 16]
        if isim_value_adayi_mi(item) and item["x1"] <= 700 and item["y1"] > tc_item["y2"] + 20
    ]

    if len(adaylar) < 2:
        return None, None

    adaylar = sorted(adaylar, key=lambda x: (x["y1"], x["x1"]))
    soyad_item, ad_item = adaylar[0], adaylar[1]
    return ad_item, soyad_item


# =========================================================
# AYNI KUTU
# =========================================================

def ayni_item_mi(a, b):
    if a is None or b is None:
        return False
    return a["x1"] == b["x1"] and a["y1"] == b["y1"] and a["x2"] == b["x2"] and a["y2"] == b["y2"]


# =========================================================
# TÜRKÇE SECOND PASS GEREKLİ Mİ?
# =========================================================

def turkce_second_pass_gerekli_mi(text):
    """Yalnızca Türkçe karşılığı olabilecek karakterlerden biri varsa
    (S,I,U,O,C,G) second-pass çalıştırılır. Böylece 'KENAN', 'MERYEM' gibi
    şüpheli harf içermeyen isimlerde gereksiz ikinci OCR yapılmaz."""
    if not text:
        return False
    text = normalize_text(text)
    supheli_harfler = {"S", "I", "U", "O", "C", "G"}
    return any(karakter in supheli_harfler for karakter in text)


# =========================================================
# KUTUYU HAFİF GENİŞLET
# =========================================================

def isim_kutusunu_hafif_genislet(kart, item):
    if item is None:
        return None

    h, w = kart.shape[:2]
    ust, alt, sol, sag = 5, 3, 2, 2

    x1 = max(0, item["x1"] - sol)
    y1 = max(0, item["y1"] - ust)
    x2 = min(w, item["x2"] + sag)
    y2 = min(h, item["y2"] + alt)

    roi = kart[y1:y2, x1:x2]
    return roi if roi.size else None


# =========================================================
# TÜRKÇE HARF SECOND PASS
# =========================================================

def turkce_harf_iyilestir(kart, item):
    if item is None:
        return None, 0.0, 0.0

    ilk_text = isim_temizle(item["text"])
    ilk_conf = float(item["conf"])

    if not ilk_text:
        return None, ilk_conf, 0.0

    # Hızlandırma: S/I/U/O/C/G yoksa ikinci OCR yok.
    if not turkce_second_pass_gerekli_mi(ilk_text):
        return ilk_text, ilk_conf, 0.0

    baslangic = time.perf_counter()
    roi = isim_kutusunu_hafif_genislet(kart, item)
    if roi is None:
        return ilk_text, ilk_conf, 0.0

    roi = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    try:
        sonuclar = reader.readtext(roi, detail=1, paragraph=False, decoder="greedy")
    except Exception:
        return ilk_text, ilk_conf, time.perf_counter() - baslangic

    sure = time.perf_counter() - baslangic
    if not sonuclar:
        return ilk_text, ilk_conf, sure

    uygunlar = []
    for _, ikinci_text, ikinci_conf in sonuclar:
        ikinci_text = isim_temizle(ikinci_text)
        if not ikinci_text:
            continue
        # İkinci OCR tamamen farklı kelime üretemez.
        if normalize_text(ikinci_text) != normalize_text(ilk_text):
            continue
        uygunlar.append((float(ikinci_conf), ikinci_text))

    if not uygunlar:
        return ilk_text, ilk_conf, sure

    ikinci_conf, ikinci_text = max(uygunlar, key=lambda x: x[0])

    ilk_turkce = len(re.findall(r"[ÇĞİÖŞÜ]", ilk_text))
    ikinci_turkce = len(re.findall(r"[ÇĞİÖŞÜ]", ikinci_text))

    # Yalnızca Türkçe karakter gerçekten kazanıldıysa değiştirilir.
    if ikinci_turkce > ilk_turkce:
        return ikinci_text, ikinci_conf, sure
    return ilk_text, ilk_conf, sure


# =========================================================
# DEBUG
# =========================================================

def debug_resmi_olustur(kart, tc_item, soyad_label, soyad_item, ad_label, ad_item):
    debug = kart.copy()

    def kutu_ciz(item, etiket, renk, kalinlik, font_olcek):
        if item is None:
            return
        cv2.rectangle(debug, (item["x1"], item["y1"]), (item["x2"], item["y2"]), renk, kalinlik)
        cv2.putText(
            debug, etiket, (item["x1"], max(25, item["y1"] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, font_olcek, renk, 2 if kalinlik > 2 else 1,
        )

    kutu_ciz(tc_item, "TC", (0, 255, 0), 4, 0.7)
    kutu_ciz(soyad_label, "SOYAD LABEL", (0, 180, 255), 2, 0.45)
    kutu_ciz(soyad_item, "SOYAD", (0, 255, 255), 4, 0.7)
    kutu_ciz(ad_label, "AD LABEL", (255, 180, 0), 2, 0.45)
    kutu_ciz(ad_item, "AD", (255, 255, 0), 4, 0.7)

    return debug


# =========================================================
# ANA FONKSİYON
# =========================================================

def bilgileri_cimbizla(orijinal_kart, sayfa_no=None, debug=False):
    if orijinal_kart is None:
        return {
            "sayfa_no": sayfa_no, "tc_no": "Bulunamadi", "ad": "Bulunamadi", "soyad": "Bulunamadi",
            "guven": "dusuk", "ad_conf": 0.0, "soyad_conf": 0.0,
            "ocr_suresi": 0.0, "ana_ocr_suresi": 0.0, "second_pass_suresi": 0.0,
            "debug_resmi": None, "tum_ocr": [],
        }

    # ---- 1. OCR bölgesi ----------------------------------------------------
    h, w = orijinal_kart.shape[:2]
    x1, x2 = int(w * 0.02), int(w * 0.70)
    y1, y2 = int(h * 0.15), int(h * 0.70)
    bilgi_bolgesi = orijinal_kart[y1:y2, x1:x2]

    # ---- 2. Ana OCR ----------------------------------------------------
    ocr_sonuclari, ana_ocr_suresi = easyocr_oku(bilgi_bolgesi, offset_x=x1, offset_y=y1)

    # ---- 3. TC -------------------------------------------------------------
    tc_no, tc_item = tc_bul(ocr_sonuclari)

    # ---- 4. Label'lar -----------------------------------------------------
    soyad_label = fuzzy_label_bul(ocr_sonuclari, SOYAD_LABEL_HEDEFLERI, esik=0.40)
    ad_label = fuzzy_label_bul(ocr_sonuclari, AD_LABEL_HEDEFLERI, esik=0.40)

    # ---- 5-6. Soyad / Ad ----------------------------------------------------
    soyad_item = label_degerini_bul(
        ocr_sonuclari, soyad_label, sonraki_label=ad_label, maksimum_dikey=130, maksimum_yatay=220
    )
    ad_item = label_degerini_bul(
        ocr_sonuclari, ad_label, sonraki_label=None, maksimum_dikey=140, maksimum_yatay=220
    )

    # ---- 7. Fallback -----------------------------------------------------
    if soyad_item is None or ad_item is None:
        fallback_ad, fallback_soyad = fallback_ad_soyad_bul(ocr_sonuclari, tc_item)
        soyad_item = soyad_item or fallback_soyad
        ad_item = ad_item or fallback_ad

    # ---- 8. Cinsiyet güvenliği ---------------------------------------------
    if ad_item is not None and cinsiyet_degeri_mi(ad_item["text"]):
        ad_item = None
    if soyad_item is not None and cinsiyet_degeri_mi(soyad_item["text"]):
        soyad_item = None

    # ---- 9. Aynı kutu -----------------------------------------------------
    if ayni_item_mi(ad_item, soyad_item):
        ad_item = None
        soyad_item = None

    # ---- 10. Türkçe second pass ---------------------------------------------
    ad, ad_conf, ad_second_sure = turkce_harf_iyilestir(orijinal_kart, ad_item)
    soyad, soyad_conf, soyad_second_sure = turkce_harf_iyilestir(orijinal_kart, soyad_item)

    second_pass_suresi = ad_second_sure + soyad_second_sure
    ocr_suresi = ana_ocr_suresi + second_pass_suresi

    ad = ad or "Bulunamadi"
    soyad = soyad or "Bulunamadi"

    # ---- 11. Güven -----------------------------------------------------
    bulunan = sum([tc_no != "Bulunamadi", ad != "Bulunamadi", soyad != "Bulunamadi"])
    if bulunan == 3:
        guven = "orta" if (ad_conf < 0.50 or soyad_conf < 0.50) else "yuksek"
    elif bulunan > 0:
        guven = "orta"
    else:
        guven = "dusuk"

    # ---- 12. Debug -----------------------------------------------------
    debug_resmi = None
    tum_ocr = []
    if debug:
        debug_resmi = debug_resmi_olustur(orijinal_kart, tc_item, soyad_label, soyad_item, ad_label, ad_item)
        tum_ocr = [
            {
                "no": index + 1, "text": item["text"], "conf": round(item["conf"], 3),
                "buyuk_harf": round(buyuk_harf_orani(item["text"]), 2),
                "x1": item["x1"], "y1": item["y1"], "x2": item["x2"], "y2": item["y2"],
            }
            for index, item in enumerate(ocr_sonuclari)
        ]

    return {
        "sayfa_no": sayfa_no, "tc_no": tc_no, "ad": ad, "soyad": soyad, "guven": guven,
        "ad_conf": ad_conf, "soyad_conf": soyad_conf,
        "ocr_suresi": ocr_suresi, "ana_ocr_suresi": ana_ocr_suresi, "second_pass_suresi": second_pass_suresi,
        "debug_resmi": debug_resmi, "tum_ocr": tum_ocr,
    }