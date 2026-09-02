# -*- coding: utf-8 -*-
"""Excel dosyalarındaki gömülü kimlik fotoğraflarını okumak.

Elimizdeki örnek dosyada her kişi için bir "bant" var: aynı satır hizasında
solda kimliğin ÖN yüzü, sağda ARKA yüzü, F sütununda da kişinin adı yazıyor.
Ölçtük — arka yüzler kart olarak tespit edilmiyor (12 fotoğrafta 0 yanlış
pozitif), ön yüzlerin hepsi bulunuyor. Bu yüzden her bant tek bir "sayfa"
gibi ele alınıyor: bandın fotoğrafları yan yana birleştirilip tek görüntü
olarak veriliyor, böylece kişi başına tek satır üretiliyor ve önizlemede
Excel'deki görünümün aynısı çıkıyor.

Kişinin Excel'de yazan adı da bantla birlikte taşınıyor; OCR adı okuyamazsa
oradan tamamlanabiliyor.
"""

import os

import cv2
import numpy as np

try:
    import openpyxl
except ImportError:  # openpyxl zaten Excel çıktısı için gerekli
    openpyxl = None


EXCEL_UZANTILARI = (".xlsx", ".xlsm")

# Bir bandın fotoğrafları arasındaki en fazla satır farkı. Fotoğraflar tam
# aynı satıra tutturulmayabiliyor (ör. 51 ve 50); bantlar ise ~10 satır arayla.
BANT_SATIR_TOLERANSI = 5

# Birleştirilmiş bant görüntüsü için sınırlar
BANT_AZAMI_YUKSEKLIK = 1100
BANT_AZAMI_GENISLIK = 3400
BANT_ARA_BOSLUK = 16

# Kişinin adının arandığı sütun (F)
AD_SUTUNU = 6

_KITAP_ONBELLEK = {}


def excel_mi(yol):
    return str(yol).lower().endswith(EXCEL_UZANTILARI)


def _kitabi_ac(yol):
    """Aynı dosya tekrar tekrar açılmasın (görüntüler tembel okunuyor)."""
    anahtar = (os.path.abspath(yol), os.path.getmtime(yol))
    kitap = _KITAP_ONBELLEK.get(anahtar)
    if kitap is None:
        if openpyxl is None:
            raise RuntimeError("openpyxl kurulu değil; Excel dosyaları okunamıyor.")
        kitap = openpyxl.load_workbook(yol)
        _KITAP_ONBELLEK.clear()
        _KITAP_ONBELLEK[anahtar] = kitap
    return kitap


def _gorsel_verisi(gorsel):
    veri = gorsel.ref
    if hasattr(veri, "getvalue"):
        return veri.getvalue()
    if callable(getattr(gorsel, "_data", None)):
        return gorsel._data()
    if isinstance(veri, (bytes, bytearray)):
        return bytes(veri)
    with open(veri, "rb") as f:
        return f.read()


def _resmi_coz(gorsel):
    try:
        veri = np.frombuffer(_gorsel_verisi(gorsel), dtype=np.uint8)
        return cv2.imdecode(veri, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _satirdaki_ad(sayfa, ilk_satir, son_satir):
    """Bandın satır aralığında F sütununda yazan ilk dolu değeri döner."""
    for satir in range(max(1, ilk_satir), son_satir + 1):
        try:
            deger = sayfa.cell(row=satir, column=AD_SUTUNU).value
        except Exception:
            continue
        if deger not in (None, ""):
            return str(deger).strip()
    return ""


def bantlari_bul(yol):
    """Dosyadaki bantlar: [{"no", "sayfa_adi", "excel_satiri", "ad", "gorseller"}]

    "gorseller" o bandın openpyxl görüntü nesneleri (soldan sağa sıralı);
    görüntü verisi ancak istendiğinde çözülüyor."""
    kitap = _kitabi_ac(yol)
    bantlar = []

    for sayfa in kitap.worksheets:
        gorseller = list(getattr(sayfa, "_images", []) or [])
        if not gorseller:
            continue

        yerlesik = []
        for gorsel in gorseller:
            try:
                bas = gorsel.anchor._from
                son = getattr(gorsel.anchor, "to", None)
                yerlesik.append({
                    "gorsel": gorsel,
                    "satir": int(bas.row),
                    "sutun": int(bas.col),
                    "son_satir": int(getattr(son, "row", bas.row)),
                })
            except Exception:
                continue

        yerlesik.sort(key=lambda g: (g["satir"], g["sutun"]))

        grup = []
        for oge in yerlesik:
            if grup and oge["satir"] - grup[-1]["satir"] > BANT_SATIR_TOLERANSI:
                bantlar.append((sayfa, grup))
                grup = []
            grup.append(oge)
        if grup:
            bantlar.append((sayfa, grup))

    sonuc = []
    for no, (sayfa, grup) in enumerate(bantlar, start=1):
        ilk = min(g["satir"] for g in grup) + 1          # openpyxl 0 tabanlı
        son = max(g["son_satir"] for g in grup) + 1
        sonuc.append({
            "no": no,
            "sayfa_adi": sayfa.title,
            "excel_satiri": ilk,
            "ad": _satirdaki_ad(sayfa, ilk, son),
            "gorseller": [g["gorsel"] for g in sorted(grup, key=lambda g: g["sutun"])],
        })
    return sonuc


def bandi_birlestir(gorseller):
    """Bandın fotoğraflarını Excel'deki gibi yan yana tek görüntüye getirir."""
    resimler = [r for r in (_resmi_coz(g) for g in gorseller) if r is not None and r.size]
    if not resimler:
        return None
    if len(resimler) == 1:
        return np.ascontiguousarray(resimler[0])

    hedef_h = min(BANT_AZAMI_YUKSEKLIK, max(r.shape[0] for r in resimler))
    olcekli = []
    for resim in resimler:
        olcek = hedef_h / float(resim.shape[0])
        yeni_g = max(1, int(round(resim.shape[1] * olcek)))
        olcekli.append(cv2.resize(resim, (yeni_g, hedef_h),
                                  interpolation=cv2.INTER_AREA if olcek < 1 else cv2.INTER_CUBIC))

    toplam_g = sum(r.shape[1] for r in olcekli) + BANT_ARA_BOSLUK * (len(olcekli) - 1)
    if toplam_g > BANT_AZAMI_GENISLIK:
        kucult = BANT_AZAMI_GENISLIK / float(toplam_g)
        hedef_h = max(1, int(hedef_h * kucult))
        olcekli = [cv2.resize(r, (max(1, int(r.shape[1] * kucult)), hedef_h),
                              interpolation=cv2.INTER_AREA) for r in olcekli]
        toplam_g = sum(r.shape[1] for r in olcekli) + BANT_ARA_BOSLUK * (len(olcekli) - 1)

    tuval = np.full((hedef_h, toplam_g, 3), 255, np.uint8)
    x = 0
    for resim in olcekli:
        tuval[:, x:x + resim.shape[1]] = resim
        x += resim.shape[1] + BANT_ARA_BOSLUK
    return tuval


def bant_goruntusu(yol, bant_no):
    """Tek bir bandın birleştirilmiş görüntüsü (önizleme için)."""
    for bant in bantlari_bul(yol):
        if bant["no"] == bant_no:
            return bandi_birlestir(bant["gorseller"])
    return None


def adi_ayir(tam_ad):
    """"ZÜBEYDE KAYA" -> ("ZÜBEYDE", "KAYA"). Son kelime soyadı sayılır."""
    parcalar = [p for p in str(tam_ad or "").split() if p]
    if len(parcalar) < 2:
        return (parcalar[0], "") if parcalar else ("", "")
    return " ".join(parcalar[:-1]), parcalar[-1]
