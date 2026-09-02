# -*- coding: utf-8 -*-
"""Yakın zamanlı tarama sonuçlarının yerel kaydı.

Veritabanı değil: her tarama tek bir .zip dosyasına yazılır. İçinde tablo
verisinin tamamı (JSON) ve her kimliğin kırpılmış kart görüntüsü (JPEG) durur.
Böylece eski bir tarama açıldığında tablo da önizlemeler de eksiksiz gelir —
kaynak PDF silinmiş/taşınmış olsa bile.

Kayıtlar kullanıcının kendi veri klasöründe tutulur (.exe güncellense de
silinmez) ve en yeni AZAMI_KAYIT tanesi saklanır; eskiler kendiliğinden
temizlenir.

DİKKAT: Bu dosyalar kimlik numarası ve isim içerir. Arayüzdeki "Geçmişi
temizle" hepsini siler.
"""

import json
import os
import sys
import uuid
import zipfile
from datetime import datetime

import cv2
import numpy as np


# En yeni kaç tarama saklansın
AZAMI_KAYIT = 20

# Güvenlik freni: tek bir taramanın görüntüleri için üst sınır. Çok büyük
# işlerde (binlerce kimlik) disk sessizce dolmasın diye, sınır aşılınca kalan
# satırların görüntüsü saklanmaz — tablo verisi yine eksiksiz yazılır.
AZAMI_GORSEL_BAYT = 300 * 1024 * 1024

# Bütün geçmişin toplam üst sınırı. Kimlik başına ~75 KB görüntü tutuluyor;
# sınır aşılınca kayıt sayısı 20'nin altında olsa bile en eskiler silinir.
AZAMI_TOPLAM_BAYT = 1536 * 1024 * 1024

JPEG_KALITESI = 80

# Satırın kaydedilecek alanları. Görüntüler ve geçici alanlar hariç her şey.
METIN_ALANLARI = (
    "Dosya", "Sayfa", "Kart", "Belge Türü", "Kimlik No", "Ad", "Soyad",
    "Bitiş Tarihi", "Geçerlilik", "Durum", "Süre",
)
BAYRAK_ALANLARI = (
    "_belge_tipi", "_belge_gecerli", "_kart_bulundu", "_tc_supheli",
    "_kurtarildi", "_manuel_eklendi", "_manuel_duzenlendi", "_duzenlenen_alanlar",
    "_ad_conf", "_soyad_conf", "_kaynak_yol", "_gorsel_id",
    "_excel_adi", "_excelden_alanlar", "_ad_uyusmazligi",
)


def _gorsel_adi(gorsel_id):
    return f"gorseller/{gorsel_id}.jpg"


def gecmis_dizini():
    """Kayıtların tutulduğu klasör; yoksa oluşturulur."""
    if sys.platform.startswith("win"):
        kok = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        dizin = os.path.join(kok, "KimlikOkuyucu", "gecmis")
    elif sys.platform == "darwin":
        dizin = os.path.expanduser("~/Library/Application Support/KimlikOkuyucu/gecmis")
    else:
        dizin = os.path.expanduser("~/.local/share/KimlikOkuyucu/gecmis")

    os.makedirs(dizin, exist_ok=True)
    return dizin


# =========================================================
# KAYDETME
# =========================================================

def _satiri_serilestir(satir):
    kayit = {alan: satir.get(alan) for alan in METIN_ALANLARI}
    for alan in BAYRAK_ALANLARI:
        if satir.get(alan) is not None:
            kayit[alan] = satir[alan]

    koseler = satir.get("_koseler")
    if koseler is not None:
        kayit["_koseler"] = [[int(round(float(x))), int(round(float(y)))] for x, y in koseler]

    # Debug panelinin gösterdiği ayrıntılar; ham OCR listesi çok büyüyebildiği
    # için dışarıda bırakılıyor.
    for alan in ("_kart_sonuc", "_ocr_sonuc"):
        kaynak = satir.get(alan) or {}
        kayit[alan] = {
            k: v for k, v in kaynak.items()
            if k != "tum_ocr" and isinstance(v, (str, int, float, bool, type(None), list, dict))
        }
    return kayit


def _jpeg_kodla(resim):
    if resim is None or getattr(resim, "size", 0) == 0:
        return None
    try:
        ok, veri = cv2.imencode(".jpg", resim, [cv2.IMWRITE_JPEG_QUALITY, JPEG_KALITESI])
    except cv2.error:
        return None
    return veri.tobytes() if ok else None


def _yeni_kayit_yolu():
    dizin = gecmis_dizini()
    damga = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    yol = os.path.join(dizin, f"{damga}.kimlik.zip")

    # Aynı saniyede biten iki tarama birbirini ezmesin.
    ek = 2
    while os.path.exists(yol):
        yol = os.path.join(dizin, f"{damga}_{ek}.kimlik.zip")
        ek += 1
    return yol


def taramayi_kaydet(satirlar, kaynak_dosyalar=(), etiket="", yol=None):
    """Taramayı geçmişe yazar ve dosya yolunu döner; satır yoksa None.

    `yol` verilirse o kayıt YERİNDE güncellenir (tarama sonrası yapılan
    düzenlemeler için). Güncellemede değişmeyen kart görüntüleri eski
    dosyadan olduğu gibi kopyalanır — JPEG yeniden üretilmez, bu yüzden
    güncelleme büyük taramalarda da hızlıdır.

    Satırlara `_gorsel_id` yazılır; görüntünün kimliği satırla birlikte
    taşındığı için sıra değişse de doğru görüntü eşleşir."""
    if not satirlar:
        return None

    eski_ozet = {}
    eski_gorseller = {}
    if yol and os.path.exists(yol):
        try:
            with zipfile.ZipFile(yol) as z:
                eski_ozet = json.loads(z.read("ozet.json").decode("utf-8"))
                for ad in z.namelist():
                    if ad.startswith("gorseller/"):
                        eski_gorseller[ad] = z.read(ad)
        except Exception:
            eski_ozet, eski_gorseller = {}, {}

    hedef = yol or _yeni_kayit_yolu()

    kayitlar = []
    gorseller = []
    toplam_gorsel = 0

    for satir in satirlar:
        gorsel_id = satir.get("_gorsel_id")
        veri = None

        # Değişmemiş görüntüyü eski dosyadan aynen al.
        if gorsel_id and _gorsel_adi(gorsel_id) in eski_gorseller:
            veri = eski_gorseller[_gorsel_adi(gorsel_id)]
        elif satir.get("_preview") is not None:
            veri = _jpeg_kodla(satir.get("_preview"))
            if veri:
                gorsel_id = uuid.uuid4().hex[:12]
                satir["_gorsel_id"] = gorsel_id

        kayit = _satiri_serilestir(satir)
        if veri and gorsel_id and toplam_gorsel + len(veri) <= AZAMI_GORSEL_BAYT:
            gorseller.append((_gorsel_adi(gorsel_id), veri))
            kayit["_gorsel_id"] = gorsel_id
            toplam_gorsel += len(veri)
        else:
            kayit.pop("_gorsel_id", None)
        kayitlar.append(kayit)

    simdi = datetime.now().isoformat(timespec="seconds")
    ozet = {
        "surum": 2,
        "tarih": eski_ozet.get("tarih") or simdi,
        "etiket": etiket or eski_ozet.get("etiket", ""),
        # Var olan bir kayıt güncelleniyorsa dosya listesi ONUN listesidir.
        # Çağıranın o anki dosya seçimi (başka bir tarama için seçilmiş
        # olabilir) kaydın adını değiştirmemeli.
        "dosyalar": eski_ozet.get("dosyalar")
                    or sorted({os.path.basename(d) for d in kaynak_dosyalar})
                    or sorted({str(s.get("Dosya", "")) for s in satirlar}),
        "satir_sayisi": len(satirlar),
        "kimlik_sayisi": sum(1 for s in satirlar if s.get("_kart_bulundu")),
        "gorsel_sayisi": len(gorseller),
    }
    if yol and eski_ozet:
        ozet["guncellendi"] = simdi

    gecici = hedef + ".tmp"
    try:
        with zipfile.ZipFile(gecici, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("ozet.json", json.dumps(ozet, ensure_ascii=False))
            z.writestr("kayitlar.json", json.dumps(kayitlar, ensure_ascii=False))
            for ad, veri in gorseller:
                # JPEG zaten sıkıştırılmış; tekrar sıkıştırmak sadece zaman yer.
                z.writestr(ad, veri, compress_type=zipfile.ZIP_STORED)
        os.replace(gecici, hedef)
    except Exception:
        if os.path.exists(gecici):
            try:
                os.remove(gecici)
            except OSError:
                pass
        raise

    eskileri_temizle()
    return hedef


def eskileri_temizle(azami=None, azami_bayt=None):
    """İki sınırı birden uygular: en yeni `azami` kayıt kalır ve toplam boyut
    `azami_bayt`ı aşmaz. İkisinde de en eskiden başlanarak silinir.

    Sınırlar çağrı anında okunuyor; böylece modül sabitleri değiştirildiğinde
    (ör. testte) yeni değer geçerli oluyor."""
    azami = AZAMI_KAYIT if azami is None else azami
    azami_bayt = AZAMI_TOPLAM_BAYT if azami_bayt is None else azami_bayt

    kayitlar = taramalari_listele()

    for kayit in kayitlar[azami:]:
        taramayi_sil(kayit["yol"])
    kalan = kayitlar[:azami]

    toplam = sum(k["boyut"] for k in kalan)
    while kalan and toplam > azami_bayt:
        eski = kalan.pop()
        if taramayi_sil(eski["yol"]):
            toplam -= eski["boyut"]


# =========================================================
# LİSTELEME / YÜKLEME / SİLME
# =========================================================

def taramalari_listele():
    """Kayıtlı taramalar, en yeniden eskiye doğru."""
    dizin = gecmis_dizini()
    kayitlar = []

    for ad in os.listdir(dizin):
        if not ad.endswith(".kimlik.zip"):
            continue
        yol = os.path.join(dizin, ad)
        try:
            with zipfile.ZipFile(yol) as z:
                ozet = json.loads(z.read("ozet.json").decode("utf-8"))
        except Exception:
            # Bozuk/yarım kalmış dosya listeyi çökertmesin.
            continue

        ozet["yol"] = yol
        ozet["boyut"] = os.path.getsize(yol)
        ozet["_sira"] = os.path.getmtime(yol)
        kayitlar.append(ozet)

    kayitlar.sort(key=lambda k: k["_sira"], reverse=True)
    return kayitlar


def taramayi_yukle(yol):
    """Kayıtlı taramayı tablo satırlarına geri çevirir."""
    satirlar = []
    with zipfile.ZipFile(yol) as z:
        kayitlar = json.loads(z.read("kayitlar.json").decode("utf-8"))
        mevcut = set(z.namelist())

        for kayit in kayitlar:
            satir = dict(kayit)
            # Sürüm 1 kayıtlarında görüntü adı doğrudan yazılıyordu.
            gorsel_adi = satir.pop("_gorsel", None)
            if satir.get("_gorsel_id"):
                gorsel_adi = _gorsel_adi(satir["_gorsel_id"])

            satir["_preview"] = None
            if gorsel_adi and gorsel_adi in mevcut:
                try:
                    veri = np.frombuffer(z.read(gorsel_adi), dtype=np.uint8)
                    satir["_preview"] = cv2.imdecode(veri, cv2.IMREAD_COLOR)
                except Exception:
                    satir["_preview"] = None

            satir.setdefault("_kart_sonuc", {})
            satir.setdefault("_ocr_sonuc", {})
            satir["_debug_resmi"] = None
            satir["_kart_debug"] = None
            satir["_gecmisten"] = True
            satirlar.append(satir)

    return satirlar


def taramayi_sil(yol):
    try:
        os.remove(yol)
        return True
    except OSError:
        return False


def gecmisi_temizle():
    """Bütün kayıtları siler (kimlik numarası içerdikleri için tek hamlede)."""
    dizin = gecmis_dizini()
    silinen = 0
    for ad in os.listdir(dizin):
        if ad.endswith(".kimlik.zip") and taramayi_sil(os.path.join(dizin, ad)):
            silinen += 1
    return silinen


def toplam_boyut():
    return sum(k["boyut"] for k in taramalari_listele())
