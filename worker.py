# -*- coding: utf-8 -*-
"""Arka planda calisan is parcaciklari: sayfalari/bantlari tarayip kimlik
tespiti + OCR yapan Worker, ve kullanicinin fareyle secip okuttugu tek
bir alani isleyen AlanOkuyucu.

desktop.py'den ayrildi -- arayuz kodundan bagimsiz calisir, MainWindow
sadece bu iki QThread'i baslatip sinyallerini dinler."""

import os
import time

import cv2
import numpy as np
import pymupdf

from PySide6.QtCore import QThread, Signal

from goruntu_isleme import kartlari_tespit_et_ve_duzelt
from metin_ayiklama import bilgileri_cimbizla, normalize_text, benzerlik
import excel_kaynak
from gorsel_araclari import dosyayi_goruntulere_ayir


# Tespit cozemedigi halde "kart buyuklugunde" duran bloklar, dogrudan
# OCR'lanip kurtarilmaya calisilir: gecerli bir TC numarasi cikarsa o kimlik
# sessizce kaybolmak yerine tabloya "kurtarildi" satiri olarak girer.
KURTARMA_AZAMI_BOLGE = 3
KURTARMA_HEDEF_GENISLIK = 1000

# Excel'de yazan ad ile okunan ad bu orandan az benzerse uyusmazlik bildirilir.
# Karsilastirma oncesi Turkce harfler sadelestigi icin (S->S, U->U...) dogru
# okumalar 1.00 veriyor; esik bu yuzden yuksek tutulabiliyor.
AD_BENZERLIK_ESIGI = 0.95


class AlanOkuyucu(QThread):
    """Kullanıcının seçtiği alanı arka planda okur (arayüz donmasın).

    Önce seçilen parçada normal kart tespiti denenir — bulunursa kart
    hizalanmış haliyle okunur, bu en iyi sonucu verir. Bulunamazsa seçilen
    alan doğrudan büyütülüp okunur."""

    bitti = Signal(dict)
    hata = Signal(str)

    def __init__(self, kirpma, debug=False):
        super().__init__()
        self.kirpma = kirpma
        self.debug = debug

    def run(self):
        try:
            kart = None
            belge_tipi = "tc"
            hizalandi = False
            try:
                adaylar = [
                    k for k in kartlari_tespit_et_ve_duzelt(self.kirpma, debug_kart=self.debug)
                    if k.get("basarili")
                ]
            except Exception:
                adaylar = []

            if adaylar:
                kart = adaylar[0].get("kart")
                belge_tipi = adaylar[0].get("belge_tipi", "tc")
                hizalandi = True
            else:
                kart = self.kirpma
                if kart.shape[1] < KURTARMA_HEDEF_GENISLIK:
                    olcek = KURTARMA_HEDEF_GENISLIK / float(kart.shape[1])
                    kart = cv2.resize(kart, None, fx=olcek, fy=olcek,
                                      interpolation=cv2.INTER_CUBIC)
                kart = np.ascontiguousarray(kart)

            ocr_sonuc = bilgileri_cimbizla(kart, debug=self.debug, belge_tipi=belge_tipi)
            self.bitti.emit({
                "kart": kart, "ocr": ocr_sonuc,
                "belge_tipi": belge_tipi, "hizalandi": hizalandi,
            })
        except Exception as exc:
            self.hata.emit(f"{type(exc).__name__}: {exc}")


class Worker(QThread):
    progress = Signal(int, int, str)
    row_ready = Signal(dict)
    finished_ok = Signal(list, object, bool)
    failed = Signal(str)

    def __init__(self, dosyalar, debug=False, kurtar=True, coklu=True, derin=True):
        super().__init__()
        self.dosyalar = dosyalar
        self.debug = debug
        self.kurtar = kurtar
        self.coklu = coklu
        self.derin = derin
        self._durduruldu = False

    def durdur(self):
        """Dışarıdan (ana thread'den) çağrılır. Basit bir bool bayrak —
        Python'da GIL sayesinde tek bir attribute yazımı thread-safe kabul
        edilir, ekstra kilide gerek yok. run() döngüsü bunu her sayfa
        aralığında kontrol eder; şu an işlenmekte olan sayfa yarıda
        kesilmez, bir SONRAKİ sayfaya geçmeden döngü sonlanır."""
        self._durduruldu = True

    def run(self):
        try:
            isler = []
            for yol in self.dosyalar:
                if yol.lower().endswith(".pdf"):
                    try:
                        with pymupdf.open(yol) as d:
                            for p in range(1, d.page_count + 1):
                                isler.append((yol, p))
                    except Exception:
                        isler.append((yol, None))
                elif excel_kaynak.excel_mi(yol):
                    try:
                        for bant in excel_kaynak.bantlari_bul(yol):
                            isler.append((yol, bant["no"]))
                    except Exception:
                        isler.append((yol, None))
                else:
                    isler.append((yol, None))

            toplam = max(1, len(isler))
            sonuclar = []
            sayac = 0

            for yol in self.dosyalar:
                if self._durduruldu:
                    break
                for sayfa_no, resim, girdi_hatasi, ek_bilgi in dosyayi_goruntulere_ayir(yol):
                    if self._durduruldu:
                        break
                    sayac += 1
                    self.progress.emit(sayac, toplam, f"{os.path.basename(yol)} işleniyor")

                    baslangic = time.perf_counter()
                    kart_sonuclari = []
                    tespit_hatasi = girdi_hatasi

                    if resim is not None:
                        try:
                            # Bir sayfada birden fazla kimlik olabilir; hepsi
                            # ayrı ayrı tespit edilip ayrı satır olarak işlenir.
                            kart_sonuclari = kartlari_tespit_et_ve_duzelt(
                                resim, debug_kart=self.debug, ek_tarama=self.coklu
                            )
                        except Exception as e:
                            tespit_hatasi = f"Kart tespit hatası: {e}"

                    if not kart_sonuclari:
                        kart_sonuclari = [{}]

                    tespit_suresi = time.perf_counter() - baslangic
                    kart_sayisi = sum(1 for k in kart_sonuclari if k.get("basarili"))

                    sayfa_satirlari = []
                    for sira, kart_sonuc in enumerate(kart_sonuclari, start=1):
                        sayfa_satirlari.append(self._satir_olustur(
                            yol=yol,
                            sayfa_no=sayfa_no,
                            resim=resim,
                            kart_sonuc=kart_sonuc,
                            sira=sira,
                            kart_sayisi=kart_sayisi,
                            tespit_hatasi=tespit_hatasi,
                            # Tespit süresi sayfanın tamamı için bir kez harcanır;
                            # sayfanın ilk satırına yazılır, diğerlerine yazılmaz.
                            ek_sure=tespit_suresi if sira == 1 else 0.0,
                            ek_bilgi=ek_bilgi,
                        ))

                    if ek_bilgi is not None:
                        sayfa_satirlari = self._excel_bandini_sadelestir(
                            sayfa_satirlari, ek_bilgi
                        )

                    # Tespitin çözemediği kart bloklarını doğrudan okumayı dene.
                    # Sadece sayfadan HİÇ kart çıkmadığında: kart bulunan
                    # sayfalarda arta kalan bloklar çoğunlukla kimliğin arka
                    # yüzü oluyor ve oradan üretilen satır hayalet kayıt olur.
                    if self.kurtar and resim is not None and kart_sayisi == 0:
                        kurtarilan = self._kurtarma_satirlari(
                            yol=yol,
                            sayfa_no=sayfa_no,
                            resim=resim,
                            bolgeler=(kart_sonuclari[0] or {}).get("kurtarma_bolgeleri", []),
                            mevcut_satirlar=sayfa_satirlari,
                        )
                        if kurtarilan:
                            # Sayfada hiç kart bulunamadıysa "bulunamadı" satırı
                            # artık yanıltıcı olur; kurtarılan kimlikler onun yerini alır.
                            if kart_sayisi == 0:
                                sayfa_satirlari = []
                            sayfa_satirlari.extend(kurtarilan)

                    for row in sayfa_satirlari:
                        sonuclar.append(row)
                        self.row_ready.emit(row)

            # PDF artık burada üretilmiyor: kullanıcı elle kimlik ekleyip
            # sırayı değiştirebildiği için, kaydetme anında tablodaki güncel
            # sıradan üretiliyor (bkz. MainWindow.pdf_kaydet).
            self.finished_ok.emit(sonuclar, None, self._durduruldu)
        except Exception as e:
            self.failed.emit(repr(e))

    def _excel_bandini_sadelestir(self, satirlar, ek_bilgi):
        """Excel'de bir bant tek kişidir: banttan TEK satır bırakılır.

        Kartın arka yüzü nadiren kart olarak da tespit edilebiliyor ve boş bir
        ikinci satır üretiyordu. En çok alanı okunan satır tutulur; ad/soyad
        eksikse Excel'in kendi yazdığı isimden tamamlanır, okunduysa onunla
        karşılaştırılır."""
        if not satirlar:
            return satirlar

        def dolu_alan(satir):
            return sum(
                1 for alan in ("Kimlik No", "Ad", "Soyad")
                if str(satir.get(alan, "")).strip() not in ("", "Bulunamadi")
            )

        satir = max(satirlar, key=dolu_alan)
        satir["Kart"] = "1/1" if satir.get("_kart_bulundu") else "-"

        excel_adi = (ek_bilgi or {}).get("excel_adi") or ""
        excelden = []
        ad_uyusmazligi = ""

        if excel_adi:
            e_ad, e_soyad = excel_kaynak.adi_ayir(excel_adi)
            if str(satir.get("Ad")) == "Bulunamadi" and e_ad:
                satir["Ad"] = e_ad
                excelden.append("Ad")
            if str(satir.get("Soyad")) == "Bulunamadi" and e_soyad:
                satir["Soyad"] = e_soyad
                excelden.append("Soyad")
            if not excelden:
                okunan = normalize_text(f"{satir.get('Ad')} {satir.get('Soyad')}")
                if benzerlik(okunan, normalize_text(excel_adi)) < AD_BENZERLIK_ESIGI:
                    ad_uyusmazligi = excel_adi

        satir["_excelden_alanlar"] = excelden
        satir["_ad_uyusmazligi"] = bool(ad_uyusmazligi)
        if excelden:
            satir["_kart_bulundu"] = True

        # Durum, tamamlanan alanlardan sonra yeniden yazılıyor.
        eksikler = [
            alan for alan in ("Kimlik No", "Ad", "Soyad")
            if str(satir.get(alan, "")).strip() in ("", "Bulunamadi")
        ]
        if satir.get("_hata") and not satir.get("_kart_bulundu"):
            durum = str(satir["_hata"])
        elif eksikler:
            durum = "Eksik alan: " + ", ".join(eksikler)
        elif satir.get("_tc_supheli"):
            durum = "Kimlik No doğrulanamadı — kontrol edin"
        else:
            durum = "Başarılı"
        if excelden:
            durum += f" ({'/'.join(excelden)} Excel'den alındı)"
        if ad_uyusmazligi:
            durum += f" — Excel'de: {ad_uyusmazligi}"
        satir["Durum"] = durum

        return [satir]

    def _kurtarma_satirlari(self, yol, sayfa_no, resim, bolgeler, mevcut_satirlar):
        """Tespitin kart çıkaramadığı blokları doğrudan OCR'lar.

        Kimlik kartı hizalanamadığında eskiden o kimlik tamamen kayboluyordu.
        Burada blok kırpılıp büyütülüyor ve normal alan okuma akışından
        geçiriliyor; GEÇERLİ (checksum'ı tutan) bir kimlik numarası çıkarsa
        satır olarak ekleniyor.

        Kartın arka yüzü de kart büyüklüğünde bir bloktur ve MRZ'sinde aynı
        numara yazar; o yüzden sayfada ZATEN görülmüş numaralar atlanıyor —
        arka yüzler kendiliğinden elenmiş oluyor."""
        gorulen_tcler = {
            str(r.get("Kimlik No", "")).strip()
            for r in mevcut_satirlar
            if str(r.get("Kimlik No", "")).strip() not in ("", "Bulunamadi")
        }

        satirlar = []
        for x0, y0, x1, y1 in list(bolgeler)[:KURTARMA_AZAMI_BOLGE]:
            if self._durduruldu:
                break

            baslangic = time.perf_counter()
            blok = resim[int(y0):int(y1), int(x0):int(x1)]
            if blok is None or blok.size == 0 or blok.shape[0] < 2 or blok.shape[1] < 2:
                continue

            # OCR, düzeltilmiş kart boyutlarında eğitilmiş oranlara göre
            # çalışıyor; küçük bloğu o ölçeğe büyütüyoruz.
            if blok.shape[1] < KURTARMA_HEDEF_GENISLIK:
                olcek = KURTARMA_HEDEF_GENISLIK / float(blok.shape[1])
                blok = cv2.resize(blok, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_CUBIC)
            blok = np.ascontiguousarray(blok)

            try:
                ocr_sonuc = bilgileri_cimbizla(
                    blok, sayfa_no=sayfa_no, debug=self.debug, belge_tipi="tc",
                    derin=self.derin,
                )
            except Exception:
                continue

            kimlik_no = str(ocr_sonuc.get("tc_no", "Bulunamadi")).strip()
            if kimlik_no in ("", "Bulunamadi") or kimlik_no in gorulen_tcler:
                continue
            # Kurtarmada yalnız doğrulanmış numaraya güveniyoruz; hizalanmamış
            # bloktan gelen şüpheli okuma yanlış kişi kaydı üretebilir.
            if not ocr_sonuc.get("tc_dogrulandi", True):
                continue
            gorulen_tcler.add(kimlik_no)

            ad = ocr_sonuc.get("ad", "Bulunamadi")
            soyad = ocr_sonuc.get("soyad", "Bulunamadi")

            eksikler = [
                alan for alan, deger in (("Ad", ad), ("Soyad", soyad))
                if deger == "Bulunamadi"
            ]
            durum = "Kart hizalanamadı, sayfadan okundu"
            if eksikler:
                durum += " — eksik alan: " + ", ".join(eksikler)

            satirlar.append({
                "Dosya": os.path.basename(yol),
                "Sayfa": sayfa_no if sayfa_no is not None else "-",
                "Kart": "K",
                "Belge Türü": "T.C. Kimlik",
                "Kimlik No": kimlik_no,
                "Ad": ad,
                "Soyad": soyad,
                "Bitiş Tarihi": "-",
                "Geçerlilik": "-",
                "Durum": durum,
                "Süre": round(time.perf_counter() - baslangic, 2),
                "_belge_tipi": "tc",
                "_belge_gecerli": None,
                "_kart_bulundu": True,
                "_kurtarildi": True,
                "_kaynak_yol": yol,
                "_koseler": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                "_ad_conf": float(ocr_sonuc.get("ad_conf", 0.0) or 0.0),
                "_soyad_conf": float(ocr_sonuc.get("soyad_conf", 0.0) or 0.0),
                "_preview": blok,
                "_debug_resmi": ocr_sonuc.get("debug_resmi"),
                "_kart_debug": None,
                "_kart_sonuc": {
                    "belge_tipi": "tc",
                    "mesaj": "Kart tespit edilemedi; blok doğrudan okundu (kurtarma).",
                    "arama_modu": "kurtarma",
                    "referans_hatalari": {},
                },
                "_ocr_sonuc": {
                    "guven": ocr_sonuc.get("guven"),
                    "kimlik_no_conf": ocr_sonuc.get("kimlik_no_conf", 0.0),
                    "ad_conf": ocr_sonuc.get("ad_conf", 0.0),
                    "soyad_conf": ocr_sonuc.get("soyad_conf", 0.0),
                    "ocr_suresi": ocr_sonuc.get("ocr_suresi", 0.0),
                    "tum_ocr": ocr_sonuc.get("tum_ocr", []) if self.debug else [],
                },
            })

        return satirlar

    def _satir_olustur(self, yol, sayfa_no, resim, kart_sonuc, sira, kart_sayisi,
                       tespit_hatasi, ek_sure, ek_bilgi=None):
        """Tek bir kimlik (sayfadaki bir kart) için OCR'ı çalıştırıp tablo
        satırını üretir. Sayfada hiç kart bulunamadıysa da tek bir
        "bulunamadı" satırı üretilir."""
        baslangic = time.perf_counter()

        hata = tespit_hatasi
        belge_tipi = "bilinmiyor"
        kart = None
        ocr_sonuc = {}

        if kart_sonuc.get("basarili", False):
            kart = kart_sonuc.get("kart")
            belge_tipi = kart_sonuc.get("belge_tipi", "tc")
        else:
            eslesen = kart_sonuc.get("belge_tipi")
            if eslesen in {"tc", "eski_tc", "gocmen"}:
                belge_tipi = eslesen
            if not hata and kart_sonuc:
                hata = kart_sonuc.get("mesaj") or "Kart tespit edilemedi."

        # Streamlit backend'indeki kritik bağlantı:
        # OCR yalnızca tespit edilen/düzeltilen kart üzerinde ve belge_tipi ile çalışır.
        if kart is not None:
            try:
                ocr_sonuc = bilgileri_cimbizla(
                    kart,
                    sayfa_no=sayfa_no,
                    debug=self.debug,
                    belge_tipi=belge_tipi,
                    derin=self.derin,
                )
                if ocr_sonuc.get("hata"):
                    hata = str(ocr_sonuc["hata"])
            except Exception as e:
                hata = f"OCR hatası: {e}"

        kimlik_no = ocr_sonuc.get("tc_no", "Bulunamadi")
        ad = ocr_sonuc.get("ad", "Bulunamadi")
        soyad = ocr_sonuc.get("soyad", "Bulunamadi")
        bitis = ocr_sonuc.get("bitis_tarihi", "")
        belge_gecerli = ocr_sonuc.get("belge_gecerli")
        ad_conf = float(ocr_sonuc.get("ad_conf", 0.0) or 0.0)
        soyad_conf = float(ocr_sonuc.get("soyad_conf", 0.0) or 0.0)

        # Numara okundu ama checksum'ı tutmuyor: kaybetmek yerine
        # "doğrulanamadı" diye gösteriyoruz, kullanıcı karta bakıp düzeltebilsin.
        tc_dogrulandi = bool(ocr_sonuc.get("tc_dogrulandi", True))
        tc_supheli = kimlik_no != "Bulunamadi" and not tc_dogrulandi

        excel_adi = (ek_bilgi or {}).get("excel_adi") or ""

        eksikler = []
        if kart is None:
            eksikler.append("Kimlik")
        else:
            if kimlik_no == "Bulunamadi":
                eksikler.append("Kimlik No")
            if ad == "Bulunamadi":
                eksikler.append("Ad")
            if soyad == "Bulunamadi":
                eksikler.append("Soyad")
            if belge_tipi == "gocmen" and not bitis:
                eksikler.append("Bitiş Tarihi")

        if hata:
            durum = hata
        elif eksikler:
            durum = "Eksik alan: " + ", ".join(eksikler)
        elif tc_supheli:
            durum = "Kimlik No doğrulanamadı — kontrol edin"
        else:
            durum = "Başarılı"

        if tc_supheli and eksikler:
            durum += " (Kimlik No doğrulanamadı)"

        belge_yazi = {
            "tc": "T.C. Kimlik",
            "eski_tc": "Eski T.C. Kimlik",
            "gocmen": "Göçmen / Yabancı",
        }.get(belge_tipi, "Bilinmiyor")

        if belge_tipi == "gocmen":
            if belge_gecerli is False:
                gecerlilik = "GEÇERSİZ"
            elif belge_gecerli is True:
                gecerlilik = "Geçerli"
            else:
                gecerlilik = "Kontrol Edilemedi"
        else:
            gecerlilik = "-"

        return {
            "Dosya": os.path.basename(yol),
            "Sayfa": sayfa_no if sayfa_no is not None else "-",
            "Kart": f"{sira}/{kart_sayisi}" if kart_sayisi else "-",
            "Belge Türü": belge_yazi,
            "Kimlik No": kimlik_no,
            "Ad": ad,
            "Soyad": soyad,
            "Bitiş Tarihi": bitis if belge_tipi == "gocmen" else "-",
            "Geçerlilik": gecerlilik,
            "Durum": durum,
            "Süre": round(time.perf_counter() - baslangic + ek_sure, 2),
            "_belge_tipi": belge_tipi,
            "_belge_gecerli": belge_gecerli,
            "_kart_bulundu": kart is not None,
            "_tc_supheli": tc_supheli,
            "_excel_adi": excel_adi,
            "_hata": hata,
            # Sayfanın tamamını işaretli göstermek için: kaynak dosya + kartın
            # sayfa üzerindeki köşeleri. Sayfa görüntüsü saklanmıyor, gerektiğinde
            # dosyadan yeniden üretiliyor (yüzlerce sayfada bellek şişmesin).
            "_kaynak_yol": yol,
            "_koseler": (kart_sonuc.get("koseler").tolist()
                         if kart_sonuc.get("koseler") is not None else None),
            "_ad_conf": ad_conf,
            "_soyad_conf": soyad_conf,
            "_preview": kart if kart is not None else resim,
            "_debug_resmi": ocr_sonuc.get("debug_resmi"),
            "_kart_debug": kart_sonuc.get("debug_resmi"),
            "_kart_sonuc": {
                "belge_tipi": kart_sonuc.get("belge_tipi"),
                "kart_sirasi": kart_sonuc.get("kart_sirasi", sira),
                "kart_sayisi": kart_sonuc.get("kart_sayisi", kart_sayisi),
                "arama_modu": kart_sonuc.get("arama_modu"),
                "duzeltildi": kart_sonuc.get("duzeltildi", False),
                "duzeltme_yontemi": kart_sonuc.get("duzeltme_yontemi"),
                "fallback": kart_sonuc.get("fallback", False),
                "iyi_eslesme": kart_sonuc.get("iyi_eslesme", 0),
                "inlier": kart_sonuc.get("inlier", 0),
                "inlier_orani": kart_sonuc.get("inlier_orani", 0.0),
                "skor": kart_sonuc.get("skor", 0.0),
                "mesaj": kart_sonuc.get("mesaj", ""),
                "aday_sirasi": kart_sonuc.get("aday_sirasi"),
                "referans_hatalari": kart_sonuc.get("referans_hatalari", {}),
            },
            "_ocr_sonuc": {
                "guven": ocr_sonuc.get("guven"),
                "tc_dogrulandi": tc_dogrulandi,
                "tc_kaynak": ocr_sonuc.get("tc_kaynak"),
                "kimlik_no_conf": ocr_sonuc.get("kimlik_no_conf", 0.0),
                "ad_conf": ocr_sonuc.get("ad_conf", 0.0),
                "soyad_conf": ocr_sonuc.get("soyad_conf", 0.0),
                "baslangic_tarihi": ocr_sonuc.get("baslangic_tarihi", ""),
                "bitis_tarihi": ocr_sonuc.get("bitis_tarihi", ""),
                "gecerlilik_durumu": ocr_sonuc.get("gecerlilik_durumu"),
                "ocr_suresi": ocr_sonuc.get("ocr_suresi", 0.0),
                "hizli_yol_kullanildi": ocr_sonuc.get("hizli_yol_kullanildi"),
                "hizli_hucre_ocr_suresi": ocr_sonuc.get("hizli_hucre_ocr_suresi", 0.0),
                "detector_fallback_kullanildi": ocr_sonuc.get("detector_fallback_kullanildi"),
                "detector_ocr_suresi": ocr_sonuc.get("detector_ocr_suresi", 0.0),
                "fallback_denenen_alanlar": ocr_sonuc.get("fallback_denenen_alanlar", []),
                "fallback_kullanilan_alanlar": ocr_sonuc.get("fallback_kullanilan_alanlar", []),
                "fallback_ocr_suresi": ocr_sonuc.get("fallback_ocr_suresi", 0.0),
                "tum_ocr": ocr_sonuc.get("tum_ocr", []) if self.debug else [],
            },
        }


