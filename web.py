import streamlit as st
import cv2
import numpy as np
import os
import pymupdf
import pandas as pd

from io import BytesIO
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font

from goruntu_isleme import kart_tespit_et_ve_duzelt
from metin_ayiklama import bilgileri_cimbizla


# =========================================================
# DOSYA -> NUMPY (DİSK YOK)
# =========================================================
#
# Eski akış: yükleneni diske yaz -> (PDF ise) her sayfayı PNG olarak TEKRAR
# diske yaz -> np.fromfile + cv2.imdecode ile GERİ oku. Bu round-trip'lerin
# hepsi gereksiz disk I/O'suydu — "dosyayı alma" süresinin büyük kısmı buradan
# geliyordu. Şimdi her şey bellekte kalıyor:
#   - PDF: pymupdf.open(stream=...) ile diskte dosya oluşturmadan açılır,
#     her sayfanın pixmap'i doğrudan numpy dizisine çevrilir (pix.samples).
#   - Görsel: yüklenen bytes doğrudan cv2.imdecode'a verilir.
# PNG kaydet/oku ile bellekten dönüştürme YÖNTEMLERİ piksel-özdeş sonuç
# verir (lossless PNG) — sadece disk turu eksilmiş oluyor.

PDF_RENDER_DPI = 200  # 300'den düşürüldü: goruntu_isleme zaten en uzun kenarı
                       # MAX_CALISMA_BOYUTU=1800'e indiriyor; 200dpi'de A4
                       # sayfası hâlâ ~2300px (1800'ün üstünde) çıkıyor, yani
                       # doğrulukta kayıp yok, sadece render/decode daha hızlı.


def _pixmap_to_bgr(pix):
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if pix.n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if pix.n == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _yuklenen_dosyayi_goruntulere_ayir(dosya):
    """Bir yüklemeyi (görsel ya da PDF) diske hiç dokunmadan {isim, resim,
    sayfa_no, pdf_mi} sözlüklerinden oluşan bir listeye çevirir."""
    ad_lower = dosya.name.lower()

    if ad_lower.endswith(".pdf"):
        sonuc = []
        veri = dosya.getvalue()
        doc = pymupdf.open(stream=veri, filetype="pdf")
        for sayfa_no, sayfa in enumerate(doc, start=1):
            pix = sayfa.get_pixmap(dpi=PDF_RENDER_DPI)
            sonuc.append({
                "dosya_adi": dosya.name,
                "gorunen_isim": f"{dosya.name} - Sayfa {sayfa_no}",
                "sayfa_no": sayfa_no,
                "pdf_mi": True,
                "resim": _pixmap_to_bgr(pix),
            })
        doc.close()
        return sonuc

    veri = np.frombuffer(dosya.getvalue(), dtype=np.uint8)
    resim = cv2.imdecode(veri, cv2.IMREAD_COLOR)
    return [{
        "dosya_adi": dosya.name, "gorunen_isim": dosya.name,
        "sayfa_no": None, "pdf_mi": False, "resim": resim,
    }]


# =========================================================
# SAYFA
# =========================================================

st.set_page_config(page_title="Kimlik Okuyucu", page_icon="🪪", layout="wide")
st.title("🪪 Kimlik Kartı Okuyucu")
st.caption("Kimlik kartlarından T.C. Kimlik No, Ad ve Soyad bilgilerini çıkarır.")


# =========================================================
# SESSION
# =========================================================

if "sonuc_df" not in st.session_state:
    st.session_state.sonuc_df = None
if "excel_data" not in st.session_state:
    st.session_state.excel_data = None
if "bulunamayanlar" not in st.session_state:
    st.session_state.bulunamayanlar = []


# =========================================================
# REFERANS
# =========================================================

REFERANS_YOLU = "referans_kimlik.jpg"

if not os.path.exists(REFERANS_YOLU):
    st.error("referans_kimlik.jpg bulunamadı.")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("Ayarlar")
    debug_modu = st.checkbox("OCR debug", value=False)
    kart_siniri_goster = st.checkbox("Kart sınırını göster", value=False)
    feature_debug = st.checkbox("SIFT eşleşmelerini göster", value=False)
    ham_ocr_goster = st.checkbox("Ham OCR sonuçlarını göster", value=False)

debug_aktif = any([debug_modu, kart_siniri_goster, feature_debug, ham_ocr_goster])


# =========================================================
# DOSYA
# =========================================================

dosyalar = st.file_uploader(
    "Kimlik görsellerini veya PDF dosyalarını yükle",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True,
)


# =========================================================
# ÖNİZLEME
# =========================================================

def onizleme_hazirla(resim):
    if resim is None:
        return None

    h, w = resim.shape[:2]
    if max(h, w) > 1000:
        oran = 1000 / max(h, w)
        resim = cv2.resize(resim, None, fx=oran, fy=oran, interpolation=cv2.INTER_AREA)

    basarili, encoded = cv2.imencode(".jpg", resim, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return encoded.tobytes() if basarili else None


# =========================================================
# EXCEL
# =========================================================

def excel_olustur(df, meta):
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)

    wb = load_workbook(buffer)
    ws = wb.active
    ws.title = "Kimlik Sonuçları"

    kirmizi = PatternFill("solid", fgColor="FFC7CE")
    sari = PatternFill("solid", fgColor="FFF2CC")

    for hucre in ws[1]:
        hucre.font = Font(bold=True)

    basliklar = {hucre.value: hucre.column for hucre in ws[1]}
    tc_col, ad_col, soyad_col = basliklar["T.C."], basliklar["Ad"], basliklar["Soyad"]

    for satir_no, item in enumerate(meta, start=2):
        if not item["tc_bulundu"]:
            ws.cell(satir_no, tc_col).fill = kirmizi

        if not item["ad_bulundu"]:
            ws.cell(satir_no, ad_col).fill = kirmizi
        elif item["ad_conf"] < 0.50:
            ws.cell(satir_no, ad_col).fill = kirmizi
        elif item["ad_conf"] < 0.70:
            ws.cell(satir_no, ad_col).fill = sari

        if not item["soyad_bulundu"]:
            ws.cell(satir_no, soyad_col).fill = kirmizi
        elif item["soyad_conf"] < 0.50:
            ws.cell(satir_no, soyad_col).fill = kirmizi
        elif item["soyad_conf"] < 0.70:
            ws.cell(satir_no, soyad_col).fill = sari

    for col, width in zip("ABCD", (12, 18, 25, 25)):
        ws.column_dimensions[col].width = width

    sonuc = BytesIO()
    wb.save(sonuc)
    sonuc.seek(0)
    return sonuc.getvalue()


# =========================================================
# İŞLEM
# =========================================================

if dosyalar and st.button("🚀 İşlemi Başlat", type="primary"):
    st.session_state.sonuc_df = None
    st.session_state.excel_data = None
    st.session_state.bulunamayanlar = []

    durum = st.empty()
    durum.write("Dosyalar hazırlanıyor...")

    # ---- Dosyaları hazırla (diske hiç dokunmadan) --------------------------
    islenecekler = []
    for dosya in dosyalar:
        islenecekler.extend(_yuklenen_dosyayi_goruntulere_ayir(dosya))

    # ---- Process -----------------------------------------------------------
    tum_sonuclar, meta, bulunamayanlar = [], [], []
    toplam = len(islenecekler)
    progress = st.progress(0)

    for index, bilgi in enumerate(islenecekler):
        sayfa_no, pdf_mi = bilgi["sayfa_no"], bilgi["pdf_mi"]
        durum.write(f"Kimlikler işleniyor... {index + 1}/{toplam}")

        resim = bilgi["resim"]
        tc_no, ad, soyad = "Bulunamadi", "Bulunamadi", "Bulunamadi"
        ad_conf, soyad_conf = 0.0, 0.0
        kart, ocr_sonuc = None, {}

        # ---- Kart -----------------------------------------------------------
        if resim is not None:
            try:
                kart_sonuc = kart_tespit_et_ve_duzelt(
                    resim, REFERANS_YOLU, debug_match=feature_debug, debug_kart=kart_siniri_goster
                )
            except Exception:
                kart_sonuc = {"basarili": False}

            if kart_sonuc.get("basarili", False):
                kart = kart_sonuc.get("kart")

        # ---- OCR -----------------------------------------------------------
        sonuc_sayfa_no = sayfa_no
        if kart is not None:
            try:
                ocr_sonuc = bilgileri_cimbizla(kart, sayfa_no=sayfa_no, debug=debug_modu or ham_ocr_goster)
            except Exception:
                ocr_sonuc = {}

            sonuc_sayfa_no = ocr_sonuc.get("sayfa_no", sayfa_no)
            tc_no = ocr_sonuc.get("tc_no", "Bulunamadi")
            ad = ocr_sonuc.get("ad", "Bulunamadi")
            soyad = ocr_sonuc.get("soyad", "Bulunamadi")
            ad_conf = float(ocr_sonuc.get("ad_conf", 0.0))
            soyad_conf = float(ocr_sonuc.get("soyad_conf", 0.0))

        # ---- Tablo -----------------------------------------------------------
        tablo_sayfa = sonuc_sayfa_no if pdf_mi else "-"
        tum_sonuclar.append({"Sayfa No": tablo_sayfa, "T.C.": tc_no, "Ad": ad, "Soyad": soyad})

        # ---- Meta -----------------------------------------------------------
        tc_bulundu = tc_no != "Bulunamadi"
        ad_bulundu = ad != "Bulunamadi"
        soyad_bulundu = soyad != "Bulunamadi"
        meta.append({
            "tc_bulundu": tc_bulundu, "ad_bulundu": ad_bulundu, "soyad_bulundu": soyad_bulundu,
            "ad_conf": ad_conf, "soyad_conf": soyad_conf,
        })

        # ---- Bulunamayan -----------------------------------------------------
        eksikler = [
            etiket for etiket, bulundu in
            [("T.C.", tc_bulundu), ("Ad", ad_bulundu), ("Soyad", soyad_bulundu)]
            if not bulundu
        ]
        if eksikler:
            bulunamayanlar.append({
                "pdf_mi": pdf_mi, "sayfa_no": sonuc_sayfa_no, "dosya": bilgi["dosya_adi"],
                "eksik": ", ".join(eksikler),
                "resim": onizleme_hazirla(kart if kart is not None else resim),
            })

        # ---- Debug -----------------------------------------------------------
        if debug_aktif:
            st.divider()
            st.write(f"### Sayfa {sayfa_no}" if pdf_mi else f"### {bilgi['dosya_adi']}")
            st.write(f"T.C.: {tc_no}")
            st.write(f"Ad: {ad}")
            st.write(f"Soyad: {soyad}")

            if debug_modu and ocr_sonuc.get("debug_resmi") is not None:
                st.image(cv2.cvtColor(ocr_sonuc["debug_resmi"], cv2.COLOR_BGR2RGB))

            if ham_ocr_goster:
                with st.expander("Ham OCR"):
                    for item in ocr_sonuc.get("tum_ocr", []):
                        st.write(f'{item["text"]} — {item["conf"]}')

        progress.progress((index + 1) / toplam)

    # ---- Sonuç -----------------------------------------------------------
    durum.success(f"✅ {toplam} sayfa işlendi.")

    df = pd.DataFrame(tum_sonuclar, columns=["Sayfa No", "T.C.", "Ad", "Soyad"])
    st.session_state.sonuc_df = df
    st.session_state.excel_data = excel_olustur(df, meta)
    st.session_state.bulunamayanlar = bulunamayanlar


# =========================================================
# SONUÇ TABLOSU
# =========================================================

if st.session_state.sonuc_df is not None:
    st.divider()
    st.header("📋 Sonuçlar")
    st.dataframe(st.session_state.sonuc_df, use_container_width=True, hide_index=True)

    st.download_button(
        "📥 Excel indir",
        data=st.session_state.excel_data,
        file_name="kimlik_sonuclari.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    bulunamayanlar = st.session_state.bulunamayanlar
    if bulunamayanlar:
        with st.expander(f"⚠️ Bulunamayan Kimlikler ({len(bulunamayanlar)})"):
            for item in bulunamayanlar:
                st.markdown("---")
                if item["pdf_mi"]:
                    st.subheader(f'📄 {item["dosya"]} — Sayfa {item["sayfa_no"]}')
                else:
                    st.subheader(f'🖼️ {item["dosya"]}')

                st.write(f'**Bulunamayan alanlar:** {item["eksik"]}')
                if item["resim"] is not None:
                    st.image(item["resim"], width=650)