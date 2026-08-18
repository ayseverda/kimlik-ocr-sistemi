import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import pymupdf
import pandas as pd
import time
import re

from io import BytesIO
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment

from goruntu_isleme import kart_tespit_et_ve_duzelt
from metin_ayiklama import bilgileri_cimbizla


# =========================================================
# SAYFA
# =========================================================

st.set_page_config(
    page_title="Kimlik Okuyucu",
    page_icon="🪪",
    layout="wide"
)

st.title("🪪 Kimlik Kartı Okuyucu")

st.caption(
    "Kimlik kartlarından T.C. Kimlik No, Ad ve Soyad bilgilerini çıkarır."
)


# =========================================================
# SESSION STATE
# =========================================================

if "sonuc_df" not in st.session_state:
    st.session_state.sonuc_df = None

if "excel_data" not in st.session_state:
    st.session_state.excel_data = None

if "bulunamayanlar" not in st.session_state:
    st.session_state.bulunamayanlar = []

if "guven_meta" not in st.session_state:
    st.session_state.guven_meta = []


# =========================================================
# REFERANS
# =========================================================

REFERANS_YOLU = "referans_kimlik.jpg"

if not os.path.exists(REFERANS_YOLU):

    st.error(
        "❌ referans_kimlik.jpg bulunamadı."
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Ayarlar")

    debug_modu = st.checkbox(
        "Debug görüntülerini göster",
        value=False
    )

    kart_siniri_goster = st.checkbox(
        "Kart sınırını göster",
        value=False
    )

    feature_debug = st.checkbox(
        "SIFT eşleşmelerini göster",
        value=False
    )

    ham_ocr_goster = st.checkbox(
        "Ham OCR sonuçlarını göster",
        value=False
    )


debug_aktif = (
    debug_modu
    or kart_siniri_goster
    or feature_debug
    or ham_ocr_goster
)


# =========================================================
# DOSYA YÜKLE
# =========================================================

dosyalar = st.file_uploader(
    "Kimlik görsellerini veya PDF dosyalarını yükle",
    type=[
        "jpg",
        "jpeg",
        "png",
        "pdf"
    ],
    accept_multiple_files=True
)


# =========================================================
# SAYFA NO
# =========================================================

def sayfa_no_bul(isim):

    eslesme = re.search(
        r"Sayfa\s+(\d+)",
        isim,
        re.IGNORECASE
    )

    if eslesme:

        return int(
            eslesme.group(1)
        )

    # Direkt resim yüklenmişse PDF sayfası yok
    return "-"


# =========================================================
# ÖNİZLEME RESMİ
# =========================================================

def onizleme_hazirla(resim):
    """
    Bulunamayan kimlikler için session_state içerisinde
    dev görüntü tutmamak adına küçük JPEG oluşturur.
    """

    if resim is None:
        return None

    h, w = resim.shape[:2]

    maksimum = 1000

    if max(h, w) > maksimum:

        oran = (
            maksimum
            /
            max(h, w)
        )

        resim = cv2.resize(
            resim,
            None,
            fx=oran,
            fy=oran,
            interpolation=cv2.INTER_AREA
        )

    basarili, encoded = cv2.imencode(
        ".jpg",
        resim,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            80
        ]
    )

    if not basarili:
        return None

    return encoded.tobytes()


# =========================================================
# EXCEL
# =========================================================

def excel_olustur(
    df,
    guven_meta
):

    buffer = BytesIO()

    df.to_excel(
        buffer,
        index=False,
        engine="openpyxl"
    )

    buffer.seek(0)

    wb = load_workbook(
        buffer
    )

    ws = wb.active

    ws.title = "Kimlik Sonuçları"


    # =====================================================
    # RENKLER
    # =====================================================

    kirmizi = PatternFill(
        fill_type="solid",
        fgColor="FFC7CE"
    )

    sari = PatternFill(
        fill_type="solid",
        fgColor="FFF2CC"
    )

    yesil = PatternFill(
        fill_type="solid",
        fgColor="E2F0D9"
    )


    # =====================================================
    # BAŞLIK
    # =====================================================

    for hucre in ws[1]:

        hucre.font = Font(
            bold=True
        )

        hucre.alignment = Alignment(
            horizontal="center"
        )


    basliklar = {
        cell.value: cell.column
        for cell in ws[1]
    }

    tc_col = basliklar["T.C."]
    ad_col = basliklar["Ad"]
    soyad_col = basliklar["Soyad"]


    # =====================================================
    # RENKLENDİR
    # =====================================================

    for index, meta in enumerate(
        guven_meta,
        start=2
    ):

        tc_hucre = ws.cell(
            index,
            tc_col
        )

        ad_hucre = ws.cell(
            index,
            ad_col
        )

        soyad_hucre = ws.cell(
            index,
            soyad_col
        )


        # -------------------------------------------------
        # TC
        # -------------------------------------------------

        if meta["tc_bulundu"]:

            tc_hucre.fill = yesil

        else:

            tc_hucre.fill = kirmizi


        # -------------------------------------------------
        # AD
        # -------------------------------------------------

        ad_conf = meta["ad_conf"]

        if not meta["ad_bulundu"]:

            ad_hucre.fill = kirmizi

        elif ad_conf < 0.50:

            ad_hucre.fill = kirmizi

        elif ad_conf < 0.70:

            ad_hucre.fill = sari


        # -------------------------------------------------
        # SOYAD
        # -------------------------------------------------

        soyad_conf = meta[
            "soyad_conf"
        ]

        if not meta["soyad_bulundu"]:

            soyad_hucre.fill = kirmizi

        elif soyad_conf < 0.50:

            soyad_hucre.fill = kirmizi

        elif soyad_conf < 0.70:

            soyad_hucre.fill = sari


    # =====================================================
    # GENİŞLİKLER
    # =====================================================

    genislikler = {
        "A": 12,
        "B": 18,
        "C": 25,
        "D": 25
    }

    for kolon, genislik in (
        genislikler.items()
    ):

        ws.column_dimensions[
            kolon
        ].width = genislik


    # =====================================================
    # ÇIKTI
    # =====================================================

    sonuc = BytesIO()

    wb.save(
        sonuc
    )

    sonuc.seek(0)

    return sonuc.getvalue()


# =========================================================
# İŞLEM
# =========================================================

if dosyalar:

    if st.button(
        "🚀 İşlemi Başlat",
        type="primary"
    ):

        # Eski sonucu temizle
        st.session_state.sonuc_df = None
        st.session_state.excel_data = None
        st.session_state.bulunamayanlar = []
        st.session_state.guven_meta = []


        gecici_klasor = (
            tempfile.mkdtemp()
        )

        islenecekler = []

        tum_sonuclar = []

        guven_meta = []

        bulunamayanlar = []


        # =================================================
        # DOSYALARI HAZIRLA
        # =================================================

        durum = st.empty()

        durum.write(
            "Dosyalar hazırlanıyor..."
        )


        for dosya in dosyalar:

            dosya_yolu = os.path.join(
                gecici_klasor,
                dosya.name
            )


            with open(
                dosya_yolu,
                "wb"
            ) as f:

                f.write(
                    dosya.getbuffer()
                )


            # =============================================
            # PDF
            # =============================================

            if dosya.name.lower().endswith(
                ".pdf"
            ):

                try:

                    doc = pymupdf.open(
                        dosya_yolu
                    )


                    for sayfa_no, sayfa in enumerate(
                        doc,
                        start=1
                    ):

                        pix = (
                            sayfa.get_pixmap(
                                dpi=300
                            )
                        )


                        sayfa_yolu = (
                            os.path.join(
                                gecici_klasor,
                                (
                                    f"{os.path.splitext(dosya.name)[0]}"
                                    f"_sayfa_{sayfa_no}.png"
                                )
                            )
                        )


                        pix.save(
                            sayfa_yolu
                        )


                        islenecekler.append({
                            "isim": (
                                f"{dosya.name} "
                                f"- Sayfa {sayfa_no}"
                            ),

                            "sayfa_no":
                                sayfa_no,

                            "yol":
                                sayfa_yolu
                        })


                    doc.close()


                except Exception as e:

                    st.error(
                        f"{dosya.name} PDF hatası: {e}"
                    )


            # =============================================
            # RESİM
            # =============================================

            else:

                islenecekler.append({
                    "isim":
                        dosya.name,

                    "sayfa_no":
                        "-",

                    "yol":
                        dosya_yolu
                })


        if not islenecekler:

            durum.empty()

            st.warning(
                "İşlenecek görüntü bulunamadı."
            )

            st.stop()


        # =================================================
        # PROGRESS
        # =================================================

        toplam = len(
            islenecekler
        )

        progress = st.progress(0)


        # =================================================
        # HER SAYFA
        # =================================================

        for index, bilgi in enumerate(
            islenecekler
        ):

            dosya_adi = (
                bilgi["isim"]
            )

            sayfa_no = (
                bilgi["sayfa_no"]
            )

            yol = (
                bilgi["yol"]
            )


            durum.write(
                (
                    f"Kimlikler işleniyor... "
                    f"{index + 1} / {toplam}"
                )
            )


            if debug_aktif:

                st.divider()

                st.subheader(
                    f"📄 {dosya_adi}"
                )


            # =============================================
            # RESİM OKU
            # =============================================

            try:

                veri = np.fromfile(
                    yol,
                    dtype=np.uint8
                )

                resim = cv2.imdecode(
                    veri,
                    cv2.IMREAD_COLOR
                )


            except Exception:

                resim = None


            # =============================================
            # DEFAULT
            # =============================================

            tc_no = "Bulunamadi"
            ad = "Bulunamadi"
            soyad = "Bulunamadi"

            ad_conf = 0.0
            soyad_conf = 0.0


            # =============================================
            # RESİM OKUNAMADI
            # =============================================

            if resim is None:

                tum_sonuclar.append({
                    "Sayfa No":
                        sayfa_no,

                    "T.C.":
                        tc_no,

                    "Ad":
                        ad,

                    "Soyad":
                        soyad
                })


                guven_meta.append({
                    "tc_bulundu":
                        False,

                    "ad_bulundu":
                        False,

                    "soyad_bulundu":
                        False,

                    "ad_conf":
                        0.0,

                    "soyad_conf":
                        0.0
                })


                bulunamayanlar.append({
                    "sayfa_no":
                        sayfa_no,

                    "dosya":
                        dosya_adi,

                    "eksik":
                        "T.C., Ad, Soyad",

                    "resim":
                        None
                })


                progress.progress(
                    (index + 1)
                    /
                    toplam
                )

                continue


            # =============================================
            # KART TESPİTİ
            # =============================================

            try:

                kart_sonuc = (
                    kart_tespit_et_ve_duzelt(
                        resim,
                        REFERANS_YOLU,

                        debug_match=(
                            feature_debug
                        ),

                        debug_kart=(
                            kart_siniri_goster
                        )
                    )
                )

            except Exception as e:

                kart_sonuc = {
                    "basarili": False
                }

                if debug_aktif:

                    st.error(
                        f"Kart tespit hatası: {e}"
                    )


            # =============================================
            # KART BULUNAMADI
            # =============================================

            if not kart_sonuc.get(
                "basarili",
                False
            ):

                tum_sonuclar.append({
                    "Sayfa No":
                        sayfa_no,

                    "T.C.":
                        tc_no,

                    "Ad":
                        ad,

                    "Soyad":
                        soyad
                })


                guven_meta.append({
                    "tc_bulundu":
                        False,

                    "ad_bulundu":
                        False,

                    "soyad_bulundu":
                        False,

                    "ad_conf":
                        0.0,

                    "soyad_conf":
                        0.0
                })


                bulunamayanlar.append({
                    "sayfa_no":
                        sayfa_no,

                    "dosya":
                        dosya_adi,

                    "eksik":
                        "Kart tespit edilemedi",

                    "resim":
                        onizleme_hazirla(
                            resim
                        )
                })


                progress.progress(
                    (index + 1)
                    /
                    toplam
                )

                continue


            kart = kart_sonuc.get(
                "kart"
            )


            # =============================================
            # OCR
            # =============================================

            try:

                ocr_sonuc = (
                    bilgileri_cimbizla(
                        kart,

                        debug=(
                            debug_modu
                            or
                            ham_ocr_goster
                        )
                    )
                )


                tc_no = ocr_sonuc.get(
                    "tc_no",
                    "Bulunamadi"
                )

                ad = ocr_sonuc.get(
                    "ad",
                    "Bulunamadi"
                )

                soyad = ocr_sonuc.get(
                    "soyad",
                    "Bulunamadi"
                )

                ad_conf = float(
                    ocr_sonuc.get(
                        "ad_conf",
                        0.0
                    )
                )

                soyad_conf = float(
                    ocr_sonuc.get(
                        "soyad_conf",
                        0.0
                    )
                )


            except Exception as e:

                if debug_aktif:

                    st.error(
                        f"OCR hatası: {e}"
                    )

                ocr_sonuc = {}


            # =============================================
            # EXCEL SATIRI
            # =============================================

            tum_sonuclar.append({
                "Sayfa No":
                    sayfa_no,

                "T.C.":
                    tc_no,

                "Ad":
                    ad,

                "Soyad":
                    soyad
            })


            # =============================================
            # CONFIDENCE META
            # =============================================

            tc_bulundu = (
                tc_no
                !=
                "Bulunamadi"
            )

            ad_bulundu = (
                ad
                !=
                "Bulunamadi"
            )

            soyad_bulundu = (
                soyad
                !=
                "Bulunamadi"
            )


            guven_meta.append({
                "tc_bulundu":
                    tc_bulundu,

                "ad_bulundu":
                    ad_bulundu,

                "soyad_bulundu":
                    soyad_bulundu,

                "ad_conf":
                    ad_conf,

                "soyad_conf":
                    soyad_conf
            })


            # =============================================
            # EKSİK ALANLAR
            # =============================================

            eksikler = []


            if not tc_bulundu:

                eksikler.append(
                    "T.C."
                )


            if not ad_bulundu:

                eksikler.append(
                    "Ad"
                )


            if not soyad_bulundu:

                eksikler.append(
                    "Soyad"
                )


            # =============================================
            # BULUNAMAYANLARA EKLE
            # =============================================

            if eksikler:

                # Mümkünse düzeltilmiş kartı göster.
                # Kart yoksa orijinal sayfayı göster.

                gosterilecek_resim = (
                    kart
                    if kart is not None
                    else resim
                )


                bulunamayanlar.append({
                    "sayfa_no":
                        sayfa_no,

                    "dosya":
                        dosya_adi,

                    "eksik":
                        ", ".join(
                            eksikler
                        ),

                    "resim":
                        onizleme_hazirla(
                            gosterilecek_resim
                        )
                })


            # =============================================
            # DEBUG
            # =============================================

            if debug_aktif:

                st.write(
                    f"**T.C.:** {tc_no}"
                )

                st.write(
                    f"**Ad:** {ad}"
                )

                st.write(
                    f"**Soyad:** {soyad}"
                )


                if kart_siniri_goster:

                    kart_debug = (
                        kart_sonuc.get(
                            "debug_resmi"
                        )
                    )

                    if kart_debug is not None:

                        st.image(
                            cv2.cvtColor(
                                kart_debug,
                                cv2.COLOR_BGR2RGB
                            ),
                            caption="Kart sınırı",
                            use_container_width=True
                        )


                if debug_modu:

                    ocr_debug = (
                        ocr_sonuc.get(
                            "debug_resmi"
                        )
                    )

                    if ocr_debug is not None:

                        st.image(
                            cv2.cvtColor(
                                ocr_debug,
                                cv2.COLOR_BGR2RGB
                            ),
                            caption="TC / Ad / Soyad",
                            use_container_width=True
                        )


                if feature_debug:

                    match_debug = (
                        kart_sonuc.get(
                            "match_debug"
                        )
                    )

                    if match_debug is not None:

                        with st.expander(
                            "SIFT eşleşmeleri"
                        ):

                            st.image(
                                cv2.cvtColor(
                                    match_debug,
                                    cv2.COLOR_BGR2RGB
                                ),
                                use_container_width=True
                            )


                if ham_ocr_goster:

                    with st.expander(
                        "Ham EasyOCR sonuçları"
                    ):

                        for item in (
                            ocr_sonuc.get(
                                "tum_ocr",
                                []
                            )
                        ):

                            st.write(
                                (
                                    f'`{item.get("text", "")}` '
                                    f'— '
                                    f'{item.get("conf", 0)}'
                                )
                            )


            # =============================================
            # PROGRESS
            # =============================================

            progress.progress(
                (index + 1)
                /
                toplam
            )


        # =================================================
        # TAMAMLANDI
        # =================================================

        progress.progress(
            1.0
        )

        durum.success(
            f"✅ {toplam} sayfa işlendi."
        )


        # =================================================
        # DATAFRAME
        # =================================================

        df = pd.DataFrame(
            tum_sonuclar,
            columns=[
                "Sayfa No",
                "T.C.",
                "Ad",
                "Soyad"
            ]
        )


        # =================================================
        # EXCEL
        # =================================================

        try:

            excel_data = excel_olustur(
                df,
                guven_meta
            )

        except Exception as e:

            excel_data = None

            st.error(
                f"Excel oluşturulamadı: {e}"
            )


        # =================================================
        # SESSION
        # =================================================

        st.session_state.sonuc_df = (
            df
        )

        st.session_state.excel_data = (
            excel_data
        )

        st.session_state.bulunamayanlar = (
            bulunamayanlar
        )

        st.session_state.guven_meta = (
            guven_meta
        )


# =========================================================
# SONUÇ TABLOSU
# =========================================================

if (
    st.session_state.sonuc_df
    is not None
):

    st.divider()

    st.header(
        "📋 Sonuçlar"
    )


    st.dataframe(
        st.session_state.sonuc_df,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # EXCEL İNDİR
    # =====================================================

    if (
        st.session_state.excel_data
        is not None
    ):

        st.download_button(
            label="📥 Excel indir",

            data=(
                st.session_state.excel_data
            ),

            file_name=(
                "kimlik_sonuclari.xlsx"
            ),

            mime=(
                "application/"
                "vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        )


    # =====================================================
    # BULUNAMAYAN KİMLİKLER
    # =====================================================

    bulunamayanlar = (
        st.session_state.bulunamayanlar
    )


    if bulunamayanlar:

        with st.expander(
            (
                f"⚠️ Bulunamayan Kimlikler "
                f"({len(bulunamayanlar)})"
            ),
            expanded=False
        ):

            for item in bulunamayanlar:

                st.markdown(
                    "---"
                )


                # PDF ise sayfa numarası
                if item["sayfa_no"] != "-":

                    st.subheader(
                        f'📄 Sayfa {item["sayfa_no"]}'
                    )

                else:

                    st.subheader(
                        f'📄 {item["dosya"]}'
                    )


                st.write(
                    (
                        "**Bulunamayan alan:** "
                        f'{item["eksik"]}'
                    )
                )


                # =========================================
                # RESİM
                # =========================================

                if item["resim"] is not None:

                    st.image(
                        item["resim"],
                        width=650
                    )

                else:

                    st.info(
                        "Bu sayfanın görüntüsü oluşturulamadı."
                    )