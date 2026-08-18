import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import pymupdf
import pandas as pd

from io import BytesIO

from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font

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

st.title(
    "🪪 Kimlik Kartı Okuyucu"
)

st.caption(
    "Kimlik kartlarından T.C. Kimlik No, Ad ve Soyad bilgilerini çıkarır."
)


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

REFERANS_YOLU = (
    "referans_kimlik.jpg"
)


if not os.path.exists(
    REFERANS_YOLU
):

    st.error(
        "referans_kimlik.jpg bulunamadı."
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "Ayarlar"
    )


    debug_modu = st.checkbox(
        "OCR debug",
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


debug_aktif = any([
    debug_modu,
    kart_siniri_goster,
    feature_debug,
    ham_ocr_goster
])


# =========================================================
# DOSYA
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
# ÖNİZLEME
# =========================================================

def onizleme_hazirla(resim):

    if resim is None:

        return None


    h, w = (
        resim.shape[:2]
    )


    if max(h, w) > 1000:

        oran = (
            1000
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


    basarili, encoded = (
        cv2.imencode(
            ".jpg",
            resim,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                82
            ]
        )
    )


    if not basarili:

        return None


    return encoded.tobytes()


# =========================================================
# EXCEL
# =========================================================

def excel_olustur(
    df,
    meta
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

    ws.title = (
        "Kimlik Sonuçları"
    )


    # =====================================================
    # RENKLER
    # =====================================================

    kirmizi = PatternFill(
        "solid",
        fgColor="FFC7CE"
    )


    sari = PatternFill(
        "solid",
        fgColor="FFF2CC"
    )


    for hucre in ws[1]:

        hucre.font = Font(
            bold=True
        )


    basliklar = {
        hucre.value:
            hucre.column

        for hucre in ws[1]
    }


    tc_col = basliklar[
        "T.C."
    ]

    ad_col = basliklar[
        "Ad"
    ]

    soyad_col = basliklar[
        "Soyad"
    ]


    for satir_no, item in enumerate(
        meta,
        start=2
    ):

        # TC bulunamadı
        if not item["tc_bulundu"]:

            ws.cell(
                satir_no,
                tc_col
            ).fill = kirmizi


        # AD
        if not item["ad_bulundu"]:

            ws.cell(
                satir_no,
                ad_col
            ).fill = kirmizi


        elif item["ad_conf"] < 0.50:

            ws.cell(
                satir_no,
                ad_col
            ).fill = kirmizi


        elif item["ad_conf"] < 0.70:

            ws.cell(
                satir_no,
                ad_col
            ).fill = sari


        # SOYAD
        if not item["soyad_bulundu"]:

            ws.cell(
                satir_no,
                soyad_col
            ).fill = kirmizi


        elif item["soyad_conf"] < 0.50:

            ws.cell(
                satir_no,
                soyad_col
            ).fill = kirmizi


        elif item["soyad_conf"] < 0.70:

            ws.cell(
                satir_no,
                soyad_col
            ).fill = sari


    ws.column_dimensions[
        "A"
    ].width = 12

    ws.column_dimensions[
        "B"
    ].width = 18

    ws.column_dimensions[
        "C"
    ].width = 25

    ws.column_dimensions[
        "D"
    ].width = 25


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

        st.session_state.sonuc_df = None
        st.session_state.excel_data = None
        st.session_state.bulunamayanlar = []


        gecici_klasor = (
            tempfile.mkdtemp()
        )


        islenecekler = []


        # =================================================
        # DOSYALARI HAZIRLA
        # =================================================

        durum = st.empty()

        durum.write(
            "Dosyalar hazırlanıyor..."
        )


        for dosya in dosyalar:

            dosya_yolu = (
                os.path.join(
                    gecici_klasor,
                    dosya.name
                )
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

                doc = pymupdf.open(
                    dosya_yolu
                )


                for sayfa_no, sayfa in enumerate(
                    doc,
                    start=1
                ):

                    pix = sayfa.get_pixmap(
                        dpi=300
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
                        "dosya_adi":
                            dosya.name,

                        "gorunen_isim":
                            (
                                f"{dosya.name} "
                                f"- Sayfa {sayfa_no}"
                            ),

                        "sayfa_no":
                            sayfa_no,

                        "pdf_mi":
                            True,

                        "yol":
                            sayfa_yolu
                    })


                doc.close()


            # =============================================
            # NORMAL RESİM
            # =============================================

            else:

                islenecekler.append({
                    "dosya_adi":
                        dosya.name,

                    "gorunen_isim":
                        dosya.name,

                    "sayfa_no":
                        None,

                    "pdf_mi":
                        False,

                    "yol":
                        dosya_yolu
                })


        # =================================================
        # PROCESS
        # =================================================

        tum_sonuclar = []
        meta = []
        bulunamayanlar = []


        toplam = len(
            islenecekler
        )


        progress = st.progress(0)


        for index, bilgi in enumerate(
            islenecekler
        ):

            sayfa_no = (
                bilgi["sayfa_no"]
            )

            pdf_mi = (
                bilgi["pdf_mi"]
            )


            durum.write(
                (
                    f"Kimlikler işleniyor... "
                    f"{index + 1}/{toplam}"
                )
            )


            # =============================================
            # GÖRÜNTÜ
            # =============================================

            veri = np.fromfile(
                bilgi["yol"],
                dtype=np.uint8
            )


            resim = cv2.imdecode(
                veri,
                cv2.IMREAD_COLOR
            )


            tc_no = "Bulunamadi"
            ad = "Bulunamadi"
            soyad = "Bulunamadi"

            ad_conf = 0.0
            soyad_conf = 0.0


            kart = None
            ocr_sonuc = {}


            # =============================================
            # KART
            # =============================================

            if resim is not None:

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


                except Exception:

                    kart_sonuc = {
                        "basarili": False
                    }


                if kart_sonuc.get(
                    "basarili",
                    False
                ):

                    kart = (
                        kart_sonuc.get(
                            "kart"
                        )
                    )


            # =============================================
            # OCR
            # =============================================

            if kart is not None:

                try:

                    ocr_sonuc = (
                        bilgileri_cimbizla(
                            kart,

                            # EN ÖNEMLİ KISIM
                            sayfa_no=sayfa_no,

                            debug=(
                                debug_modu
                                or
                                ham_ocr_goster
                            )
                        )
                    )


                except Exception:

                    ocr_sonuc = {}


                # OCR'ın döndürdüğü sayfa numarası
                sonuc_sayfa_no = (
                    ocr_sonuc.get(
                        "sayfa_no",
                        sayfa_no
                    )
                )


                tc_no = (
                    ocr_sonuc.get(
                        "tc_no",
                        "Bulunamadi"
                    )
                )


                ad = (
                    ocr_sonuc.get(
                        "ad",
                        "Bulunamadi"
                    )
                )


                soyad = (
                    ocr_sonuc.get(
                        "soyad",
                        "Bulunamadi"
                    )
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


            else:

                sonuc_sayfa_no = (
                    sayfa_no
                )


            # =============================================
            # TABLO
            # =============================================

            # PDF ise gerçek sayfa no
            # normal resim ise "-"
            tablo_sayfa = (
                sonuc_sayfa_no
                if pdf_mi
                else "-"
            )


            tum_sonuclar.append({
                "Sayfa No":
                    tablo_sayfa,

                "T.C.":
                    tc_no,

                "Ad":
                    ad,

                "Soyad":
                    soyad
            })


            # =============================================
            # META
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


            meta.append({
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
            # BULUNAMAYAN
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


            if eksikler:

                bulunamayanlar.append({
                    "pdf_mi":
                        pdf_mi,

                    "sayfa_no":
                        sonuc_sayfa_no,

                    "dosya":
                        bilgi["dosya_adi"],

                    "eksik":
                        ", ".join(
                            eksikler
                        ),

                    "resim":
                        onizleme_hazirla(
                            (
                                kart
                                if kart is not None
                                else resim
                            )
                        )
                })


            # =============================================
            # DEBUG
            # =============================================

            if debug_aktif:

                st.divider()

                if pdf_mi:

                    st.write(
                        f"### Sayfa {sayfa_no}"
                    )

                else:

                    st.write(
                        f"### {bilgi['dosya_adi']}"
                    )


                st.write(
                    f"T.C.: {tc_no}"
                )

                st.write(
                    f"Ad: {ad}"
                )

                st.write(
                    f"Soyad: {soyad}"
                )


                if (
                    debug_modu
                    and
                    ocr_sonuc.get(
                        "debug_resmi"
                    )
                    is not None
                ):

                    st.image(
                        cv2.cvtColor(
                            ocr_sonuc[
                                "debug_resmi"
                            ],

                            cv2.COLOR_BGR2RGB
                        )
                    )


                if ham_ocr_goster:

                    with st.expander(
                        "Ham OCR"
                    ):

                        for item in (
                            ocr_sonuc.get(
                                "tum_ocr",
                                []
                            )
                        ):

                            st.write(
                                (
                                    f'{item["text"]} '
                                    f'— {item["conf"]}'
                                )
                            )


            progress.progress(
                (index + 1)
                /
                toplam
            )


        # =================================================
        # SONUÇ
        # =================================================

        durum.success(
            f"✅ {toplam} sayfa işlendi."
        )


        df = pd.DataFrame(
            tum_sonuclar,

            columns=[
                "Sayfa No",
                "T.C.",
                "Ad",
                "Soyad"
            ]
        )


        excel_data = (
            excel_olustur(
                df,
                meta
            )
        )


        st.session_state.sonuc_df = (
            df
        )

        st.session_state.excel_data = (
            excel_data
        )

        st.session_state.bulunamayanlar = (
            bulunamayanlar
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
    # EXCEL
    # =====================================================

    st.download_button(
        "📥 Excel indir",

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
    # BULUNAMAYANLAR
    # =====================================================

    bulunamayanlar = (
        st.session_state.bulunamayanlar
    )


    if bulunamayanlar:

        with st.expander(
            (
                "⚠️ Bulunamayan Kimlikler "
                f"({len(bulunamayanlar)})"
            )
        ):

            for item in bulunamayanlar:

                st.markdown("---")


                # PDF'DEN GELDİYSE
                if item["pdf_mi"]:

                    st.subheader(
                        (
                            f'📄 {item["dosya"]} '
                            f'— Sayfa {item["sayfa_no"]}'
                        )
                    )


                # NORMAL RESİM
                else:

                    st.subheader(
                        f'🖼️ {item["dosya"]}'
                    )


                st.write(
                    (
                        "**Bulunamayan alanlar:** "
                        f'{item["eksik"]}'
                    )
                )


                if item["resim"] is not None:

                    st.image(
                        item["resim"],
                        width=650
                    )