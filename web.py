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
# DOSYA
# =========================================================

# PDF sayfalarını 200 DPI render ediyoruz.
PDF_RENDER_DPI = 200


def _pixmap_to_bgr(pix):
    """
    PyMuPDF pixmap -> OpenCV BGR görüntü.
    """

    img = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(
        pix.height,
        pix.width,
        pix.n
    )

    if pix.n == 4:
        return cv2.cvtColor(
            img,
            cv2.COLOR_RGBA2BGR
        )

    if pix.n == 3:
        return cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

    if pix.n == 1:
        return cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2BGR
        )

    return img


def _yuklenen_dosyayi_goruntulere_ayir(dosya):
    """
    Bir yüklemeyi:
        - görsel
        - PDF

    işlenebilir OpenCV görüntülerine dönüştürür.
    """

    ad_lower = dosya.name.lower()


    # =====================================================
    # PDF
    # =====================================================

    if ad_lower.endswith(".pdf"):

        sonuc = []

        veri = dosya.getvalue()

        doc = pymupdf.open(
            stream=veri,
            filetype="pdf"
        )


        for sayfa_no, sayfa in enumerate(
            doc,
            start=1
        ):

            pix = sayfa.get_pixmap(
                dpi=PDF_RENDER_DPI
            )


            sonuc.append({
                "dosya_adi":
                    dosya.name,

                "gorunen_isim":
                    f"{dosya.name} - Sayfa {sayfa_no}",

                "sayfa_no":
                    sayfa_no,

                "pdf_mi":
                    True,

                "resim":
                    _pixmap_to_bgr(
                        pix
                    )
            })


        doc.close()

        return sonuc


    # =====================================================
    # NORMAL GÖRSEL
    # =====================================================

    veri = np.frombuffer(
        dosya.getvalue(),
        dtype=np.uint8
    )


    resim = cv2.imdecode(
        veri,
        cv2.IMREAD_COLOR
    )


    return [{
        "dosya_adi":
            dosya.name,

        "gorunen_isim":
            dosya.name,

        "sayfa_no":
            None,

        "pdf_mi":
            False,

        "resim":
            resim
    }]


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
    "Kimlik kartlarından T.C. Kimlik No, "
    "Ad ve Soyad bilgilerini çıkarır."
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
# KİMLİK PDF SESSION
# =========================================================

if "kimlik_pdf_data" not in st.session_state:
    st.session_state.kimlik_pdf_data = None


if "kimlik_pdf_sayisi" not in st.session_state:
    st.session_state.kimlik_pdf_sayisi = 0


if "kimlik_pdf_duzeltilen" not in st.session_state:
    st.session_state.kimlik_pdf_duzeltilen = 0


if "kimlik_pdf_orijinal" not in st.session_state:
    st.session_state.kimlik_pdf_orijinal = 0


# =========================================================
# REFERANS
# =========================================================

REFERANS_YOLU = "referans_kimlik.jpg"


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


    # =====================================================
    # YENİ:
    # KİMLİKLERİ PDF YAP
    # =====================================================

    kimlik_pdf_olustur = st.checkbox(
        "📄 Kimlikleri PDF olarak birleştir",
        value=False,
        help=(
            "Kart tespit edilirse perspektifi düzeltilmiş "
            "hali PDF'e eklenir. Kart tespit edilemezse "
            "orijinal görüntü/sayfa PDF'e eklenir. "
            "Böylece hiçbir kimlik kaybolmaz."
        )
    )


    st.divider()


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

    accept_multiple_files=True,

    max_upload_size=150
)


# =========================================================
# ÖNİZLEME
# =========================================================

def onizleme_hazirla(resim):

    if resim is None:
        return None


    h, w = resim.shape[:2]


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


    basarili, encoded = cv2.imencode(
        ".jpg",
        resim,

        [
            cv2.IMWRITE_JPEG_QUALITY,
            82
        ]
    )


    if not basarili:
        return None


    return encoded.tobytes()


# =========================================================
# PDF'E GÖRÜNTÜ EKLE
# =========================================================

def resmi_pdfe_ekle(
    pdf_doc,
    resim
):
    """
    OpenCV görüntüsünü PDF'e tek sayfa olarak ekler.

    Buraya:
        - bulunan/düzeltilmiş kart
        VEYA
        - bulunamayan kartın orijinal görüntüsü

    gönderilebilir.
    """

    if (
        pdf_doc is None
        or
        resim is None
    ):
        return False


    try:

        h, w = resim.shape[:2]

        if h <= 0 or w <= 0:
            return False


        # -------------------------------------------------
        # PDF'e eklemek için JPEG oluştur
        # -------------------------------------------------

        basarili, encoded = cv2.imencode(
            ".jpg",
            resim,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                92
            ]
        )


        if not basarili:
            return False


        jpg_bytes = encoded.tobytes()


        # -------------------------------------------------
        # Sayfanın oranını görüntüyle aynı yap
        # -------------------------------------------------

        pdf_genislik = 720.0

        pdf_yukseklik = (
            pdf_genislik
            *
            h
            /
            w
        )


        sayfa = pdf_doc.new_page(
            width=pdf_genislik,
            height=pdf_yukseklik
        )


        rect = pymupdf.Rect(
            0,
            0,
            pdf_genislik,
            pdf_yukseklik
        )


        sayfa.insert_image(
            rect,
            stream=jpg_bytes,
            keep_proportion=True
        )


        return True


    except Exception:

        return False


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


    # =====================================================
    # BAŞLIKLAR
    # =====================================================

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


    # =====================================================
    # GÜVEN RENKLENDİRMESİ
    # =====================================================

    for satir_no, item in enumerate(
        meta,
        start=2
    ):

        # TC
        if not item[
            "tc_bulundu"
        ]:

            ws.cell(
                satir_no,
                tc_col
            ).fill = kirmizi


        # AD
        if not item[
            "ad_bulundu"
        ]:

            ws.cell(
                satir_no,
                ad_col
            ).fill = kirmizi


        elif item[
            "ad_conf"
        ] < 0.50:

            ws.cell(
                satir_no,
                ad_col
            ).fill = kirmizi


        elif item[
            "ad_conf"
        ] < 0.70:

            ws.cell(
                satir_no,
                ad_col
            ).fill = sari


        # SOYAD
        if not item[
            "soyad_bulundu"
        ]:

            ws.cell(
                satir_no,
                soyad_col
            ).fill = kirmizi


        elif item[
            "soyad_conf"
        ] < 0.50:

            ws.cell(
                satir_no,
                soyad_col
            ).fill = kirmizi


        elif item[
            "soyad_conf"
        ] < 0.70:

            ws.cell(
                satir_no,
                soyad_col
            ).fill = sari


    # =====================================================
    # SÜTUN GENİŞLİKLERİ
    # =====================================================

    for col, width in zip(
        "ABCD",
        (
            12,
            18,
            25,
            25
        )
    ):

        ws.column_dimensions[
            col
        ].width = width


    sonuc = BytesIO()


    wb.save(
        sonuc
    )


    sonuc.seek(0)


    return sonuc.getvalue()


# =========================================================
# İŞLEM
# =========================================================

if (
    dosyalar
    and
    st.button(
        "🚀 İşlemi Başlat",
        type="primary"
    )
):

    # =====================================================
    # ESKİ SONUÇLARI TEMİZLE
    # =====================================================

    st.session_state.sonuc_df = None

    st.session_state.excel_data = None

    st.session_state.bulunamayanlar = []

    st.session_state.kimlik_pdf_data = None

    st.session_state.kimlik_pdf_sayisi = 0

    st.session_state.kimlik_pdf_duzeltilen = 0

    st.session_state.kimlik_pdf_orijinal = 0


    durum = st.empty()


    durum.write(
        "Dosyalar hazırlanıyor..."
    )


    # =====================================================
    # PDF BELGESİNİ HAZIRLA
    # =====================================================

    kimlik_pdf_doc = None

    pdf_toplam = 0

    pdf_duzeltilen = 0

    pdf_orijinal = 0


    if kimlik_pdf_olustur:

        kimlik_pdf_doc = (
            pymupdf.open()
        )


    # =====================================================
    # DOSYALARI HAZIRLA
    # =====================================================

    islenecekler = []


    for dosya in dosyalar:

        islenecekler.extend(
            _yuklenen_dosyayi_goruntulere_ayir(
                dosya
            )
        )


    # =====================================================
    # SONUÇ LİSTELERİ
    # =====================================================

    tum_sonuclar = []

    meta = []

    bulunamayanlar = []


    toplam = len(
        islenecekler
    )


    if toplam == 0:

        durum.error(
            "İşlenecek dosya bulunamadı."
        )

        st.stop()


    progress = st.progress(
        0
    )


    # =====================================================
    # HER SAYFA / GÖRSEL
    # =====================================================

    for index, bilgi in enumerate(
        islenecekler
    ):

        sayfa_no = bilgi[
            "sayfa_no"
        ]


        pdf_mi = bilgi[
            "pdf_mi"
        ]


        durum.write(
            (
                "Kimlikler işleniyor... "
                f"{index + 1}/{toplam}"
            )
        )


        resim = bilgi[
            "resim"
        ]


        tc_no = (
            "Bulunamadi"
        )

        ad = (
            "Bulunamadi"
        )

        soyad = (
            "Bulunamadi"
        )


        ad_conf = 0.0

        soyad_conf = 0.0


        kart = None

        kart_sonuc = {}

        ocr_sonuc = {}


        # =================================================
        # 1. KART TESPİT
        # =================================================

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
                    "basarili":
                        False
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


        # =================================================
        # 2. KİMLİK PDF'E EKLE
        #
        # KART BULUNURSA:
        #     düzeltmiş kart
        #
        # KART BULUNAMAZSA:
        #     orijinal sayfa/görüntü
        #
        # Böylece hiçbir yükleme kaybolmaz.
        # =================================================

        if kimlik_pdf_olustur:

            if kart is not None:

                pdf_resmi = kart

                duzeltilmis_mi = True

            else:

                pdf_resmi = resim

                duzeltilmis_mi = False


            if pdf_resmi is not None:

                basarili = (
                    resmi_pdfe_ekle(
                        kimlik_pdf_doc,
                        pdf_resmi
                    )
                )


                if basarili:

                    pdf_toplam += 1


                    if duzeltilmis_mi:

                        pdf_duzeltilen += 1

                    else:

                        pdf_orijinal += 1


        # =================================================
        # 3. OCR
        # =================================================

        sonuc_sayfa_no = (
            sayfa_no
        )


        if kart is not None:

            try:

                ocr_sonuc = (
                    bilgileri_cimbizla(
                        kart,

                        sayfa_no=(
                            sayfa_no
                        ),

                        debug=(
                            debug_modu
                            or
                            ham_ocr_goster
                        )
                    )
                )


            except Exception:

                ocr_sonuc = {}


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


        # =================================================
        # 4. TABLO
        # =================================================

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


        # =================================================
        # 5. META
        # =================================================

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


        # =================================================
        # 6. BULUNAMAYANLAR
        # =================================================

        eksikler = [
            etiket

            for etiket, bulundu
            in [
                (
                    "T.C.",
                    tc_bulundu
                ),

                (
                    "Ad",
                    ad_bulundu
                ),

                (
                    "Soyad",
                    soyad_bulundu
                )
            ]

            if not bulundu
        ]


        if eksikler:

            bulunamayanlar.append({
                "pdf_mi":
                    pdf_mi,

                "sayfa_no":
                    sonuc_sayfa_no,

                "dosya":
                    bilgi[
                        "dosya_adi"
                    ],

                "eksik":
                    ", ".join(
                        eksikler
                    ),

                "resim":
                    onizleme_hazirla(
                        kart
                        if kart is not None
                        else resim
                    )
            })


        # =================================================
        # 7. DEBUG
        # =================================================

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


            # OCR kutuları
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


            # Kart sınırı
            if (
                kart_siniri_goster
                and
                kart_sonuc.get(
                    "debug_resmi"
                )
                is not None
            ):

                st.image(
                    cv2.cvtColor(
                        kart_sonuc[
                            "debug_resmi"
                        ],
                        cv2.COLOR_BGR2RGB
                    )
                )


            # SIFT eşleşmeleri
            if (
                feature_debug
                and
                kart_sonuc.get(
                    "match_debug"
                )
                is not None
            ):

                st.image(
                    cv2.cvtColor(
                        kart_sonuc[
                            "match_debug"
                        ],
                        cv2.COLOR_BGR2RGB
                    )
                )


            # Ham OCR
            if ham_ocr_goster:

                with st.expander(
                    "Ham OCR"
                ):

                    ham_sonuclar = (
                        ocr_sonuc.get(
                            "tum_ocr",
                            []
                        )
                    )


                    if not ham_sonuclar:

                        st.info(
                            "OCR sonucu yok."
                        )


                    for item in ham_sonuclar:

                        st.write(
                            (
                                f'{item["text"]} '
                                f'— {item["conf"]}'
                            )
                        )


        # =================================================
        # 8. PROGRESS
        # =================================================

        progress.progress(
            (index + 1)
            /
            toplam
        )


    # =====================================================
    # KİMLİK PDF'İ TAMAMLA
    # =====================================================

    if kimlik_pdf_doc is not None:

        if pdf_toplam > 0:

            try:

                pdf_bytes = (
                    kimlik_pdf_doc.tobytes(
                        garbage=3,
                        deflate=True
                    )
                )


                st.session_state.kimlik_pdf_data = (
                    pdf_bytes
                )


                st.session_state.kimlik_pdf_sayisi = (
                    pdf_toplam
                )


                st.session_state.kimlik_pdf_duzeltilen = (
                    pdf_duzeltilen
                )


                st.session_state.kimlik_pdf_orijinal = (
                    pdf_orijinal
                )


            except Exception:

                st.session_state.kimlik_pdf_data = (
                    None
                )


        kimlik_pdf_doc.close()


    # =====================================================
    # SONUÇ
    # =====================================================

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


    st.session_state.sonuc_df = (
        df
    )


    st.session_state.excel_data = (
        excel_olustur(
            df,
            meta
        )
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
    # İNDİRME BUTONLARI
    # =====================================================

    if (
        st.session_state.kimlik_pdf_data
        is not None
    ):

        col1, col2 = st.columns(
            2
        )


        # -------------------------------------------------
        # Excel
        # -------------------------------------------------

        with col1:

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
                ),

                use_container_width=True
            )


        # -------------------------------------------------
        # Kimlik PDF
        # -------------------------------------------------

        with col2:

            st.download_button(
                (
                    "📄 Kimlik PDF'ini indir "
                    f"({st.session_state.kimlik_pdf_sayisi})"
                ),

                data=(
                    st.session_state.kimlik_pdf_data
                ),

                file_name=(
                    "kimlikler.pdf"
                ),

                mime="application/pdf",

                use_container_width=True
            )


        # -------------------------------------------------
        # PDF özeti
        # -------------------------------------------------

        st.caption(
            (
                f"PDF: "
                f"{st.session_state.kimlik_pdf_duzeltilen} "
                f"kimlik düzeltilmiş, "
                f"{st.session_state.kimlik_pdf_orijinal} "
                f"kimlik/sayfa tespit edilemediği için "
                f"orijinal haliyle eklendi."
            )
        )


    else:

        # Yalnızca Excel
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
    # BULUNAMAYAN KİMLİKLER
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

                st.markdown(
                    "---"
                )


                if item[
                    "pdf_mi"
                ]:

                    st.subheader(
                        (
                            f'📄 {item["dosya"]} '
                            f'— Sayfa '
                            f'{item["sayfa_no"]}'
                        )
                    )


                else:

                    st.subheader(
                        (
                            f'🖼️ '
                            f'{item["dosya"]}'
                        )
                    )


                st.write(
                    (
                        "**Bulunamayan alanlar:** "
                        f'{item["eksik"]}'
                    )
                )


                if item[
                    "resim"
                ] is not None:

                    st.image(
                        item["resim"],
                        width=650
                    )