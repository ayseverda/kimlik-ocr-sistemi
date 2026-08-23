import streamlit as st
import cv2
import numpy as np
import pymupdf
import pandas as pd

from io import BytesIO

from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font

from goruntu_isleme import kart_tespit_et_ve_duzelt
from metin_ayiklama import bilgileri_cimbizla


# =========================================================
# PDF
# =========================================================

PDF_RENDER_DPI = 200


def _pixmap_to_bgr(pix):

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


def _yuklenen_dosyayi_goruntulere_ayir(
    dosya
):

    ad_lower = dosya.name.lower()

    if ad_lower.endswith(
        ".pdf"
    ):

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
                    (
                        f"{dosya.name} "
                        f"- Sayfa {sayfa_no}"
                    ),

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
    "Yeni T.C. kimlik, eski nüfus cüzdanı ve "
    "göçmen/yabancı kimlik belgelerini otomatik olarak tespit eder."
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

if "kimlik_pdf_data" not in st.session_state:
    st.session_state.kimlik_pdf_data = None

if "kimlik_pdf_sayisi" not in st.session_state:
    st.session_state.kimlik_pdf_sayisi = 0

if "kimlik_pdf_duzeltilen" not in st.session_state:
    st.session_state.kimlik_pdf_duzeltilen = 0

if "kimlik_pdf_orijinal" not in st.session_state:
    st.session_state.kimlik_pdf_orijinal = 0


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "Ayarlar"
    )

    kimlik_pdf_olustur = st.checkbox(
        "📄 Kimlikleri PDF olarak birleştir",
        value=False,
        help=(
            "Kart bulunursa düzeltilmiş hali, "
            "bulunamazsa orijinal görüntü PDF'e eklenir."
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

    accept_multiple_files=True,

    max_upload_size=150
)


# =========================================================
# ÖNİZLEME
# =========================================================

def onizleme_hazirla(
    resim
):

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
# PDF'E RESİM EKLE
# =========================================================

def resmi_pdfe_ekle(
    pdf_doc,
    resim
):

    if (
        pdf_doc is None
        or
        resim is None
    ):
        return False

    try:

        h, w = resim.shape[:2]

        if (
            h <= 0
            or
            w <= 0
        ):
            return False

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

        jpg_bytes = (
            encoded.tobytes()
        )

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

    buffer.seek(
        0
    )

    wb = load_workbook(
        buffer
    )

    ws = wb.active

    ws.title = (
        "Kimlik Sonuçları"
    )

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

    kimlik_no_col = (
        basliklar[
            "Kimlik No"
        ]
    )

    ad_col = (
        basliklar[
            "Ad"
        ]
    )

    soyad_col = (
        basliklar[
            "Soyad"
        ]
    )

    for satir_no, item in enumerate(
        meta,
        start=2
    ):

        # Kimlik No
        if not item[
            "kimlik_no_bulundu"
        ]:

            ws.cell(
                satir_no,
                kimlik_no_col
            ).fill = kirmizi

        # Ad
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

        # Soyad
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

        # =================================================
        # SÜRESİ GEÇMİŞ GÖÇMEN BELGESİ
        # TÜM SATIR SARI
        # =================================================

        if item.get(
            "belge_gecerli"
        ) is False:

            for sutun_no in range(
                1,
                ws.max_column + 1
            ):

                hucre = ws.cell(
                    satir_no,
                    sutun_no
                )

                hucre.fill = sari

                hucre.font = Font(
                    bold=True
                )

    genislikler = {
        "A": 12,
        "B": 20,
        "C": 20,
        "D": 25,
        "E": 25,
        "F": 18,
        "G": 18,
    }

    for col, width in genislikler.items():

        ws.column_dimensions[
            col
        ].width = width

    sonuc = BytesIO()

    wb.save(
        sonuc
    )

    sonuc.seek(
        0
    )

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

    kimlik_pdf_doc = None

    pdf_toplam = 0
    pdf_duzeltilen = 0
    pdf_orijinal = 0

    if kimlik_pdf_olustur:

        kimlik_pdf_doc = (
            pymupdf.open()
        )

    islenecekler = []

    for dosya in dosyalar:

        islenecekler.extend(
            _yuklenen_dosyayi_goruntulere_ayir(
                dosya
            )
        )

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
    # HER SAYFA
    # =====================================================

    for index, bilgi in enumerate(
        islenecekler
    ):

        sayfa_no = (
            bilgi[
                "sayfa_no"
            ]
        )

        pdf_mi = (
            bilgi[
                "pdf_mi"
            ]
        )

        resim = (
            bilgi[
                "resim"
            ]
        )

        durum.write(
            (
                "Kimlikler işleniyor... "
                f"{index + 1}/{toplam}"
            )
        )

        kimlik_no = (
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

        belge_tipi = (
            "bilinmiyor"
        )

        bitis_tarihi = ""

        belge_gecerli = None

        kart = None
        kart_sonuc = {}
        ocr_sonuc = {}

        # =================================================
        # KART + BELGE TİPİ
        # =================================================

        if resim is not None:

            try:

                kart_sonuc = (
                    kart_tespit_et_ve_duzelt(
                        resim,

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
                    "basarili":
                        False,

                    "mesaj":
                        str(e)
                }

            if kart_sonuc.get(
                "basarili",
                False
            ):

                kart = kart_sonuc.get(
                    "kart"
                )

                belge_tipi = kart_sonuc.get(
                    "belge_tipi",
                    "tc"
                )

        # =================================================
        # KİMLİK PDF
        # =================================================

        if kimlik_pdf_olustur:

            if kart is not None:

                pdf_resmi = kart
                duzeltilmis_mi = True

            else:

                pdf_resmi = resim
                duzeltilmis_mi = False

            if pdf_resmi is not None:

                basarili = resmi_pdfe_ekle(
                    kimlik_pdf_doc,
                    pdf_resmi
                )

                if basarili:

                    pdf_toplam += 1

                    if duzeltilmis_mi:
                        pdf_duzeltilen += 1

                    else:
                        pdf_orijinal += 1

        # =================================================
        # OCR
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
                        ),

                        belge_tipi=(
                            belge_tipi
                        )
                    )
                )

            except Exception as e:

                ocr_sonuc = {
                    "hata":
                        str(e)
                }

            sonuc_sayfa_no = (
                ocr_sonuc.get(
                    "sayfa_no",
                    sayfa_no
                )
            )

            kimlik_no = (
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

            bitis_tarihi = (
                ocr_sonuc.get(
                    "bitis_tarihi",
                    ""
                )
            )

            belge_gecerli = (
                ocr_sonuc.get(
                    "belge_gecerli"
                )
            )

        # =================================================
        # BELGE TÜRÜ ADI
        # =================================================

        if belge_tipi == "gocmen":

            belge_turu_yazi = (
                "Göçmen / Yabancı"
            )

        elif belge_tipi == "eski_tc":

            belge_turu_yazi = (
                "Eski T.C. Kimlik"
            )

        elif belge_tipi == "tc":

            belge_turu_yazi = (
                "T.C. Kimlik"
            )

        else:

            belge_turu_yazi = (
                "Bilinmiyor"
            )

        # =================================================
        # GEÇERLİLİK
        # =================================================

        if belge_tipi == "gocmen":

            if belge_gecerli is False:

                gecerlilik_yazi = (
                    "GEÇERSİZ"
                )

            elif belge_gecerli is True:

                gecerlilik_yazi = (
                    "Geçerli"
                )

            else:

                gecerlilik_yazi = (
                    "Kontrol Edilemedi"
                )

        else:

            gecerlilik_yazi = "-"

            bitis_tarihi = "-"

        # =================================================
        # TABLO
        # =================================================

        tablo_sayfa = (
            sonuc_sayfa_no
            if pdf_mi
            else "-"
        )

        tum_sonuclar.append({
            "Sayfa No":
                tablo_sayfa,

            "Belge Türü":
                belge_turu_yazi,

            "Kimlik No":
                kimlik_no,

            "Ad":
                ad,

            "Soyad":
                soyad,

            "Bitiş Tarihi":
                bitis_tarihi,

            "Geçerlilik":
                gecerlilik_yazi
        })

        # =================================================
        # META
        # =================================================

        kimlik_no_bulundu = (
            kimlik_no
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
            "kimlik_no_bulundu":
                kimlik_no_bulundu,

            "ad_bulundu":
                ad_bulundu,

            "soyad_bulundu":
                soyad_bulundu,

            "ad_conf":
                ad_conf,

            "soyad_conf":
                soyad_conf,

            "belge_gecerli":
                belge_gecerli,

            "belge_tipi":
                belge_tipi
        })

        # =================================================
        # BULUNAMAYANLAR
        # =================================================

        eksikler = [
            etiket

            for etiket, bulundu
            in [
                (
                    "Kimlik No",
                    kimlik_no_bulundu
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
        # DEBUG
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
                f"Belge tipi: **{belge_turu_yazi}**"
            )

            st.write(
                f"Kimlik No: {kimlik_no}"
            )

            st.write(
                f"Ad: {ad}"
            )

            st.write(
                f"Soyad: {soyad}"
            )

            if belge_tipi == "gocmen":

                st.write(
                    f"Bitiş tarihi: {bitis_tarihi}"
                )

                if belge_gecerli is False:

                    st.warning(
                        "⚠️ Bu belgenin geçerlilik süresi geçmiş."
                    )

                elif belge_gecerli is True:

                    st.success(
                        "✅ Belge geçerli."
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

        progress.progress(
            (index + 1)
            /
            toplam
        )

    # =====================================================
    # PDF TAMAMLA
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

                st.session_state.kimlik_pdf_data = None

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
            "Belge Türü",
            "Kimlik No",
            "Ad",
            "Soyad",
            "Bitiş Tarihi",
            "Geçerlilik"
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
    # DOWNLOAD
    # =====================================================

    if (
        st.session_state.kimlik_pdf_data
        is not None
    ):

        col1, col2 = st.columns(
            2
        )

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

        st.caption(
            (
                f"PDF: "
                f"{st.session_state.kimlik_pdf_duzeltilen} "
                f"belge düzeltilmiş, "
                f"{st.session_state.kimlik_pdf_orijinal} "
                f"belge tespit edilemediği için "
                f"orijinal haliyle eklendi."
            )
        )

    else:

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