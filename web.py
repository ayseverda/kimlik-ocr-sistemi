import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import pymupdf
import pandas as pd

from goruntu_isleme import kart_tespit_et_ve_duzelt
from metin_ayiklama import bilgileri_cimbizla


# =========================================================
# SAYFA AYARLARI
# =========================================================

st.set_page_config(
    page_title="Kimlik Okuyucu",
    page_icon="🪪",
    layout="wide"
)

st.title("🪪 Kimlik Kartı Okuyucu")

st.caption(
    "Kart tespiti: SIFT / Homography  •  Metin okuma: PaddleOCR"
)


# =========================================================
# REFERANS KİMLİK
# =========================================================

REFERANS_YOLU = "referans_kimlik.jpg"

if not os.path.exists(REFERANS_YOLU):

    st.error(
        "❌ referans_kimlik.jpg bulunamadı."
    )

    st.info(
        "referans_kimlik.jpg dosyasını web.py ile aynı klasöre koy."
    )

    st.stop()

else:

    st.success(
        "✅ Referans kimlik bulundu."
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Debug Ayarları")

    debug_modu = st.checkbox(
        "Debug görüntülerini göster",
        value=True
    )

    kart_siniri_goster = st.checkbox(
        "Bulunan kart sınırını göster",
        value=True
    )

    paddle_debug_goster = st.checkbox(
        "PaddleOCR kutularını göster",
        value=True
    )

    label_debug_goster = st.checkbox(
        "Ad / Soyad label debug göster",
        value=True
    )

    ham_ocr_goster = st.checkbox(
        "PaddleOCR'ın gördüğü metinleri göster",
        value=True
    )

    feature_debug = st.checkbox(
        "SIFT feature eşleşmelerini göster",
        value=False
    )


# =========================================================
# DOSYA YÜKLEME
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
# DOSYALAR VARSA
# =========================================================

if dosyalar:

    if st.button(
        "🚀 İşlemi Başlat",
        type="primary"
    ):

        gecici_klasor = tempfile.mkdtemp()

        islenecekler = []

        tum_sonuclar = []


        # =================================================
        # DOSYALARI HAZIRLA
        # =================================================

        with st.spinner(
            "Dosyalar hazırlanıyor..."
        ):

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


                # =========================================
                # PDF
                # =========================================

                if dosya.name.lower().endswith(
                    ".pdf"
                ):

                    try:

                        doc = pymupdf.open(
                            dosya_yolu
                        )

                        for sayfa_no, sayfa in enumerate(
                            doc
                        ):

                            # OCR için yüksek kalite
                            pix = sayfa.get_pixmap(
                                dpi=300
                            )

                            sayfa_yolu = os.path.join(
                                gecici_klasor,
                                (
                                    f"{os.path.splitext(dosya.name)[0]}"
                                    f"_sayfa_{sayfa_no + 1}.png"
                                )
                            )

                            pix.save(
                                sayfa_yolu
                            )

                            islenecekler.append(
                                {
                                    "isim": (
                                        f"{dosya.name} "
                                        f"- Sayfa {sayfa_no + 1}"
                                    ),
                                    "yol": sayfa_yolu
                                }
                            )

                        doc.close()

                    except Exception as e:

                        st.error(
                            f"{dosya.name} PDF dosyası açılamadı: {e}"
                        )


                # =========================================
                # FOTOĞRAF
                # =========================================

                else:

                    islenecekler.append(
                        {
                            "isim": dosya.name,
                            "yol": dosya_yolu
                        }
                    )


        # =================================================
        # DOSYA YOKSA
        # =================================================

        if not islenecekler:

            st.warning(
                "İşlenecek görüntü bulunamadı."
            )

            st.stop()


        # =================================================
        # PROGRESS
        # =================================================

        progress = st.progress(0)

        toplam = len(
            islenecekler
        )


        # =================================================
        # HER GÖRSELİ İŞLE
        # =================================================

        for index, bilgi in enumerate(
            islenecekler
        ):

            dosya_adi = bilgi["isim"]

            yol = bilgi["yol"]

            st.divider()

            st.header(
                f"📄 {dosya_adi}"
            )


            # =================================================
            # GÖRSELİ OKU
            # =================================================

            try:

                veri = np.fromfile(
                    yol,
                    dtype=np.uint8
                )

                resim = cv2.imdecode(
                    veri,
                    cv2.IMREAD_COLOR
                )

            except Exception as e:

                resim = None

                st.error(
                    f"Görüntü okunurken hata oluştu: {e}"
                )


            # =================================================
            # DEFAULT SONUÇ
            # =================================================

            satir = {

                "Dosya": dosya_adi,

                "T.C.": "Bulunamadi",

                "Ad": "Bulunamadi",

                "Soyad": "Bulunamadi",

                "Güven": "dusuk"
            }


            # =================================================
            # GÖRÜNTÜ OKUNAMADI
            # =================================================

            if resim is None:

                st.error(
                    "❌ Görüntü okunamadı."
                )

                tum_sonuclar.append(
                    satir
                )

                progress.progress(
                    (index + 1) / toplam
                )

                continue


            # =================================================
            # 1. KART TESPİTİ
            # =================================================

            with st.spinner(
                "Kimlik kartı aranıyor..."
            ):

                try:

                    kart_sonuc = kart_tespit_et_ve_duzelt(
                        resim,
                        REFERANS_YOLU
                    )

                except Exception as e:

                    st.error(
                        f"Kart tespitinde hata oluştu: {e}"
                    )

                    tum_sonuclar.append(
                        satir
                    )

                    progress.progress(
                        (index + 1) / toplam
                    )

                    continue


            # =================================================
            # FEATURE İSTATİSTİKLERİ
            # =================================================

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Feature sistemi",
                kart_sonuc.get(
                    "detector",
                    "-"
                )
                or "-"
            )

            c2.metric(
                "İyi eşleşme",
                kart_sonuc.get(
                    "iyi_eslesme",
                    0
                )
            )

            c3.metric(
                "RANSAC Inlier",
                kart_sonuc.get(
                    "inlier",
                    0
                )
            )


            # =================================================
            # KART BULUNAMADI
            # =================================================

            if not kart_sonuc.get(
                "basarili",
                False
            ):

                st.error(
                    "❌ Kimlik kartı tespit edilemedi."
                )

                st.warning(
                    kart_sonuc.get(
                        "mesaj",
                        "Bilinmeyen kart tespit hatası."
                    )
                )

                st.write(
                    "### Orijinal görüntü"
                )

                st.image(
                    cv2.cvtColor(
                        resim,
                        cv2.COLOR_BGR2RGB
                    ),
                    use_container_width=True
                )


                # ---------------------------------------------
                # FEATURE DEBUG
                # ---------------------------------------------

                match_debug = kart_sonuc.get(
                    "match_debug"
                )

                if (
                    feature_debug
                    and
                    match_debug is not None
                ):

                    with st.expander(
                        "🧩 SIFT feature eşleşmeleri",
                        expanded=False
                    ):

                        st.image(
                            cv2.cvtColor(
                                match_debug,
                                cv2.COLOR_BGR2RGB
                            ),
                            use_container_width=True
                        )


                tum_sonuclar.append(
                    satir
                )

                progress.progress(
                    (index + 1) / toplam
                )

                continue


            # =================================================
            # KART BULUNDU
            # =================================================

            st.success(
                "✅ Kimlik kartı tespit edildi."
            )

            kart = kart_sonuc.get(
                "kart"
            )

            if kart is None:

                st.error(
                    "Kart bulundu ancak perspektif sonucu alınamadı."
                )

                tum_sonuclar.append(
                    satir
                )

                progress.progress(
                    (index + 1) / toplam
                )

                continue


            # =================================================
            # KART DEBUG
            # =================================================

            if debug_modu:

                col1, col2 = st.columns(2)

                # ---------------------------------------------
                # ORİJİNAL
                # ---------------------------------------------

                with col1:

                    st.write(
                        "### 1. Orijinal görüntü"
                    )

                    st.image(
                        cv2.cvtColor(
                            resim,
                            cv2.COLOR_BGR2RGB
                        ),
                        use_container_width=True
                    )


                # ---------------------------------------------
                # KART SINIRI
                # ---------------------------------------------

                with col2:

                    st.write(
                        "### 2. Bulunan kimlik sınırı"
                    )

                    kart_debug = kart_sonuc.get(
                        "debug_resmi"
                    )

                    if (
                        kart_siniri_goster
                        and
                        kart_debug is not None
                    ):

                        st.image(
                            cv2.cvtColor(
                                kart_debug,
                                cv2.COLOR_BGR2RGB
                            ),
                            use_container_width=True
                        )

                    else:

                        st.info(
                            "Kart sınırı debug kapalı."
                        )


            # =================================================
            # PERSPEKTİF SONUCU
            # =================================================

            st.write(
                "## ✅ Perspektifi düzeltilmiş kart"
            )

            st.image(
                cv2.cvtColor(
                    kart,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )


            # =================================================
            # SIFT DEBUG
            # =================================================

            match_debug = kart_sonuc.get(
                "match_debug"
            )

            if (
                feature_debug
                and
                match_debug is not None
            ):

                with st.expander(
                    "🧩 SIFT feature eşleşmelerini göster",
                    expanded=False
                ):

                    st.image(
                        cv2.cvtColor(
                            match_debug,
                            cv2.COLOR_BGR2RGB
                        ),
                        use_container_width=True
                    )


            # =================================================
            # 2. OCR
            # =================================================

            st.write(
                "## 🔎 PaddleOCR"
            )

            with st.spinner(
                "Kart üzerindeki yazılar aranıyor..."
            ):

                try:

                    ocr_sonuc = bilgileri_cimbizla(
                        kart
                    )

                except Exception as e:

                    st.error(
                        f"OCR sırasında hata oluştu: {e}"
                    )

                    tum_sonuclar.append(
                        satir
                    )

                    progress.progress(
                        (index + 1) / toplam
                    )

                    continue


            # =================================================
            # OCR SONUCU
            # =================================================

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

            guven = ocr_sonuc.get(
                "guven",
                "dusuk"
            )


            satir["T.C."] = tc_no
            satir["Ad"] = ad
            satir["Soyad"] = soyad
            satir["Güven"] = guven

            tum_sonuclar.append(
                satir
            )


            # =================================================
            # SONUÇ METRİKLERİ
            # =================================================

            st.write(
                "### Çıkarılan bilgiler"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "T.C.",
                tc_no
            )

            c2.metric(
                "Ad",
                ad
            )

            c3.metric(
                "Soyad",
                soyad
            )

            c4.metric(
                "Güven",
                guven
            )


            # =================================================
            # PADDLE TEXT DETECTION DEBUG
            # =================================================

            paddle_debug = ocr_sonuc.get(
                "paddle_debug"
            )

            if (
                debug_modu
                and
                paddle_debug_goster
                and
                paddle_debug is not None
            ):

                st.write(
                    "## 🎯 PaddleOCR nereleri metin olarak görüyor?"
                )

                st.image(
                    cv2.cvtColor(
                        paddle_debug,
                        cv2.COLOR_BGR2RGB
                    ),
                    use_container_width=True
                )

                st.caption(
                    "Yeşil kutular PaddleOCR'ın metin olarak algıladığı "
                    "alanlardır. Kırmızı numaralar aşağıdaki OCR listesiyle "
                    "eşleşir."
                )


            # =================================================
            # AD / SOYAD LABEL DEBUG
            # =================================================

            label_debug = ocr_sonuc.get(
                "debug_resmi"
            )

            if (
                debug_modu
                and
                label_debug_goster
                and
                label_debug is not None
            ):

                st.write(
                    "## 🧭 Ad / Soyad label eşleştirmesi"
                )

                st.image(
                    cv2.cvtColor(
                        label_debug,
                        cv2.COLOR_BGR2RGB
                    ),
                    use_container_width=True
                )

                st.caption(
                    "Yeşil kutular OCR alanlarıdır. "
                    "Sarı çerçeve SOYADI / SURNAME etiketini, "
                    "turkuaz çerçeve ADI / GIVEN NAME(S) etiketini gösterir."
                )


            # =================================================
            # HAM OCR
            # =================================================

            if ham_ocr_goster:

                with st.expander(
                    "📝 PaddleOCR'ın gördüğü bütün metinleri göster",
                    expanded=True
                ):

                    tum_ocr = ocr_sonuc.get(
                        "tum_ocr",
                        []
                    )

                    if tum_ocr:

                        st.success(
                            f"✅ PaddleOCR toplam {len(tum_ocr)} metin alanı buldu."
                        )

                        for item in tum_ocr:

                            no = item.get(
                                "no",
                                "?"
                            )

                            metin = item.get(
                                "metin",
                                ""
                            )

                            guven_degeri = item.get(
                                "guven",
                                0
                            )

                            x = item.get(
                                "x",
                                0
                            )

                            y = item.get(
                                "y",
                                0
                            )

                            x2 = item.get(
                                "x2",
                                0
                            )

                            y2 = item.get(
                                "y2",
                                0
                            )

                            st.markdown(
                                f"**{no} — `{metin}`**"
                            )

                            st.caption(
                                (
                                    f"Güven: {guven_degeri} "
                                    f"| Kutu: "
                                    f"({x}, {y}) → ({x2}, {y2})"
                                )
                            )

                    else:

                        st.warning(
                            "⚠️ PaddleOCR kart üzerinde hiçbir metin "
                            "tespit edemedi."
                        )


            # =================================================
            # PROGRESS
            # =================================================

            progress.progress(
                (index + 1) / toplam
            )


        # =====================================================
        # PROGRESS BİTTİ
        # =====================================================

        progress.empty()


        # =====================================================
        # TÜM SONUÇLAR
        # =====================================================

        if tum_sonuclar:

            st.divider()

            st.header(
                "📋 Tüm Kimlik Sonuçları"
            )

            df = pd.DataFrame(
                tum_sonuclar
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # EXCEL
            # =================================================

            excel_yolu = os.path.join(
                gecici_klasor,
                "kimlik_sonuclari.xlsx"
            )

            try:

                df.to_excel(
                    excel_yolu,
                    index=False
                )

                with open(
                    excel_yolu,
                    "rb"
                ) as f:

                    excel_data = f.read()


                st.download_button(
                    label="📥 Sonuçları Excel olarak indir",
                    data=excel_data,
                    file_name="kimlik_sonuclari.xlsx",
                    mime=(
                        "application/"
                        "vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    )
                )

            except Exception as e:

                st.warning(
                    f"Excel dosyası oluşturulamadı: {e}"
                )