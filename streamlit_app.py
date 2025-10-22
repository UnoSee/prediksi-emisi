# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import lightgbm as lgb
import numpy as np
import os

st.set_page_config(page_title="🌍 Global Energy Dashboard", layout="wide")

# -----------------------------------------------------------
# 🔹 Sidebar Navigasi
st.sidebar.title("📊 Navigasi Dashboard")
menu = st.sidebar.radio("Pilih Halaman:", ["🌍 Analisis Data Global", "🤖 Prediksi Intensitas Karbon"])

# -----------------------------------------------------------
# 🔹 Fungsi Load Data
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

# ===========================================================
# 🔹 Upload Dataset atau Gunakan Default
# ===========================================================
st.sidebar.markdown("### 📁 Dataset Energi Global")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV Anda:", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success("✅ Dataset berhasil diunggah!")
else:
    default_path = "final_enriched_WITH_CLEAN_COORDS.csv"
    if os.path.exists(default_path):
        df = load_data(default_path)
        st.sidebar.info("📂 Menggunakan dataset bawaan.")
    else:
        st.sidebar.error("⚠️ Tidak ada dataset ditemukan. Harap unggah file CSV.")
        df = pd.DataFrame()

# ===========================================================
# 🌍 HALAMAN 1: ANALISIS DATA GLOBAL
# ===========================================================
if menu == "🌍 Analisis Data Global":
    st.title("🌍 Dashboard Analisis Energi & Intensitas Karbon Global")
    st.markdown(
        "Visualisasi interaktif untuk melihat hubungan antara energi terbarukan, intensitas karbon, "
        "dan total pembangkit listrik di seluruh dunia."
    )

    if df.empty:
        st.warning("Tidak ada data untuk ditampilkan. Silakan unggah file CSV terlebih dahulu.")
        st.stop()

    # 1️⃣ Filter Tahun
    if "year" in df.columns:
        years = sorted(df["year"].dropna().unique())
        selected_year = st.slider("Pilih Tahun", int(min(years)), int(max(years)), int(max(years)))
        df_year = df[df["year"] == selected_year]
    else:
        st.error("Kolom 'year' tidak ditemukan di dataset.")
        st.stop()

    # 2️⃣ Peta Global
    st.subheader("📍 Peta Kepadatan Pembangkit Listrik Global")
    if {"Latitude", "Longitude"}.issubset(df.columns):
        df_map = df.dropna(subset=["Latitude", "Longitude"])
        if not df_map.empty:
            fig_map = px.density_mapbox(
                df_map,
                lat="Latitude",
                lon="Longitude",
                z="total_twh" if "total_twh" in df.columns else None,
                radius=10,
                center=dict(lat=20, lon=0),
                zoom=1,
                mapbox_style="carto-darkmatter",
                title="Kepadatan Total Pembangkit Listrik Global (TWh)",
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Tidak ada koordinat yang valid untuk dipetakan.")
    else:
        st.warning("Kolom Latitude/Longitude tidak ditemukan di dataset ini.")

    # 3️⃣ Scatter Plot
    if {"renewable_share_percentage", "carbon_intensity", "entity"}.issubset(df.columns):
        st.subheader("⚡ Hubungan Energi Terbarukan vs Intensitas Karbon")
        fig_scatter = px.scatter(
            df_year,
            x="renewable_share_percentage",
            y="carbon_intensity",
            color="entity",
            size="total_twh" if "total_twh" in df.columns else None,
            hover_name="entity",
            labels={
                "renewable_share_percentage": "Share Energi Terbarukan (%)",
                "carbon_intensity": "Intensitas Karbon (gCO₂/kWh)",
            },
            title=f"Tahun {selected_year}: Hubungan Energi Terbarukan vs Intensitas Karbon",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Kolom penting untuk scatter plot tidak lengkap di dataset ini.")

    # 4️⃣ Tren Global
    st.subheader("📈 Tren Intensitas Karbon Global (Rata-rata)")
    if {"carbon_intensity", "renewable_share_percentage"}.issubset(df.columns):
        global_trend = df.groupby("year", as_index=False).agg(
            {"carbon_intensity": "mean", "renewable_share_percentage": "mean"}
        )

        fig_line = px.line(
            global_trend,
            x="year",
            y=["carbon_intensity", "renewable_share_percentage"],
            labels={"value": "Nilai", "year": "Tahun"},
            title="Tren Global: Intensitas Karbon vs Share Energi Terbarukan",
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("Tidak ada data tren yang cukup untuk divisualisasikan.")

    # 5️⃣ Statistik Ringkasan
    if not df_year.empty:
        st.subheader("📊 Statistik Global (Ringkasan Tahun Terpilih)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rata-rata Intensitas Karbon", f"{df_year['carbon_intensity'].mean():.2f} gCO₂/kWh")
        col2.metric("Rata-rata Energi Terbarukan", f"{df_year['renewable_share_percentage'].mean():.2f} %")
        col3.metric("Total Pembangkit", f"{df_year['total_twh'].sum():,.0f} TWh")
    else:
        st.info("Tidak ada data untuk tahun yang dipilih.")

    st.markdown("---")
    st.caption("© 2025 - Global Energy Dashboard | Dibuat dengan ❤️ menggunakan Streamlit & Plotly")


# ===========================================================
# 🤖 HALAMAN 2: PREDIKSI INTENSITAS KARBON
# ===========================================================
elif menu == "🤖 Prediksi Intensitas Karbon":
    st.title("🤖 Prediksi Intensitas Karbon Menggunakan Model Machine Learning (LightGBM)")

    st.markdown(
        "Gunakan model prediktif untuk memperkirakan **intensitas karbon (gCO₂/kWh)** "
        "berdasarkan 10 fitur input utama."
    )

    st.info("Pastikan file model `model_lightgbm.txt` ada di folder yang sama dengan streamlit_app.py")

    # Coba muat model LightGBM dari file .txt
    model_loaded = False
    try:
        model = lgb.Booster(model_file="model_lightgbm.txt")
        model_loaded = True
        st.success("✅ Model LightGBM berhasil dimuat dari model_lightgbm.txt")
        feature_names = model.feature_name()
    except Exception as e:
        st.warning(f"⚠️ Model belum dapat dimuat: {e}")
        feature_names = []

    # Input fitur utama (10 fitur)
    st.subheader("🧮 Masukkan Fitur untuk Prediksi")
    col1, col2 = st.columns(2)

    total_twh = col1.number_input("Total Listrik (TWh)", min_value=0.0, value=1000.0)
    renewable_share = col2.slider("Share Energi Terbarukan (%)", 0.0, 100.0, 30.0)
    coal_generation = col1.number_input("Coal Generation (TWh)", min_value=0.0, value=0.0)
    gas_generation = col2.number_input("Gas Generation (TWh)", min_value=0.0, value=0.0)
    hydro_generation = col1.number_input("Hydro Generation (TWh)", min_value=0.0, value=0.0)
    nuclear_generation = col2.number_input("Nuclear Generation (TWh)", min_value=0.0, value=0.0)
    oil_generation = col1.number_input("Oil Generation (TWh)", min_value=0.0, value=0.0)
    other_renewable = col2.number_input("Other Renewable (TWh)", min_value=0.0, value=0.0)
    biofuel_generation = col1.number_input("Biofuel Generation (TWh)", min_value=0.0, value=0.0)
    entity_input = st.selectbox("Wilayah/Negara", sorted(df["entity"].dropna().unique()))

    # Pilih tahun prediksi bebas (masa depan)
    selected_year = st.number_input(
        "Tahun Prediksi",
        min_value=2022,
        max_value=2100,
        value=2025,
        step=1
    )

    # Tombol prediksi
    if st.button("🔍 Prediksi Intensitas Karbon"):
        if model_loaded and feature_names:
            # Siapkan DataFrame sesuai fitur model
            X_new_full = pd.DataFrame(columns=feature_names)
            X_new_full.loc[0] = 0  # default semua 0

            # Map 10 input user ke fitur model
            feature_map = {
                "total_twh": total_twh,
                "renewable_share_percentage": renewable_share,
                "coal_generation_twh": coal_generation,
                "gas_generation_twh": gas_generation,
                "hydro_generation_twh": hydro_generation,
                "nuclear_generation_twh": nuclear_generation,
                "oil_generation_twh": oil_generation,
                "other_renewable_twh": other_renewable,
                "biofuel_generation_twh": biofuel_generation,
                "entity": hash(entity_input) % 1000,
                "year": selected_year
            }

            # Isi DataFrame sesuai fitur yang ada di model
            for f in feature_map:
                if f in X_new_full.columns:
                    X_new_full.loc[0, f] = feature_map[f]

            # Prediksi
            try:
                y_pred = model.predict(X_new_full)[0]
                st.success(f"🌱 Prediksi Intensitas Karbon Tahun {selected_year}: **{y_pred:.2f} gCO₂/kWh**")
            except Exception as e:
                st.error(f"Gagal melakukan prediksi: {e}")
        else:
            st.warning("⚠️ Model belum dimuat, gunakan mode simulasi.")
            simulated = 500 - (renewable_share * 3.5) + (total_twh / 2000)
            st.info(f"🌱 (Simulasi) Prediksi Intensitas Karbon Tahun {selected_year}: **{simulated:.2f} gCO₂/kWh**")

    st.markdown("---")
    st.caption("Model: LightGBM | 10 fitur input utama | Prediksi masa depan hingga 2100")
