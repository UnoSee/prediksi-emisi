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
# 🤖 HALAMAN 2: PREDIKSI INTENSITAS KARBON (LightGBM Forecast)
# ===========================================================
elif menu == "🤖 Prediksi Intensitas Karbon":
    st.title("🤖 Prediksi Intensitas Karbon Global (LightGBM Forecast Model)")
    st.markdown(
        "Model ini menggunakan **LightGBM Regressor** untuk memprediksi "
        "intensitas karbon global berdasarkan data historis pembangkit listrik dunia."
    )

    import lightgbm as lgb
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    # === 1️⃣ Load datasets
    try:
        df_carbon = pd.read_csv("carbon-intensity-electricity.csv")
        df_elec = pd.read_csv("elec-fossil-nuclear-renewables.csv")
        df_energy = pd.read_csv("per-capita-energy-use.csv")
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        st.stop()

    # === 2️⃣ Rename columns
    df_carbon = df_carbon.rename(columns={
        "Carbon intensity of electricity - gCO2/kWh": "carbon_intensity"
    })
    df_elec = df_elec.rename(columns={
        "Electricity from fossil fuels - TWh (adapted for visualization of chart elec-fossil-nuclear-renewables)": "fossil_twh",
        "Electricity from renewables - TWh (adapted for visualization of chart elec-fossil-nuclear-renewables)": "renewables_twh",
        "Electricity from nuclear - TWh (adapted for visualization of chart elec-fossil-nuclear-renewables)": "nuclear_twh",
    })
    df_energy = df_energy.rename(columns={
        "Primary energy consumption per capita (kWh/person)": "energy_use_per_capita"
    })

    # === 3️⃣ Merge datasets
    df_merged = pd.merge(df_carbon, df_elec, on=["Entity", "Code", "Year"], how="outer")
    df_merged = pd.merge(df_merged, df_energy, on=["Entity", "Code", "Year"], how="outer")

    df_merged = df_merged.dropna(subset=["carbon_intensity"])
    df_merged = df_merged.fillna(0)

    # === 4️⃣ Train/test split (real-world forecasting)
    train = df_merged[df_merged["Year"] < 2020]
    test = df_merged[df_merged["Year"] >= 2020]

    features = [
        "Year",
        "fossil_twh",
        "renewables_twh",
        "nuclear_twh",
        "energy_use_per_capita",
        "Entity"
    ]
    target = "carbon_intensity"

    X_train = train[features].copy()
    X_test = test[features].copy()
    y_train = train[target]
    y_test = test[target]

    X_train["Entity"] = X_train["Entity"].astype("category")
    X_test["Entity"] = X_test["Entity"].astype("category")

    # === 5️⃣ Train or load model
    model_path = "model_lightgbm_forecast.txt"
    if os.path.exists(model_path):
        model = lgb.Booster(model_file=model_path)
        st.success("✅ Model LightGBM berhasil dimuat dari file.")
    else:
        st.info("🚀 Melatih model LightGBM baru...")
        params = {
            "objective": "regression_l1",
            "metric": "l1",
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        model.booster_.save_model(model_path)
        st.success("✅ Model baru dilatih dan disimpan sebagai model_lightgbm_forecast.txt")

    # === 6️⃣ Prediction & evaluation
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("📊 Evaluasi Model (2020–2024)")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f} gCO₂/kWh")
    col2.metric("RMSE", f"{rmse:.2f} gCO₂/kWh")
    col3.metric("R²", f"{r2:.3f}")

    # === 7️⃣ Save and visualize forecast results
    test_pred = test.copy()
    test_pred["predicted"] = y_pred
    agg = test_pred.groupby("Year")[["carbon_intensity", "predicted"]].mean().reset_index()
    agg.to_csv("forecast_results.csv", index=False)

    st.success("📁 Forecast disimpan sebagai forecast_results.csv")

    # === 8️⃣ Clean interactive chart (Plotly)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["Year"],
        y=agg["carbon_intensity"],
        mode="lines+markers",
        name="Observed (Global Avg)",
        line=dict(width=3)
    ))
    fig.add_trace(go.Scatter(
        x=agg["Year"],
        y=agg["predicted"],
        mode="lines+markers",
        name="Predicted (Global Avg)",
        line=dict(width=3, dash="dash")
    ))
    fig.update_layout(
        title="🌍 Global Average Carbon Intensity: Observed vs Predicted",
        xaxis_title="Year",
        yaxis_title="Carbon Intensity (gCO₂/kWh)",
        template="plotly_white",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # === 9️⃣ Predict user-specified scenario (optional)
    st.markdown("---")
    st.subheader("🎯 Simulasi Tahun Masa Depan")

    year_input = st.number_input("Tahun Prediksi", min_value=2023, max_value=2100, value=2025, step=1)
    entity_input = st.selectbox("Pilih Entity (Negara/Wilayah)", sorted(df_merged["Entity"].dropna().unique()))
    fossil = st.number_input("Listrik dari Fossil (TWh)", min_value=0.0, value=1000.0)
    renew = st.number_input("Listrik dari Renewable (TWh)", min_value=0.0, value=500.0)
    nuclear = st.number_input("Listrik dari Nuklir (TWh)", min_value=0.0, value=100.0)
    energy_pc = st.number_input("Konsumsi Energi per Kapita (kWh/orang)", min_value=0.0, value=50000.0)

    if st.button("🔮 Prediksi Intensitas Karbon Tahun Tersebut"):
        total = fossil + renew + nuclear
        if total == 0:
            st.warning("⚠️ Semua input nol → Prediksi: 0 gCO₂/kWh")
        else:
            X_new = pd.DataFrame([{
                "Year": year_input,
                "fossil_twh": fossil,
                "renewables_twh": renew,
                "nuclear_twh": nuclear,
                "energy_use_per_capita": energy_pc,
                "Entity": entity_input
            }])
            X_new["Entity"] = X_new["Entity"].astype("category")
            pred_future = model.predict(X_new)[0]
            st.success(f"🌱 Prediksi Intensitas Karbon Tahun {year_input}: **{pred_future:.2f} gCO₂/kWh**")

    st.caption("© 2025 - Global Carbon Forecasting Model | LightGBM Regressor")