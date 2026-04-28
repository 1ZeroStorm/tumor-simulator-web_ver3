import streamlit as st
import pandas as pd
import time
import os

# Konfigurasi Halaman
st.set_page_config(page_title="OncoSteer 100-Step Monitor", layout="wide")
st.title("📊 Training Monitor (Interval 100 Steps)")

log_file = "live_metrics.csv"
placeholder = st.empty()

while True:
    with placeholder.container():
        if os.path.exists(log_file):
            try:
                # 1. Baca data asli
                df = pd.read_csv(log_file)
                
                if not df.empty:
                    # 1. Filter baris di mana 'Step' adalah kelipatan 100
                    df_filtered = df[df['Step'] % 10 == 0]
                    
                    # 2. Ambil 3000 data TERAKHIR dari hasil filter tersebut
                    # Jika data belum mencapai 3000, ia akan mengambil semua yang ada
                    df_100 = df_filtered.tail(3000)
                    
                    # Jika data kelipatan 100 belum ada, tampilkan data terakhir saja
                    display_df = df_100 if not df_100.empty else df.tail(1)

                    # 3. Layout Kolom Metrik
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Step", f"{df['Step'].iloc[-1]}")
                    with col2:
                        # Rata-rata reward dalam 100 data point terakhir di csv
                        avg_reward = df['Reward'].tail(100).mean()
                        st.metric("Avg Reward (Last 100 pts)", f"{avg_reward:.2f}")
                    with col3:
                        # Menghitung jumlah titik data yang tampil di chart
                        st.metric("Data Points Shown", len(df_100))

                    # 4. Grafik Garis
                    st.subheader("Progress Reward (Last 3000 Samples @ 100 steps)")
                    if not df_100.empty:
                        # Menggunakan index Step agar sumbu X akurat
                        st.line_chart(df_100.set_index("Step")["Reward"])
                    else:
                        st.info("Mengumpulkan data hingga mencapai 100 steps pertama...")
                    
                    # 5. Tabel Data Mentah (Opsional)
                    with st.expander("Lihat Riwayat Per 100 Steps"):
                        st.dataframe(df_100.sort_values(by="Step", ascending=False), use_container_width=True)

            except Exception as e:
                st.error(f"Error membaca data: {e}")
        else:
            st.warning("Menunggu 'live_metrics.csv' dibuat oleh script training...")

    # Refresh rate
    time.sleep(5) 
    st.rerun()