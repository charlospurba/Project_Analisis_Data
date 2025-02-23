import matplotlib.pyplot as plt
import streamlit as ss
import pandas as pd
from sklearn.cluster import KMeans

# sidebar
ss.text("Nama: Charlos Pardomuan Purba")
ss.title('Data Wrangling')

# Upload files using Streamlit file uploader
customers_df = pd.read_csv("../data/olist_customers_dataset.csv", delimiter=",") 
geolocation_df = pd.read_csv("../data/olist_geolocation_dataset.csv", delimiter=",")

# Filtering: User can select a province for analysis
selected_state = ss.selectbox("Pilih Provinsi untuk Filter", customers_df['customer_state'].unique())
filtered_by_state = customers_df[customers_df['customer_state'] == selected_state]

# Tabs for different sections
Gathering_Data, Assessing_Data, Cleaning_Data = ss.tabs(["Gathering Data", "Assessing Data", "Cleaning Data"])

# Gathering Data section
with Gathering_Data:
    tab1, tab2 = ss.tabs(["1", "2"])
    with tab1:
        ss.text("Dataset Customers")
        ss.dataframe(filtered_by_state)
    with tab2:
        ss.text("Dataset Geolocation")
        ss.dataframe(geolocation_df)

# Assessing Data section
with Assessing_Data:
    tab1, tab2, tab3 = ss.tabs(["1", "2", "3"])
    with tab1:
        ss.title("Analisis Data Pelanggan")
        ss.subheader("Jumlah Duplikasi:")
        ss.write(customers_df.isna().sum())
        ss.subheader("Nilai Unik dalam Kolom 'customer_state':")
        ss.write(customers_df['customer_state'].unique())
    with tab2:
        ss.title("Analisis Data Lokasi")
        ss.subheader("Jumlah Data Duplikat:")
        ss.write(geolocation_df.duplicated().sum())
    with tab3:
        ss.title("Analisis Data Lokasi")
        ss.subheader("Jumlah Nilai yang Hilang di Setiap Kolom:")
        ss.write(geolocation_df.isnull().sum())

# Cleaning Data section
with Cleaning_Data:
    tab1, tab2 = ss.tabs(["1", "2"])
    with tab1:
        ss.title("Pembersihan Data Pelanggan")
        customers_df.drop_duplicates(inplace=True)
        customers_df.drop(columns=['customer_unique_id'], inplace=True)
        ss.subheader("Data Pelanggan Setelah Pembersihan:")
        ss.dataframe(customers_df.head())
    with tab2:
        ss.title("Pembersihan Data Lokasi")
        geolocation_df.drop_duplicates(inplace=True)
        geolocation_df.drop(columns=['geolocation_zip_code_prefix'], inplace=True)
        ss.subheader("Data Lokasi Setelah Pembersihan:")
        ss.dataframe(geolocation_df.head())

# Exploratory Data Analysis (EDA)
ss.title('Exploratory Data Analysis (EDA)')
Tab1, Tab2, Tab3 = ss.tabs(["1", "2", "3"])

with Tab1:
    ss.title("Analisis Data Pelanggan per Provinsi")
    state_counts = customers_df.groupby('customer_state')['customer_id'].count()
    ss.subheader("Jumlah Pelanggan per Provinsi:")
    ss.dataframe(state_counts)

with Tab2:
    ss.title("Analisis Jumlah Pelanggan Berdasarkan Kota")
    # Menghitung jumlah pelanggan berdasarkan kota
    city_counts = customers_df.groupby('customer_city')['customer_id'].count()  # Pastikan 'customer_city' ada dalam dataset
    ss.subheader("Jumlah Pelanggan per Kota:")
    ss.dataframe(city_counts)

    # Filter kota dengan jumlah pelanggan > 1000 (atau sesuai dengan batas yang diinginkan)
    filtered_city_counts = city_counts[city_counts > 1000] 

    # Visualisasi Jumlah Pelanggan Berdasarkan Kota
    plt.figure(figsize=(12, 8))
    filtered_city_counts.plot(kind='bar', color='skyblue', width=0.8)  # Menyesuaikan lebar bar
    plt.title('Jumlah Pelanggan Berdasarkan Kota')
    plt.xlabel('Kota')
    plt.ylabel('Jumlah Pelanggan')
    plt.xticks(rotation=45, ha='right')  # Memutar label agar lebih terbaca
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    ss.pyplot(plt)

with Tab3:
    ss.title("Analisis Statistik Data Pelanggan")
    ss.subheader("Ringkasan Statistik Data:")
    stats_summary = customers_df.describe(include="all")
    ss.dataframe(stats_summary)

# Visualization & Explanatory Analysis
ss.title("Visualization & Explanatory Analysis")
Pertanyaan1, Pertanyaan2 = ss.tabs(['Distribusi Pelanggan per Provinsi', 'Distribusi Pelanggan per kota'])

with Pertanyaan1:
    ss.title("Distribusi Pelanggan per Provinsi")
    state_counts = customers_df.groupby('customer_state')['customer_id'].count()
    plt.figure(figsize=(10, 6))
    state_counts.plot(kind='bar', color='skyblue')
    plt.title('Jumlah Pelanggan Berdasarkan Provinsi')
    plt.xlabel('Provinsi')
    plt.ylabel('Jumlah Pelanggan')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    ss.pyplot(plt)

with Pertanyaan2:
    ss.title("Distribusi Jumlah Pelanggan Berdasarkan Kota")
    city_counts = customers_df.groupby('customer_city')['customer_id'].count()
    filtered_city_counts = city_counts[city_counts > 1000]  # Hanya menampilkan kota dengan lebih dari 1000 pelanggan
    plt.figure(figsize=(12, 8))
    filtered_city_counts.plot(kind='bar', color='skyblue')
    plt.title('Jumlah Pelanggan Berdasarkan Kota')
    plt.xlabel('Kota')
    plt.ylabel('Jumlah Pelanggan')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    ss.pyplot(plt)

# Advanced Analysis (Geospatial & Clustering)
ss.title("Analisis Lanjutan")
Geospatial_Analysis, Clustering = ss.tabs(["Geospatial Analysis", "Clustering"])

with Geospatial_Analysis:
    ss.title("Distribusi Pelanggan Berdasarkan Lokasi")
    plt.figure(figsize=(12, 8))
    plt.scatter(geolocation_df['geolocation_lng'], geolocation_df['geolocation_lat'],
                c='blue', alpha=0.5, edgecolors='w', s=10)
    plt.title('Distribusi Pelanggan Berdasarkan Lokasi Geografis')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    ss.pyplot(plt)

with Clustering:
    ss.title("Analisis Clustering Berdasarkan Lokasi Pelanggan")
    
    # Sample data for clustering to avoid memory issues
    geolocation_sample = geolocation_df.sample(1000)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    geolocation_sample['Cluster'] = kmeans.fit_predict(geolocation_sample[['geolocation_lat', 'geolocation_lng']])
    
    plt.figure(figsize=(12, 8))
    plt.scatter(geolocation_sample['geolocation_lng'], geolocation_sample['geolocation_lat'],
                c=geolocation_sample['Cluster'], cmap='viridis', alpha=0.6)
    plt.title('Clustered Locations of Customers')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    ss.pyplot(plt)

with ss.expander("Conclusion"):
    ss.write(
        """Kesimpulan:
        1. Analisis pelanggan berdasarkan provinsi menunjukkan bahwa wilayah seperti São Paulo memiliki konsentrasi pelanggan yang lebih tinggi.
        2. Penyebaran lokasi pelanggan menunjukkan konsentrasi utama di beberapa kota besar, seperti São Paulo.
        3. Clustering geospatial membantu mengidentifikasi pola distribusi pelanggan yang lebih terfokus di beberapa wilayah.
        """
    )
