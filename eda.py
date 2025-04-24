# Import library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set style visualisasi
sns.set(style='whitegrid', palette='pastel')

# Load data
try:
    df = pd.read_csv('heart.csv')  
except FileNotFoundError:
    print("File heart.csv tidak ditemukan. Pastikan file berada di direktori yang sama dengan script ini.")
    exit()

# Cek struktur data
print("Informasi Dataset:")
print(df.info())

# Cek statistik deskriptif
print("\nStatistik Deskriptif:")
print(df.describe())

# Cek nilai kosong
print("\nNilai Kosong per Kolom:")
print(df.isnull().sum())

# === VISUALISASI ===

# 1. Distribusi Target
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette='Set2')
plt.title('Distribusi Target Penyakit Jantung')
plt.xlabel('Target (0 = Tidak, 1 = Ya)')
plt.ylabel('Jumlah')
plt.tight_layout()
plt.show()

# 2. Distribusi Usia
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
plt.title('Distribusi Usia Pasien')
plt.xlabel('Usia')
plt.ylabel('Jumlah')
plt.tight_layout()
plt.show()

# 3. Boxplot Usia vs Target
plt.figure(figsize=(6, 4))
sns.boxplot(x='target', y='age', data=df, palette='pastel')
plt.title('Distribusi Usia Berdasarkan Status Penyakit')
plt.xlabel('Target')
plt.ylabel('Usia')
plt.tight_layout()
plt.show()

# 4. Tekanan Darah Istirahat (trestbps)
plt.figure(figsize=(8, 5))
sns.histplot(df['trestbps'], bins=30, kde=True, color='salmon')
plt.title('Distribusi Tekanan Darah Istirahat')
plt.xlabel('Tekanan Darah')
plt.ylabel('Jumlah')
plt.tight_layout()
plt.show()

# 5. Kolesterol (chol)
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='chol', hue='target', kde=True, palette='flare', bins=30)
plt.title('Kolesterol Berdasarkan Status Penyakit')
plt.xlabel('Kolesterol')
plt.ylabel('Jumlah')
plt.tight_layout()
plt.show()

# 6. Detak Jantung Maksimum (thalach)
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='thalach', hue='target', kde=True, palette='crest', bins=30)
plt.title('Detak Jantung Maksimum vs Penyakit')
plt.xlabel('Thalach')
plt.ylabel('Jumlah')
plt.tight_layout()
plt.show()

# 7. Jenis Kelamin vs Target
plt.figure(figsize=(6, 4))
sns.countplot(x='sex', hue='target', data=df, palette='coolwarm')
plt.title('Jenis Kelamin vs Target')
plt.xlabel('Jenis Kelamin (0 = Perempuan, 1 = Laki-laki)')
plt.ylabel('Jumlah')
plt.legend(title='Target', labels=['Tidak', 'Ya'])
plt.tight_layout()
plt.show()

# 8. Tipe Nyeri Dada (cp) vs Target
plt.figure(figsize=(6, 4))
sns.countplot(x='cp', hue='target', data=df, palette='Set3')
plt.title('Tipe Nyeri Dada dan Penyakit Jantung')
plt.xlabel('Chest Pain Type')
plt.ylabel('Jumlah')
plt.legend(title='Target', labels=['Tidak', 'Ya'])
plt.tight_layout()
plt.show()

# 9. Thal (Thalassemia)
plt.figure(figsize=(6, 4))
sns.countplot(x='thal', hue='target', data=df, palette='magma')
plt.title('Distribusi Thal Berdasarkan Target')
plt.xlabel('Thal')
plt.ylabel('Jumlah')
plt.legend(title='Target', labels=['Tidak', 'Ya'])
plt.tight_layout()
plt.show()

# 10. Heatmap Korelasi
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasi Antar Fitur')
plt.tight_layout()
plt.show()

# 11. Pairplot Fitur Penting
selected_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']
sns.pairplot(df[selected_features], hue='target', palette='Set1')
plt.suptitle('Pairplot Fitur Penting vs Target', y=1.02)
plt.show()
