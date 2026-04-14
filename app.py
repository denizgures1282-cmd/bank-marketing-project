import streamlit as st
import pandas as pd
import joblib

# 1. Kaydettiğimiz Pipeline ve Sütun İsimlerini Yükleyelim
model = joblib.load('bank_model_pipeline.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# 2. Web Sayfası Başlığı
st.set_page_config(page_title="Banka Abonelik Tahmin Sistemi", layout="centered")
st.title("🏦 Banka Mevduat Abonelik Tahmini")
st.write("Müşteri bilgilerini girerek abone olup olmayacağını tahmin edin.")

# 3. Kullanıcı Giriş Alanları (Sidebar - Yan Menü)
st.sidebar.header("Müşteri Bilgileri")

age = st.sidebar.number_input("Yaş", min_value=18, max_value=100, value=30)
duration = st.sidebar.number_input("Görüşme Süresi (Saniye)", min_value=0, value=200)
campaign = st.sidebar.number_input("Kampanya Araması Sayısı", min_value=1, value=1)
pdays = st.sidebar.number_input("Önceki Aramadan Geçen Gün (999=Hiç aranmadı)", value=999)
previous = st.sidebar.number_input("Önceki Temas Sayısı", min_value=0, value=0)

# 4. Tahmin Butonu ve İşlem
if st.button("Tahmin Et"):
    # Girdi verilerini modelin anlayacağı formata sokmamız lazım
    # (Hızlı olması için basit bir örnek veri seti oluşturuyoruz)
    input_data = pd.DataFrame([[age, duration, campaign, pdays, previous]], 
                              columns=['age', 'duration', 'campaign', 'pdays', 'previous'])
    
    # Eksik olan diğer sütunları (meslekler, aylar vb.) 0 ile dolduralım
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Sütun sırasını modelle aynı yapalım
    input_data = input_data[feature_columns]
    
    # Tahmin
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    # Sonucu Göster
    st.divider()
    if prediction[0] == 1:
        st.success(f"🎉 Tahmin: Müşteri ABONE OLUR! (Olasılık: %{probability*100:.2f})")
    else:
        st.error(f"❌ Tahmin: Müşteri ABONE OLMAZ. (Olasılık: %{(1-probability)*100:.2f})")