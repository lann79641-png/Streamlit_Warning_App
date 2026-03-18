import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# Load công cụ
with open('all_tools.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
tfidf = data['tfidf']
svd = data['svd']
imputer = data['imputer']
num_cols = data['num_cols']

st.title("Hệ thống Dự báo Cảnh báo học vụ 🎓")

# Giao diện nhập liệu
st.sidebar.header("Thông tin sinh viên")
advisor_notes = st.text_area("Ghi chú của cố vấn (Advisor Notes):", "Sinh viên đi học đầy đủ, tích cực.")

# Tạo các ô nhập số dựa trên các cột số trong data của bạn
input_dict = {}
for col in num_cols:
    input_dict[col] = st.sidebar.number_input(f"Nhập {col}", value=0.0)

if st.button("Kiểm tra trạng thái"):
    # 1. Xử lý Text
    def clean_text(text):
        text = str(text).lower()
        tokens = re.findall(r'[a-z0-9áàảãạăắằặẳẵâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+', text)
        return " ".join(tokens)
    
    clean_notes = clean_text(advisor_notes)
    text_tfidf = tfidf.transform([clean_notes])
    text_svd = svd.transform(text_tfidf)

    # 2. Xử lý Số
    num_df = pd.DataFrame([input_dict])
    num_imputed = imputer.transform(num_df)

    # 3. Kết hợp và Dự đoán
    final_input = np.hstack([num_imputed, text_svd])
    prediction = model.predict(final_input)
    
    target_map = {0: "Bình thường (Normal)", 1: "Cảnh báo (Warning)", 2: "Nghỉ học (Dropout)"}
    result = target_map[int(prediction[0])]
    
    if int(prediction[0]) == 0:
        st.success(f"Kết quả: {result}")
    else:
        st.error(f"Kết quả: {result}")