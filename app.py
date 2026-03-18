import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.impute import SimpleImputer

# =========================================================
# 1. LOAD MODEL & TOOLS
# =========================================================
@st.cache_resource # Dùng cache để app load nhanh hơn mỗi khi refresh
def load_assets():
    with open('all_tools.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

try:
    data = load_assets()
    model = data['model']
    tfidf = data['tfidf']
    svd = data['svd']
    num_cols = data['num_cols']
    
    # TỰ KHỞI TẠO IMPUTER MỚI (Để tránh lỗi phiên bản scikit-learn)
    # Chúng ta dùng chiến lược điền giá trị 0 hoặc median giả định
    imputer = SimpleImputer(strategy="constant", fill_value=0) 
except Exception as e:
    st.error(f"Lỗi load model: {e}")
    st.stop()

# =========================================================
# 2. GIAO DIỆN APP
# =========================================================
st.set_page_config(page_title="Dự đoán Học vụ BAV", layout="centered")

st.title("🎓 Hệ thống Cảnh báo học vụ (Demo)")
st.markdown("---")

# Cột bên trái: Nhập ghi chú văn bản
st.subheader("📝 Nhập Ghi chú của Cố vấn")
advisor_notes = st.text_area("Advisor Notes:", "Sinh viên vắng nhiều buổi, kết quả kiểm tra thấp.", height=150)

# Cột bên dưới: Nhập các chỉ số số học
st.subheader("📊 Chỉ số học tập (Numeric Features)")
col1, col2 = st.columns(2)

input_dict = {}
for i, col in enumerate(num_cols):
    # Chia làm 2 cột cho đẹp giao diện
    with col1 if i % 2 == 0 else col2:
        input_dict[col] = st.number_input(f"{col}", value=0.0)

# =========================================================
# 3. XỬ LÝ DỰ ĐOÁN
# =========================================================
if st.button("🚀 KIỂM TRA NGAY", use_container_width=True):
    
    # --- Bước A: Xử lý Text ---
    def clean_text(text):
        text = str(text).lower()
        tokens = re.findall(r'[a-z0-9áàảãạăắằặẳẵâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+', text)
        return " ".join(tokens)
    
    cleaned_notes = clean_text(advisor_notes)
    text_tfidf = tfidf.transform([cleaned_notes])
    text_svd = svd.transform(text_tfidf)

    # --- Bước B: Xử lý Số ---
    num_df = pd.DataFrame([input_dict])
    # Ép kiểu về float để tránh lỗi dtype
    num_arr = num_df.values.astype(np.float64)
    # Vì imputer mới chưa được fit, ta sẽ xử lý thủ công điền Na bằng 0 nếu có
    num_imputed = np.nan_to_num(num_arr, nan=0.0)

    # --- Bước C: Kết hợp & Dự đoán ---
    final_input = np.hstack([num_imputed, text_svd])
    prediction = model.predict(final_input)
    
    # Map kết quả (Dựa trên nhãn trong Kaggle của bạn)
    target_map = {0: "Bình thường (Normal)", 1: "Cảnh báo (Warning)", 2: "Nghỉ học (Dropout)"}
    res_label = target_map.get(int(prediction[0]), "Không xác định")

    # --- Hiển thị kết quả ---
    st.markdown("---")
    if int(prediction[0]) == 0:
        st.success(f"### TRẠNG THÁI: {res_label}")
    elif int(prediction[0]) == 1:
        st.warning(f"### TRẠNG THÁI: {res_label}")
    else:
        st.error(f"### TRẠNG THÁI: {res_label}")

st.markdown("<br><center><small>Powered by Gemini & Streamlit Cloud</small></center>", unsafe_content_html=True)
