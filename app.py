import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# Cấu hình trang
st.set_page_config(page_title="Dự báo Cảnh báo học vụ", layout="wide")

# =========================================================
# 1. LOAD ASSETS
# =========================================================
@st.cache_resource
def load_assets():
    with open('all_tools.pkl', 'rb') as f:
        return pickle.load(f)

data = load_assets()
model, tfidf, svd, num_cols = data['model'], data['tfidf'], data['svd'], data['num_cols']

# Hàm làm sạch văn bản
def clean_text(text):
    text = str(text).lower()
    tokens = re.findall(r'[a-z0-9áàảãạăắằặẳẵâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+', text)
    return " ".join(tokens)

# =========================================================
# 2. GIAO DIỆN CHÍNH
# =========================================================
st.title("🎓 Hệ thống Cảnh báo học vụ - BAV ITDE")
st.info("Hướng dẫn: Bạn có thể nhập lẻ từng sinh viên hoặc Upload file CSV (như file test) để dự đoán hàng loạt.")

tab1, tab2 = st.tabs(["📊 Dự đoán hàng loạt (Upload CSV)", "🔍 Dự đoán đơn lẻ"])

# ---------------------------------------------------------
# ---------------------------------------------------------
# TAB 1: DỰ ĐOÁN HÀNG LOẠT (Cập nhật để tự xử lý cột thiếu)
# ---------------------------------------------------------
with tab1:
    uploaded_file = st.file_uploader("Kéo thả file test.csv vào đây", type=["csv"])
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        
        if st.button("🚀 Chạy dự đoán toàn bộ"):
            df_proc = df_input.copy()
            
            # 1. Tự động tính toán Attendance Features nếu thiếu (Giống code Kaggle của bạn)
            att_cols_in_df = [c for c in df_proc.columns if "Att_Subject_" in c]
            if len(att_cols_in_df) > 0:
                vals = df_proc[att_cols_in_df].values
                valid = (vals >= 0) & (vals <= 20)
                df_proc["Att_Count"] = np.sum(valid, axis=1)
                df_proc["Att_Mean"]  = np.nanmean(np.where(valid, vals, np.nan), axis=1)
                df_proc["Att_Std"]   = np.nanstd(np.where(valid, vals, np.nan), axis=1)
                df_proc["Att_Low"]   = np.sum((vals >= 0) & (vals < 5), axis=1)
                df_proc[["Att_Mean", "Att_Std"]] = df_proc[["Att_Mean", "Att_Std"]].fillna(-1)

            # 2. Kiểm tra và bù các cột số còn thiếu bằng 0
            for col in num_cols:
                if col not in df_proc.columns:
                    df_proc[col] = 0
            
            # 3. Tiền xử lý Text
            texts = df_proc["Advisor_Notes"].apply(clean_text)
            x_text = svd.transform(tfidf.transform(texts))
            
            # 4. Lấy dữ liệu số theo đúng thứ tự num_cols
            x_num = df_proc[num_cols].fillna(0).values
            
            # 5. Dự đoán
            final_x = np.hstack([x_num, x_text])
            preds = model.predict(final_x)
            
            # Hiển thị
            target_map = {0: "Normal", 1: "Warning", 2: "Dropout"}
            df_input["Prediction"] = [target_map.get(int(p), "Unknown") for p in np.array(preds).flatten()]
            
            st.success("Đã dự đoán xong!")
            st.dataframe(df_input[["Student_ID", "Advisor_Notes", "Prediction"]], use_container_width=True)
            st.bar_chart(df_input["Prediction"].value_counts())
# ---------------------------------------------------------
# TAB 2: DỰ ĐOÁN ĐƠN LẺ (SỬA LỖI VÀ LÀM ĐẸP)
# ---------------------------------------------------------
with tab2:
    with st.form("single_predict"):
        st.subheader("Thông tin sinh viên")
        notes = st.text_area("Ghi chú của cố vấn (Advisor Notes)", "Sinh viên nghỉ học quá 20%")
        
        cols = st.columns(3)
        input_data = {}
        for i, col_name in enumerate(num_cols):
            with cols[i % 3]:
                # Tự động lấy giá trị trung bình mẫu để người dùng dễ test
                input_data[col_name] = st.number_input(f"{col_name}", value=0.0)
        
        submit = st.form_submit_button("Kiểm tra")
        
        if submit:
            # Xử lý text
            txt_feat = svd.transform(tfidf.transform([clean_text(notes)]))
            # Xử lý số
            num_feat = np.array([[input_data[c] for c in num_cols]])
            
            # Predict
            final_feat = np.hstack([num_feat, txt_feat])
            p = model.predict(final_feat)
            
            # Fix lỗi TypeError bằng cách ép kiểu chắc chắn về int
            res_idx = int(np.array(p).flatten()[0])
            
            target_map = {0: "Bình thường (Normal)", 1: "Cảnh báo (Warning)", 2: "Nghỉ học (Dropout)"}
            st.metric("Kết quả dự đoán:", target_map.get(res_idx, "Không xác định"))
