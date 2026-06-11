import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------------------------------------------------
# LỆNH STREAMLIT ĐẦU TIÊN
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Hệ Thống Phát Hiện Giao Dịch Gian Lận",
    page_icon="🛡️"
)

# -----------------------------------------------------------------------------
# CÁC HÀM CACHE DÙNG CHUNG
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file_bytes, file_name):
    """
    Nạp dữ liệu từ bytes để đảm bảo cơ chế hashable của st.cache_data.
    Hỗ trợ cả file CSV và Excel.
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# Định nghĩa danh sách các biến đặc trưng (X) và biến mục tiêu (y) trích xuất từ notebook
FEATURE_COLS = [f"X_{i}" for i in range(1, 15)]
TARGET_COL = "default"

# -----------------------------------------------------------------------------
# SIDEBAR — VÙNG CẤU HÌNH
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải dữ liệu mẫu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện (CSV/Excel)", 
        type=["csv", "xlsx", "xls"],
        help="Chọn tệp dữ liệu mẫu chứa các cột từ X_1 đến X_14 và biến mục tiêu 'default'."
    )
    
    st.markdown("---")
    st.subheader("🤖 Tham số mô hình AI")
    st.caption("Thuật toán: Random Forest Classifier")
    
    # Các siêu tham số cấu hình dựa trên thuật toán RandomForest được sử dụng trong notebook
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, 
        max_value=300, 
        value=100, 
        step=10,
        help="Số lượng cây quyết định trong rừng."
    )
    
    criterion = st.selectbox(
        "Tiêu chí đo lường (criterion)", 
        options=["gini", "entropy", "log_loss"], 
        index=0,
        help="Hàm đo lường chất lượng phân tách các nút."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa (max_depth)", 
        min_value=1, 
        max_value=30, 
        value=10,
        help="Độ sâu tối đa của mỗi cây quyết định (None nếu để trống)."
    )
    
    # Cấu hình nâng cao gom gọn trong Expander
    with st.expander("🛠️ Cấu hình nâng cao"):
        random_state = st.number_input(
            "Mầm ngẫu nhiên (random_state)", 
            value=42, 
            step=1,
            help="Đảm bảo tính tái lập kết quả huấn luyện giữa các lần chạy."
        )
        test_size = st.slider(
            "Tỷ lệ dữ liệu kiểm tra (test_size)", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.3, 
            step=0.05,
            help="Tỷ lệ chia tập dữ liệu thử nghiệm/kiểm định mô hình."
        )

    st.divider()
    
    # Nút hành động duy nhất kích hoạt huấn luyện mô hình
    train_clicked = st.button(
        "🚀 Huấn luyện mô hình", 
        type="primary", 
        use_container_width=True,
        help="Bấm để bắt đầu chia dữ liệu, xử lý và huấn luyện mô hình Random Forest."
    )

# -----------------------------------------------------------------------------
# HEADER — VÙNG ĐỊNH HƯỚNG
# -----------------------------------------------------------------------------
st.title("🛡️ Ứng Dụng Phát Hiện Giao Dịch Gian Lận Tài Chính")
st.caption("Hệ thống Machine Learning phân tích hành vi và phân loại các giao dịch rủi ro/gian lận dựa trên mô hình Random Forest Classifier.")

df_main = None
if uploaded_file is not None:
    # Đọc dữ liệu qua hàm cache chung bằng cách truyền bytes và tên file
    file_bytes = uploaded_file.getvalue()
    df_main = load_data(file_bytes, uploaded_file.name)
    
    if df_main is not None:
        st.caption(f"📁 **Đang dùng tệp dữ liệu:** `{uploaded_file.name}` | Kích thước: {df_main.shape[0]} dòng, {df_main.shape[1]} cột")
        
        # Kiểm tra tính hợp lệ của cấu trúc dữ liệu
        missing_cols = [col for col in FEATURE_COLS + [TARGET_COL] if col not in df_main.columns]
        if missing_cols:
            st.error(f"❌ Tệp dữ liệu thiếu các cột bắt buộc sau: {missing_cols}")
            st.stop()
else:
    st.info("💡 **Hướng dẫn:** Vui lòng tải tệp dữ liệu mẫu (`dataset1.csv`) ở vùng thanh bên (Sidebar) bên trái để bắt đầu khám phá và huấn luyện mô hình.")
    st.stop()

st.divider()

# -----------------------------------------------------------------------------
# KHỐI XỬ LÝ HUẤN LUYỆN (Chạy khi bấm nút, lưu kết quả vào session_state)
# -----------------------------------------------------------------------------
if train_clicked and df_main is not None:
    with st.spinner("⏳ Đang huấn luyện mô hình... Vui lòng chờ trong giây lát..."):
        # Phân tách X và y
        X = df_main[FEATURE_COLS]
        y = df_main[TARGET_COL]
        
        # Chia tập dữ liệu Train/Test giống như luồng xử lý của bài toán
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Khởi tạo và huấn luyện mô hình
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Đánh giá và dự đoán mô hình
        y_pred = model.predict(X_test)
        
        try:
            y_probs = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_probs = None
            
        # Tính toán các chỉ tiêu đo lường chính hiệu năng phân loại
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Lưu các đối tượng quan trọng vào st.session_state
        st.session_state["trained_model"] = model
        st.session_state["metrics"] = metrics
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred
        st.session_state["y_probs"] = y_probs
        
    st.success("🎉 Huấn luyện mô hình thành công! Hãy chuyển sang các Tab bên dưới để xem kết quả chi tiết và dự báo.")

# -----------------------------------------------------------------------------
# CHIA CÁC TABS CHỨC NĂNG CHÍNH
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả & Kiểm định mô hình", 
    "🔮 Sử dụng mô hình dự báo"
])

# --- TAB 1: TỔNG QUAN DỮ LIỆU ---
with tab1:
    st.subheader("📋 Phân tích và Thống kê Mô tả Thô")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Tổng số dòng dữ liệu", f"{df_main.shape[0]:,}")
    with col_m2:
        st.metric("Tổng số cột đặc trưng", f"{len(FEATURE_COLS)}")
    with col_m3:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.metric("Dung lượng tệp tải lên", f"{file_size_mb:.2f} MB")
        
    st.write("### 🔍 Xem 5 hàng dữ liệu đầu tiên (Head)")
    st.dataframe(df_main.head(5), use_container_width=True)
    
    st.write("### 📐 Bảng thống kê mô tả chi tiết các biến mô hình")
    # Chỉ hiển thị mô tả cho các cột trực tiếp đưa vào mô hình hóa (X và y)
    st.dataframe(df_main[FEATURE_COLS + [TARGET_COL]].describe().T, use_container_width=True)

# --- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ---
with tab2:
    st.subheader("🖼️ Biểu đồ Phân Phối Các Biến Đầu Vào & Biến Mục Tiêu")
    
    # Biến mục tiêu phân loại nhãn nhị phân: default (0: Bình thường, 1: Gian lận)
    target_counts = df_main[TARGET_COL].value_counts().reset_index()
    target_counts.columns = ['Trạng thái', 'Số lượng']
    target_counts['Trạng thái'] = target_counts['Trạng thái'].map({0: "Bình thường (0)", 1: "Gian lận (1)"})
    
    # Tạo lưới biểu đồ 2x2 hiển thị phân phối
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        fig_target = px.bar(
            target_counts, x='Trạng thái', y='Số lượng',
            title=f"Phân bố biến mục tiêu kiểm định ({TARGET_COL})",
            color='Trạng thái', color_discrete_sequence=px.colors.qualitative.Set2,
            height=350
        )
        st.plotly_chart(fig_target, use_container_width=True)
        
    with col_g2:
        # Chọn mặc định cột X_1 đầu tiên làm phân phối số liên tục mẫu
        fig_x1 = px.histogram(
            df_main, x="X_1", color=TARGET_COL, barmode="overlay",
            title="Phân bố tần suất biến X_1 theo nhãn mục tiêu",
            marginal="box", height=350, color_discrete_sequence=["#1f77b4", "#ff7f0e"]
        )
        st.plotly_chart(fig_x1, use_container_width=True)
        
    col_g3, col_g4 = st.columns(2)
    
    with col_g3:
        # Biểu đồ heatmap ma trận tương quan giữa 6 biến đầu tiên giúp giảm tải giao diện
        corr_matrix = df_main[FEATURE_COLS[:6] + [TARGET_COL]].corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto",
            title="Ma trận tương quan tuyến tính (Một số biến đặc trưng đại diện X)",
            color_continuous_scale="RdBu_r", height=350
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with col_g4:
        # Biểu đồ hộp phân tán giá trị của biến X_13 để phát hiện các giá trị ngoại lai dị biệt
        fig_box = px.box(
            df_main, x=TARGET_COL, y="X_13",
            title="Biểu đồ hộp (Boxplot) biến X_13 phân mảnh theo nhãn",
            color=TARGET_COL, height=350
        )
        st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH ---
with tab3:
    st.subheader("🎯 Đánh Giá Hiệu Năng Phân Loại Của Thuật Toán")
    
    # Kiểm tra trạng thái đồng bộ đã bấm huấn luyện mô hình ở Sidebar chưa
    if "trained_model" not in st.session_state:
        st.info("⚠️ **Thông báo:** Hệ thống chưa nhận thấy dữ liệu huấn luyện nào được xử lý. Vui lòng thiết lập tham số và nhấn nút **'Huấn luyện mô hình'** tại thanh bên trái.")
    else:
        metrics = st.session_state["metrics"]
        
        # Khối Metrics đo lường các chỉ số chính diện
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Độ chính xác (Accuracy)", f"{metrics['accuracy']:.4f}")
        col_m2.metric("Độ chuẩn xác (Precision)", f"{metrics['precision']:.4f}")
        col_m3.metric("Độ nhạy hồi đáp (Recall)", f"{metrics['recall']:.4f}")
        col_m4.metric("Chỉ số F1-Score", f"{metrics['f1']:.4f}")
        
        st.markdown("---")
        
        col_layout1, col_layout2 = st.columns([1, 1])
        
        with col_layout1:
            st.write("### 🧮 Ma trận nhầm lẫn (Confusion Matrix)")
            cm = metrics["confusion_matrix"]
            
            # Tạo nhãn trực quan biểu diễn heatmap ma trận nhầm lẫn
            z_text = [[str(val) for val in row] for row in cm]
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế", color="Số lượng"),
                x=['Bình thường (0)', 'Gian lận (1)'],
                y=['Bình thường (0)', 'Gian lận (1)'],
                color_continuous_scale="Blues",
                height=380
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_layout2:
            st.write("### 📑 Báo cáo phân loại chi tiết (Classification Report)")
            report_df = pd.DataFrame(metrics["classification_report"]).transpose()
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)
            
            # Biểu đồ trực quan hóa tầm quan trọng các thuộc tính đầu vào (Feature Importance)
            st.write("### 🎖️ Độ quan trọng của các thuộc tính đầu vào")
            model_rf = st.session_state["trained_model"]
            importances = model_rf.feature_importances_
            feat_imp_df = pd.DataFrame({
                "Đặc trưng": FEATURE_COLS,
                "Độ quan trọng": importances
            }).sort_values(by="Độ quan trọng", ascending=True)
            
            fig_imp = px.bar(
                feat_imp_df, x="Độ quan trọng", y="Đặc trưng", orientation="h",
                color="Độ quan trọng", color_continuous_scale="Viridis",
                height=280
            )
            fig_imp.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_imp, use_container_width=True)

# --- TAB 4: SỬ DỤNG MÔ HÌNH DỰ BÁO ---
with tab4:
    st.subheader("🔮 Triển Khai Chẩn Đoán & Kiểm Tra Điểm Rủi Rủ")
    
    if "trained_model" not in st.session_state:
        st.info("⚠️ **Thông báo:** Vui lòng thực hiện bước huấn luyện mô hình ở Sidebar trước khi sử dụng tính năng dự báo rủi ro thực tế.")
    else:
        model = st.session_state["trained_model"]
        
        mode = st.radio(
            "Chọn phương thức nhập dữ liệu đầu vào:",
            options=["Chế độ 1: Nhập thông số trực tiếp đơn lẻ", "Chế độ 2: Tải file danh sách hàng loạt (Batch Inference)"],
            horizontal=True
        )
        
        st.markdown("---")
        
        if mode == "Chế độ 1: Nhập thông số trực tiếp đơn lẻ":
            st.write("### 🖊️ Điền các chỉ số thống kê của khách hàng/giao dịch")
            
            # Sử dụng st.form để bọc các widget tránh hiện tượng re-run mỗi khi người dùng đổi số
            with st.form("single_prediction_form"):
                # Tạo lưới form nhập liệu tối ưu diện tích giao diện
                cols_form = st.columns(4)
                input_data = {}
                
                for idx, col_name in enumerate(FEATURE_COLS):
                    # Tính toán giá trị mặc định dựa trên median dữ liệu gốc của notebook
                    default_val = float(df_main[col_name].median())
                    min_val = float(df_main[col_name].min())
                    max_val = float(df_main[col_name].max())
                    
                    target_col_form = cols_form[idx % 4]
                    with target_col_form:
                        input_data[col_name] = st.number_input(
                            f"Thông số {col_name}",
                            min_value=min_val - abs(min_val)*2,
                            max_value=max_value + abs(max_value)*2,
                            value=default_val,
                            format="%.6f",
                            help=f"Nhập chỉ số tương ứng cho biến {col_name}."
                        )
                
                submit_pred = st.form_submit_button("🎯 Tiến hành chẩn đoán rủi ro", type="primary")
                
            if submit_pred:
                # Chuyển đổi dữ liệu input sang cấu trúc DataFrame khớp chính xác schema
                input_df = pd.DataFrame([input_data])
                
                # Thực hiện dự đoán
                pred_class = model.predict(input_df)[0]
                pred_proba = model.predict_proba(input_df)[0]
                
                st.write("### 🕒 Kết quả phân tích hệ thống")
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    if pred_class == 1:
                        st.error("🚨 **Kết luận:** Phát hiện Giao dịch có dấu hiệu Gian lận hoặc Rủi ro cao!")
                    else:
                        st.success("✅ **Kết luận:** Giao dịch An toàn (Bình thường).")
                        
                with col_res2:
                    st.metric(
                        label="Xác suất rủi ro gian lận (Probability)", 
                        value=f"{pred_proba[1] * 100:.2f} %"
                    )
                    st.progress(float(pred_proba[1]))
                    
        else:
            st.write("### 📂 Dự báo hàng loạt qua file danh sách (X_new)")
            st.caption("Yêu cầu định dạng tệp tải lên phải chứa đầy đủ các cột thuộc tính đầu vào từ `X_1` đến `X_14`.")
            
            batch_file = st.file_uploader(
                "Tải lên tệp dữ liệu kiểm tra mới (Excel/CSV)", 
                type=["csv", "xlsx", "xls"],
                key="batch_prediction_uploader"
            )
            
            if batch_file is not None:
                batch_bytes = batch_file.getvalue()
                df_batch = load_data(batch_bytes, batch_file.name)
                
                if df_batch is not None:
                    # Kiểm tra tính đồng nhất schema cột dữ liệu mới
                    missing_batch_cols = [col for col in FEATURE_COLS if col not in df_batch.columns]
                    
                    if missing_batch_cols:
                        st.error(f"❌ Cấu trúc tệp sai lệch. Thiếu các cột bắt buộc sau: {missing_batch_cols}")
                    else:
                        # Thực thi dự đoán chuỗi hàng loạt
                        X_batch = df_batch[FEATURE_COLS]
                        batch_preds = model.predict(X_batch)
                        batch_probs = model.predict_proba(X_batch)[:, 1]
                        
                        # Đính kèm cột kết quả trực tiếp vào Dataframe xuất bản
                        df_result = df_batch.copy()
                        df_result["Dự_Báo_Nhãn"] = batch_preds
                        df_result["Xác_Suất_Rủi_Ro_Gian_Lận"] = batch_probs
                        
                        st.success(f"⚡ Đã hoàn tất phân tích hàng loạt cho {df_result.shape[0]} giao dịch thành công!")
                        
                        # Đặt trong container giới hạn chiều cao tránh tràn màn hình
                        with st.container(height=300):
                            st.dataframe(df_result, use_container_width=True)
                        
                        # Xuất bản dữ liệu kết quả để người dùng tải về máy cá nhân dưới định dạng CSV
                        csv_buffer = io.StringIO()
                        df_result.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Tải xuống bảng kết quả dự báo (.CSV)",
                            data=csv_data,
                            file_name="ket_qua_du_bao_gian_lan.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
