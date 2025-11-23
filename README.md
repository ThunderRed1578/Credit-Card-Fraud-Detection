<div style="text-align: center;"> 
    <span style="font-size: 40px; font-weight: bold">
        Credit Card Fraud Detection using NumPy
    </span>
</div>

> **Học phần:** Lập trình cho Khoa học dữ liệu - CQ2023/21  
> **Sinh viên:** Nguyễn Lê Tấn Phát - 22120262  
> **Ngày thực hiện:** Nov 22, 2025

Dự án xây dựng hệ thống phát hiện gian lận thẻ tín dụng (Fraud Detection) sử dụng thuần thư viện NumPy để triển khai các thuật toán Machine Learning và xử lý dữ liệu, thay vì phụ thuộc vào các thư viện high-level như Scikit-learn. Dự án tập trung giải quyết vấn đề mất cân bằng dữ liệu nghiêm trọng (Imbalanced Data) thông qua các kỹ thuật Resampling.

---

## **Mục lục**

I. [Giới thiệu](#i)

1. [Bài toán](#i_1)

2. [Động lực](#i_2)

3. [Mục tiêu](#i_3)

II. [Tập dữ liệu](#ii)

III. [Phương pháp](#iii)

1. [Tiền xử lý dữ liệu (Preprocessing)](#iii_1)

2. [Xử lý mất cân bằng (Resampling)](#iii_2)

3. [Thuật toán Logistic Regression](#iii_3)

IV. [Cài đặt](#iv)

1. [Clone repository](#iv_1)

2. [Tạo môi trường ảo (Khuyến nghị)](#iv_2)

3. [Cài đặt thư viện](#iv_3)

V. [Cách sử dụng](#v)

1. [01_data_exploration.ipynb](#v_1)

2. [02_preprocessing.ipynb](#v_2)

3. [03_modeling.ipynb](#v_3)

VI. [Kết quả](#vi)

VII. [Cấu trúc thư mục](#vii)

VIII. [Thách thức & Giải pháp](#viii)

1. [Hiệu năng của NumPy với dữ liệu lớn](#viii_1)

2. [Độ ổn định số học](#viii_2)

3. [Data Leakage](#viii_3)

IX. [Hướng phát triển tiếp theo](#ix)

X. [Contributor & Author](#x)

XI. [License](#xi)


---

<h2 id="i" style="font-weight: bold">I. Giới thiệu</h2>

<h3 id="i_1" style="font-weight: bold">1. Bài toán</h3>

Gian lận thẻ tín dụng là hành vi sử dụng thẻ thanh toán trái phép để chiếm đoạt tài sản. Thách thức lớn nhất là số lượng giao dịch gian lận thường rất nhỏ so với giao dịch hợp lệ, khiến các mô hình học máy thông thường dễ bị thiên lệch (bias) về phía đa số.

<h3 id="i_2" style="font-weight: bold">2. Động lực</h3>

Việc phát hiện sớm gian lận giúp giảm thiểu tổn thất tài chính cho ngân hàng và khách hàng. Dự án này không chỉ nhằm mục đích xây dựng mô hình dự đoán mà còn là cơ hội để **hiểu sâu về toán học và cách hoạt động bên dưới** của các thuật toán thông qua việc tự cài đặt bằng NumPy.

<h3 id="i_3" style="font-weight: bold">3. Mục tiêu</h3>

1. Phân tích khám phá dữ liệu (EDA) để tìm ra đặc trưng của hành vi gian lận.

2. Xử lý mất cân bằng dữ liệu bằng **Undersampling** và **SMOTE**.

3. Tự cài đặt thuật toán **Logistic Regression** bằng NumPy.

4. Đánh giá hiệu quả giữa các phương pháp xử lý dữ liệu dựa trên các chỉ số AUPRC, Recall và F1-Score.

---

<h2 id="ii" style="font-weight: bold">II. Tập dữ liệu</h2>

- **Nguồn dữ liệu:** [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

- **Kích thước:** 284,807 giao dịch.

- **Đặc điểm:**

    - **Mất cân bằng:** Chỉ có 492 giao dịch gian lận (chiếm **0.173%**).

    - **Features:**

        - `Time`: Thời gian trôi qua (giây) kể từ giao dịch đầu tiên.

        - `Amount`: Số tiền giao dịch.

        - `V1` - `V28`: Các đặc trưng ẩn đã được biến đổi qua PCA (để bảo mật thông tin).

        - `Class`: Nhãn (0: Bình thường, 1: Gian lận).

---

<h2 id="iii" style="font-weight: bold">III. Phương pháp</h2>

<h3 id="iii_1" style="font-weight: bold">1. Tiền xử lý dữ liệu (Preprocessing)</h3>

- **Robust Scaling:** Áp dụng cho `Time` và `Amount` để giảm ảnh hưởng của outliers.

    $$
    X_{scaled} = \frac{X - Q_2(X)}{Q_3(X) - Q_1(X)}
    $$

- **Outlier Removal:** Sử dụng phương pháp IQR (Interquartile Range) để loại bỏ các điểm nhiễu trong tập huấn luyện.

<h3 id="iii_2" style="font-weight: bold">2. Xử lý mất cân bằng (Resampling)</h3>

- **Random Undersampling:** Giảm ngẫu nhiên số lượng mẫu lớp đa số (Normal) bằng với số lượng lớp thiểu số (Fraud).

- **SMOTE (Synthetic Minority Over-sampling Technique):** Sinh thêm dữ liệu giả cho lớp Fraud dựa trên K-Nearest Neighbors.

    - Công thức sinh điểm mới:
    $$
    x_{new} = x_i + \lambda \times (x_{zi} - x_i)
    $$

    Trong đó $x_i$ là điểm dữ liệu gốc, $x_{zi}$ là một láng giềng ngẫu nhiên, và $\lambda \in [0, 1]$.

<h3 id="iii_3" style="font-weight: bold">3. Thuật toán Logistic Regression</h3>

Mô hình phân loại nhị phân sử dụng hàm kích hoạt Sigmoid và tối ưu hóa bằng Gradient Descent.

- **Hypothesis:**

    $$
    \hat{y} = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}
    $$

- **Cost Function (Log Loss):**

    $$
    J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]
    $$

- **Gradient Descent Update:**

    $$
    w := w - \alpha \frac{1}{m} X^T (\hat{y} - y)
    $$

---

<h2 id="iv" style="font-weight: bold">IV. Cài đặt</h2>

<h3 id="iv_1" style="font-weight: bold">1. Clone repository</h3>

```bash
git clone [https://github.com/username/credit-card-fraud-numpy.git](https://github.com/username/credit-card-fraud-numpy.git)
cd credit-card-fraud-numpy
```

<h3 id="iv_2" style="font-weight: bold">2. Tạo môi trường ảo (Khuyến nghị)</h3>

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

<h3 id="iv_3" style="font-weight: bold">3. Cài đặt thư viện</h3>

```bash
pip install numpy matplotlib seaborn
```

---

<h2 id="v" style="font-weight: bold">V. Cách sử dụng</h2>

Chạy lần lượt các file notebook:

<h3 id="v_1" style="font-weight: bold">1. 01_data_exploration.ipynb</h3>

- Phân tích phân phối dữ liệu.

- Trực quan hóa Correlation Matrix.

- Thực hiện giảm chiều dữ liệu (PCA, t-SNE) để quan sát khả năng phân tách lớp.

<h3 id="v_2" style="font-weight: bold">2. 02_preprocessing.ipynb</h3>

- Thực hiện Scaling.

- Tạo tập dữ liệu Undersampling và SMOTE.

- Lưu dữ liệu đã xử lý vào thư mục `data/processed/`.

<h3 id="v_3" style="font-weight: bold">3. 03_modeling.ipynb</h3>

- Load dữ liệu đã xử lý.

- Huấn luyện mô hình Logistic Regression.

- Đánh giá và so sánh kết quả.

---

<h2 id="vi" style="font-weight: bold">VI. Kết quả</h2>

Mô hình được đánh giá trên tập Test (20% dữ liệu gốc) với hai chiến lược huấn luyện khác nhau.

| Metric | Undersampling | SMOTE | Nhận xét |
| :--- | :--- | :--- | :--- |
| **AUPRC** | 0.6611 | **0.6729** | SMOTE tốt hơn một chút, cả hai đều khá ổn định. |
| **Recall** | 0.8476 | **0.8571** | SMOTE bắt được nhiều gian lận hơn (FN thấp nhất). |
| **Precision** | **0.0795** | 0.0582 | Cả hai đều thấp do trade-off (nhiều False Positive). |
| **F1-Score** | **0.1454** | 0.1090 | Undersampling cân bằng tốt hơn giữa Precision và Recall. |

**Biểu đồ Confusion Matrix:**

- *Undersampling:* Giảm thiểu False Negative (chỉ sót 16 ca), nhưng báo nhầm 1,030 ca.

- *SMOTE:* Giảm False Negative tối đa (chỉ sót 15 ca), nhưng báo nhầm lên tới 1,456 ca.

**Kết luận:**

- Nếu ưu tiên **không bỏ sót gian lận** (Recall cao nhất), nên chọn **SMOTE**.

- Nếu muốn giảm thiểu phiền hà cho khách hàng (bớt báo nhầm), nên chọn **Undersampling**.

---

<h2 id="vii" style="font-weight: bold">VII. Cấu trúc thư mục</h2>

```bash
creditcardfrauddetection
├── data
│  ├── raw 
│  │  └── creditcard.csv                # File gốc
│  └── processed 
│     ├── test_data.npz                 # File dữ liệu test chung
│     ├── train_smote.npz               # File train cho SMOTE
│     └── train_under.npz               # File train cho Undersampling
├── notebooks
│  ├── 01_data_exploration.ipynb        # EDA, Visualization, PCA/t-SNE 
│  ├── 02_preprocessing.ipynb           # Scaling, Resampling, Saving data 
│  └── 03_modeling.ipynb                # Training & Evaluation 
├── src
│  ├── data_processing.py               # Chứa hàm SMOTE, PCA, SVD 
│  ├── model.py                         # Class LogisticRegression 
│  └── visualization.py                 # Các hàm vẽ biểu đồ 
│ README.md
```

---

<h2 id="viii" style="font-weight: bold">VIII. Thách thức & Giải pháp</h2>

<h3 id="viii_1" style="font-weight: bold">1. Hiệu năng của NumPy với dữ liệu lớn</h3>

- **Khó khăn:** Việc tính toán khoảng cách trong KNN (cho SMOTE) hoặc Gradient Descent với vòng lặp Python thuần rất chậm.

- **Giải pháp:** Tận dụng Vectorization và Broadcasting của NumPy để thay thế hoàn toàn vòng lặp `for`, giúp tăng tốc độ tính toán.

<h3 id="viii_2" style="font-weight: bold">2. Độ ổn định số học</h3>

- **Khó khăn:** Hàm `log()` trong Log Loss gặp lỗi khi giá trị dự đoán $\hat{y}$ bằng 0 hoặc 1.

- **Giải pháp:** Sử dụng kỹ thuật Clipping giá trị $\hat{y}$ vào khoảng $[\epsilon, 1-\epsilon]$ (với $\epsilon = 1e-15$).

<h3 id="viii_3" style="font-weight: bold">3. Data Leakage</h3>

- **Khó khăn:** Áp dụng Resampling trước khi chia tập Train/Test dẫn đến kết quả ảo.

- **Giải pháp:** Luôn thực hiện `train_test_split` trước, sau đó mới áp dụng SMOTE/Undersampling chỉ trên tập Train.

---

<h2 id="ix" style="font-weight: bold">IX. Hướng phát triển tiếp theo</h2>

- Cài đặt thêm các thuật toán phức tạp hơn như Random Forest hoặc Neural Network.

- Tối ưu hóa Hyperparameter (Learning rate, Regularization) tự động.

- Xử lý False Positive tốt hơn bằng cách kết hợp các mô hình (Ensemble Learning).

---

<h2 id="x" style="font-weight: bold">X. Contributors & Author</h2>

**Tác giả:** Nguyễn Lê Tấn Phát

- **MSSV:** 22120262

- **Lớp:** Lập trình cho Khoa học dữ liệu - CQ2023/21

- **Email:** 22120262@student.hcmus.edu.vn

---

**Contact**

Nếu bạn có bất kỳ câu hỏi nào, vui lòng liên hệ qua email hoặc tạo issue trên repository này.

---

<h2 id="xi" style="font-weight: bold">XI. License</h2>

[Database: Open Database, Contents: Database Contents](https://opendatacommons.org/licenses/dbcl/1-0/).
