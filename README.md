# CNN Prediction Cardiovascular

![PTIT](https://img.shields.io/badge/PTIT-black?style=for-the-badge&logo=PTIT&logoColor=white&link=https%3A%2F%2Fptit.edu.vn%2F)


### THUẬT TOÁN HỌC MÁY

Định nghĩa: 
- Học máy (Machine Learning) là một nhánh của trí tuệ nhân tạo
- Các thuật toán giúp máy tính có thể học hỏi từ dữ liệu để giải quyết vấn đề cụ thể



1. Thuật toán hồi quy (Regression Algorithms):
Các thuật toán hồi quy dự đoán giá trị liên tục dựa trên mối quan hệ giữa các biến
- Hồi quy tuyến tính (Linear Regression): Mô hình dự đoán dựa trên mối quan hệ tuyến tính giữa các biến đầu vào và biến đầu ra. Mục tiêu là tìm đường hồi quy tốt nhất để tối thiểu hóa sai số giữa các giá trị dự đoán và giá trị thực 
    - Ứng dụng: Dự đoán giá nhà dựa trên diện tích, số phòng, vị trí
- Hồi quy Logicstic (Logicstic Regression): Mặc dù được gọi là hồi quy nhưng thực chất là một thuật toán phân loại cho các vấn đề nhị phân. Nó sử dụng hàm Logicstic (sigmoid) để chuyển đổi đầu ra thành xác suất, sau đó phân loại vào một trong hai nhóm
    - Ứng dụng: Phân loại email thành spam hoặc non-spam, dự đoán khả năng mua sản phẩm của khách hàng

2. Thuật toán phân loại (Classifier Algorithms):
Nhóm thuật toán này dùng để phân chia dữ liệu vào các nhóm hoặc nhãn dựa trên các đặc điểm của dữ liệu
- Phân loại tuyến tính (Linear Classifier): Đưa ra các quyết định phân loại dựa trên một đường thẳng phân tách các lớp
    - Ứng dụng: Phân loại hình ảnh đen trắng
- SVM: Xây dựng một mặt phẳng phân tách tối ưu giữa các lớp bằng cách tối đa hóa khoảng cách giữa hai lớp gần nhất. Hiệu quả với phân loại dữ liệu phân tán
    - Ứng dụng: Phân loại bệnh nhân thành các nhóm rủi ro khác nhau dựa trên dấu hiệu sức khỏe
- SRC: Phân loại dựa trên biểu diễn thưa của dữ liệu, rất phù hợp cho các tập dữ liệu lớn với nhiều biến
    - Ứng dụng: Nhận diện khuôn mặt trong các ứng dụng bảo mật và kiểm soát truy cập

3. Thuật toán dự trên mẫu (Instance-based Algorithms):
Các thuật toán này dựa trên so sánh các mẫu đã có trong dữ liệu để đưa ra dự đoán hoặc phân loại
- kNN: Đưa ra dự đoán hoặc phân loại bằng cách xem xét k điểm gần nhất (láng giềng) của điểm dữ liệu mới. Đây là thuật toán không cần học mà chỉ so sánh khoảng cách giữa các điểm
    - Ứng dụng: Đề xuất sản phẩm TMĐT dựa trên sản phẩm mà người dùng đã mua
- Learning Vector Quantization: Một mạng nơ-ron học có giám sát để biểu diễn vector của các lớp, giúp tối ưu hóa quá trình phân loại
    - Ứng dụng: Phân loại văn bản hoặc giọng nói thành các nhóm khác nhau (Ví dụ giọng nói nam, nữ, nam cao, nam trầm)

4. Thuật toán điều khiển (Regularization Algorithms):
Các thuật toán này được thiết kế để hạn chế overfitting bằng cách thêm các điều kiện ràng buộc vào mô hình, làm giảm mức độ phức tạp của mô hình
- Hồi quy Ridge: Thêm độ phạt L2 vào hàm mất mát, giảm thiểu ảnh hưởng cảu các biến đầu vào không quan trọng
    - Ứng dụng: Dự đoán giá cổ phiếu trong tài chính, giúp tránh hiện tượng overfitting khi có nhiều biến
- LASSO: Thêm độ phạt L1, giúp triệt tiêu hoàn toàn trọng số của biến không quan trọng, làm đơn giản hóa mô hình
    - Ứng dụng: Tối ưu hóa chọn lọc biến trong phân tích hồi quy, chẳng hạn trong phân tích các yếu tố nguy cơ sức khỏe
