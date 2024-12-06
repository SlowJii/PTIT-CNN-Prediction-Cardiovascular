
# CNN Prediction Cardiovascular


![PTIT](https://img.shields.io/badge/PTIT-black?style=for-the-badge&logo=PTIT&logoColor=white&link=https%3A%2F%2Fptit.edu.vn%2F)


### THUẬT TOÁN HỌC MÁY

![MACHINE LEARNING SUMMARY](https://github.com/SlowJii/PTIT-CNN-Prediction-Cardiovascular/blob/main/the-world-of-machine-learning-algorithms-a-summary.jpg?raw=true)


# ***TÓM TẮT***

## LÝ DO CHỌN ĐỀ TÀI

### Tầm Quan Trọng Của Việc Chẩn Đoán Sớm Bệnh Tim

Chẩn đoán sớm bệnh tim đóng vai trò quan trọng trong việc giảm thiểu tỷ lệ tử vong và các biến chứng nghiêm trọng liên quan đến bệnh tim. Những bệnh lý như bệnh tim mạch, nhồi máu cơ tim, suy tim và đột quỵ có thể gây tổn thương vĩnh viễn cho cơ thể, nếu không được phát hiện và điều trị kịp thời. Việc phát hiện sớm giúp người bệnh có thể được can thiệp sớm, sử dụng các phương pháp điều trị phù hợp, từ đó cải thiện chất lượng sống và giảm thiểu chi phí điều trị dài hạn.

Chẩn đoán sớm cũng giúp bác sĩ đưa ra các biện pháp phòng ngừa, chẳng hạn như thay đổi lối sống, dùng thuốc điều trị huyết áp, cholesterol, hoặc các phương pháp phẫu thuật cần thiết. Các nghiên cứu cũng cho thấy rằng việc phát hiện sớm các yếu tố nguy cơ bệnh tim giúp giảm tỷ lệ tử vong do các bệnh lý tim mạch.

### Ảnh Hưởng Của Bệnh Tim Đến Sức Khỏe Cộng Đồng

Bệnh tim không chỉ ảnh hưởng đến sức khỏe cá nhân mà còn gây gánh nặng lớn cho hệ thống y tế và kinh tế cộng đồng. Theo Tổ chức Y tế Thế giới (WHO), bệnh tim mạch là nguyên nhân hàng đầu gây tử vong trên toàn cầu, chiếm tới gần 30% tổng số ca tử vong. Sự gia tăng các bệnh tim mạch làm tăng gánh nặng chi phí y tế và giảm năng suất lao động, tác động trực tiếp đến sự phát triển kinh tế xã hội.

Ngoài ra, bệnh tim còn ảnh hưởng đến chất lượng sống của người bệnh, gia đình họ và cộng đồng. Những người mắc bệnh tim có thể phải đối mặt với tình trạng giảm khả năng lao động, chi phí điều trị lâu dài, và gánh nặng tâm lý. Vì vậy, việc phòng ngừa và phát hiện sớm bệnh tim là một yếu tố quan trọng trong chiến lược chăm sóc sức khỏe cộng đồng.

### Khó Khăn Khi Dự Đoán Bệnh Từ Dữ Liệu Mất Cân Bằng

Dữ liệu mất cân bằng là vấn đề phổ biến trong các hệ thống dự đoán bệnh, đặc biệt là trong trường hợp bệnh tim. Trong các bộ dữ liệu này, số lượng người khỏe mạnh thường lớn hơn rất nhiều so với số lượng người mắc bệnh tim. Điều này dẫn đến một số khó khăn trong việc xây dựng mô hình dự đoán hiệu quả. Cụ thể:

- **Khó khăn trong việc học đúng mẫu:** Khi dữ liệu bị mất cân bằng, mô hình dự đoán có xu hướng học theo lớp dữ liệu chiếm ưu thế (người khỏe mạnh), dẫn đến dự đoán sai lệch cho lớp thiểu số (người mắc bệnh tim).
- **Hiệu suất kém đối với lớp thiểu số:** Mô hình có thể dự đoán chính xác với lớp người khỏe mạnh nhưng lại không nhận diện đúng các trường hợp mắc bệnh, khiến tỷ lệ phát hiện bệnh (sensitivity) thấp.
- **Chế độ dự đoán thiên lệch:** Các mô hình học máy có thể thiên về việc dự đoán lớp lớn hơn, dẫn đến việc bỏ sót nhiều trường hợp bệnh, dù có thể là những trường hợp quan trọng cần can thiệp sớm.

## MỤC TIÊU VÀ LÝ DO NGHIÊN CỨU

### MỤC ĐÍCH

Mục đích chính của nghiên cứu này là **đề xuất một kiến trúc mạng nơ-ron hiệu quả**, với các lớp chập (convolutional layers), nhằm phân loại dữ liệu lâm sàng có sự mất cân bằng lớp (class imbalance) nghiêm trọng. Dữ liệu được sử dụng trong nghiên cứu này được thu thập từ **Khảo sát Y tế và Dinh dưỡng Quốc gia (NHANES)**, và mục tiêu cuối cùng là **dự đoán sự xuất hiện của bệnh tim mạch vành (CHD)** từ bộ dữ liệu này. Bệnh tim mạch vành là một trong những nguyên nhân gây tử vong cao nhất, do đó việc phát hiện sớm bệnh là rất quan trọng.

Trong nghiên cứu này, tác giả sử dụng một phương pháp tiếp cận **hai bước** để giải quyết vấn đề mất cân bằng lớp trong dữ liệu, bắt đầu bằng việc sử dụng **hồi quy LASSO** để đánh giá trọng số của các đặc trưng và **majority-voting** để xác định các đặc trưng quan trọng. Các đặc trưng quan trọng này sau đó được **chuẩn hóa** thông qua một lớp kết nối đầy đủ (fully connected layer), một bước quan trọng trước khi truyền kết quả này vào các lớp chập tiếp theo của mạng nơ-ron.

Ngoài ra, nghiên cứu còn đề xuất một **quy trình huấn luyện theo từng epoch**, có sự tương đồng với quá trình **annealing mô phỏng (simulated annealing)**, nhằm tối ưu hóa độ chính xác phân loại của mô hình. Mục tiêu của phương pháp này là cải thiện độ chính xác phân loại, đặc biệt là trong trường hợp dữ liệu có sự mất cân bằng lớp nghiêm trọng, qua đó giảm thiểu sai sót trong việc dự đoán các ca mắc bệnh, đặc biệt là bệnh nhân có nguy cơ cao.


### LÍ DO NGHIÊN CỨU

Lý do chính để thực hiện nghiên cứu này xuất phát từ **tầm quan trọng của việc chẩn đoán sớm bệnh tim mạch**, đặc biệt là bệnh tim mạch vành (**CHD**), trong việc giảm thiểu tỷ lệ tử vong và các biến chứng nghiêm trọng liên quan đến bệnh. **Bệnh tim mạch vành** là nguyên nhân gây tử vong hàng đầu trên thế giới, chiếm một tỷ lệ lớn trong tổng số ca tử vong, đặc biệt tại các quốc gia phát triển như Mỹ, nơi có tỷ lệ tử vong do bệnh tim chiếm gần 13% (Benjamin, 2019). Việc phát hiện sớm bệnh giúp tăng cơ hội điều trị kịp thời, từ đó giảm thiểu gánh nặng về sức khỏe và chi phí điều trị lâu dài cho bệnh nhân.

Tuy nhiên, trong thực tế, việc phân loại và chẩn đoán bệnh tim từ các dữ liệu lâm sàng có sự **mất cân bằng lớp** nghiêm trọng, với số lượng người khỏe mạnh chiếm ưu thế rất lớn so với số người mắc bệnh, gây ra các thách thức lớn trong việc phát triển các mô hình học máy chính xác. Mặc dù các mô hình học máy đã được ứng dụng rộng rãi trong các nghiên cứu về bệnh tim mạch, nhưng chúng vẫn gặp phải vấn đề **thiếu chính xác** trong việc dự đoán đúng các trường hợp mắc bệnh (**CHD**), đặc biệt khi kích thước dữ liệu kiểm tra tăng lên, và tỷ lệ người mắc bệnh giảm đi.

Do đó, lý do nghiên cứu này là nhằm khắc phục hạn chế này bằng cách **phát triển một kiến trúc mạng nơ-ron sâu (CNN)** đơn giản nhưng mạnh mẽ, có thể **chống lại sự mất cân bằng lớp** mà vẫn đạt được hiệu quả cao trong việc phân loại đúng các ca mắc bệnh và không mắc bệnh. Nghiên cứu này đặc biệt chú trọng vào việc sử dụng **LASSO** để loại bỏ các đặc trưng không quan trọng, giúp mô hình chỉ tập trung vào các đặc trưng có ảnh hưởng lớn đến dự đoán bệnh tim. Hơn nữa, nghiên cứu này muốn chứng minh rằng **mạng nơ-ron sâu** có thể vượt qua các phương pháp truyền thống như **SVM** và **Random Forest** trong việc cải thiện độ chính xác khi dự đoán cả các ca bệnh âm tính (non-CHD) và dương tính (CHD) với dữ liệu mất cân bằng.

Một yếu tố quan trọng khác của nghiên cứu này là việc mô hình đề xuất sử dụng một phương pháp huấn luyện **simulated annealing-like**, giúp tối ưu hóa độ chính xác phân loại qua các epoch huấn luyện, từ đó làm giảm sai số tổng quát giữa bộ dữ liệu huấn luyện và kiểm tra. Phương pháp này không chỉ cải thiện độ chính xác mà còn giúp tăng tính ổn định và khả năng tổng quát của mô hình, điều này đặc biệt quan trọng khi làm việc với các bộ dữ liệu lâm sàng thực tế, có sự mất cân bằng dữ liệu rõ rệt như bộ dữ liệu NHANES.

Cuối cùng, mục tiêu của nghiên cứu là đưa ra một giải pháp có thể **áp dụng rộng rãi** trong các nghiên cứu y tế khác, nơi dữ liệu có sự mất cân bằng lớp tương tự và trong các bài toán phân loại bệnh trong lĩnh vực chăm sóc sức khỏe.


## PHƯƠNG PHÁP NGHIÊN CỨU

### 1. Dữ liệu và Tiền xử lý

- **Bộ dữ liệu**: Bộ dữ liệu được sử dụng trong nghiên cứu là từ **Khảo sát Y tế và Dinh dưỡng Quốc gia (NHANES)**. Bộ dữ liệu này chứa các thông tin lâm sàng, xét nghiệm và dữ liệu khám sức khỏe từ những người tham gia khảo sát.

- **Tiền xử lý dữ liệu**: Để giải quyết vấn đề **mất cân bằng lớp** trong bộ dữ liệu (với tỷ lệ giữa nhóm không mắc bệnh tim và nhóm mắc bệnh tim mạch vành là 35:1), nghiên cứu sử dụng các phương pháp tiền xử lý để cải thiện chất lượng dữ liệu trước khi huấn luyện mô hình. Đặc biệt, **hồi quy LASSO** được sử dụng để **đánh giá trọng số các đặc trưng** và **loại bỏ các đặc trưng không quan trọng**.

### 2. Phương pháp chọn lọc đặc trưng (Feature Selection)

- **Hồi quy LASSO (Least Absolute Shrinkage and Selection Operator)**:
  - Là một phương pháp được áp dụng để **đánh giá trọng số** của các đặc trưng trong bộ dữ liệu và **loại bỏ các đặc trưng không có ảnh hưởng đáng kể** đến việc dự đoán. 
  - Hồi quy LASSO giúp **giảm thiểu số lượng đặc trưng cần thiết**, đồng thời tăng tính khả thi và hiệu quả cho mô hình.

- **Majority-Voting**:
  - Sau khi các đặc trưng quan trọng được chọn qua **LASSO**, **majority-voting** được sử dụng để xác định **các đặc trưng quan trọng nhất**, giúp tăng cường khả năng phân loại chính xác.

### 3. Kiến trúc Mạng Nơ-ron Sâu (Deep Neural Network - DNN)

- **Mạng nơ-ron sâu với các lớp chập (CNN)**:
  - Mô hình được đề xuất trong nghiên cứu là một **mạng nơ-ron sâu với các lớp chập (CNN)**. Kiến trúc này được thiết kế đơn giản nhưng mạnh mẽ, bao gồm một số lớp chập để học các đặc trưng phức tạp trong dữ liệu lâm sàng.

- **Lớp kết nối đầy đủ (Fully Connected Layer)**:
  - Các đặc trưng quan trọng được **chuẩn hóa** bằng lớp kết nối đầy đủ trước khi truyền qua các lớp chập của mạng nơ-ron.

- **Huấn luyện mô hình**:
  - Phương pháp huấn luyện mô hình theo **epoch**, với một quy trình huấn luyện được thiết kế tương tự như quá trình **simulated annealing**, giúp tối ưu hóa độ chính xác phân loại và giảm thiểu sai số tổng quát giữa bộ huấn luyện và kiểm tra.

### 4. Đánh giá mô hình và hiệu suất

- Để đánh giá hiệu quả của mô hình, các **đo lường hiệu suất** được sử dụng bao gồm các chỉ số như **độ chính xác phân loại (accuracy)**, **nhạy cảm (sensitivity)**, **đặc hiệu (specificity)**, **tỷ lệ tái phát (recall)** và **diện tích dưới đường cong (AUC)**.

- Mô hình đề xuất được so sánh với các mô hình học máy khác như **SVM** và **Random Forest** để chứng minh sự vượt trội của nó trong việc phân loại cả các ca bệnh dương tính và âm tính.

### 5. Chế độ huấn luyện đặc biệt

- Để tối ưu hóa độ chính xác và khả năng tổng quát của mô hình, **quy trình huấn luyện theo từng epoch** được thiết kế giống như quá trình **simulated annealing**, giúp giảm thiểu lỗi tổng quát giữa bộ dữ liệu huấn luyện và kiểm tra, từ đó cải thiện hiệu suất phân loại trong các tình huống dữ liệu mất cân bằng.

# ***CƠ SỞ LÝ THUYẾT***

# Machine Learning (Học Máy)

**Machine Learning (ML)** là một nhánh của trí tuệ nhân tạo (AI), tập trung vào việc phát triển các thuật toán và mô hình giúp máy tính có thể học hỏi từ dữ liệu mà không cần lập trình cụ thể. ML sử dụng các kỹ thuật thống kê để phân tích, dự đoán xu hướng, và cải thiện độ chính xác theo thời gian khi có thêm dữ liệu.

## Các loại Machine Learning:

### 1. Supervised Learning (Học có giám sát)
- **Dữ liệu huấn luyện:** Bao gồm các ví dụ có nhãn (label), tức là dữ liệu đầu vào (input) được gắn với kết quả (output) đúng.
- **Mục tiêu:** Học một hàm ánh xạ giữa đầu vào và đầu ra để dự đoán chính xác với dữ liệu chưa từng thấy.
- **Ứng dụng phổ biến:**
  - Phân loại (Classification): Nhận diện email spam, dự đoán bệnh tật.
  - Hồi quy (Regression): Dự đoán giá nhà, dự báo thời tiết.

### 2. Unsupervised Learning (Học không giám sát)
- **Dữ liệu huấn luyện:** Không có nhãn, và mục tiêu là khám phá các cấu trúc ẩn trong dữ liệu.
- **Phương pháp chính:**
  - **Phân cụm (Clustering):** Nhóm các điểm dữ liệu tương tự nhau (VD: phân nhóm khách hàng).
  - **Giảm chiều (Dimensionality Reduction):** Giảm số lượng đặc trưng mà vẫn giữ được thông tin quan trọng (VD: PCA).
- **Ứng dụng phổ biến:** Phân tích thị trường, phát hiện bất thường.

### 3. Reinforcement Learning (Học củng cố)
- **Phương pháp học:** Mô hình học cách tối ưu hành động thông qua tương tác với môi trường và nhận phản hồi (phần thưởng hoặc hình phạt).
- **Mục tiêu:** Xây dựng chiến lược hành động tốt nhất để tối đa hóa phần thưởng trong dài hạn.
- **Ứng dụng phổ biến:** 
  - Robot tự hành.
  - Chơi game (AlphaGo, cờ vua).
  - Hệ thống đề xuất.

---

## Ứng dụng của Machine Learning
- **Phân loại:** Xác định email spam, phân tích y học (như chẩn đoán bệnh).
- **Dự đoán:** Dự báo tài chính, dự đoán thời tiết.
- **Nhận diện hình ảnh:** Xác định khuôn mặt, phân tích ảnh y tế.
- **Phân nhóm:** Phân khúc khách hàng, khám phá các mẫu trong dữ liệu.
- **Tự động hóa:** Hệ thống chatbot, nhận diện giọng nói.

---

## Các đặc điểm nổi bật của Machine Learning
1. **Khả năng học hỏi từ dữ liệu:** ML cải thiện hiệu suất dựa trên dữ liệu đầu vào.
2. **Khả năng tổng quát hóa:** ML dự đoán hiệu quả trên dữ liệu chưa từng thấy nếu được huấn luyện tốt.
3. **Tính linh hoạt:** ML có thể áp dụng trong nhiều lĩnh vực như tài chính, y tế, giáo dục, công nghệ.

---

Machine Learning là nền tảng quan trọng trong trí tuệ nhân tạo, tạo ra nhiều đột phá trong các lĩnh vực khoa học và công nghệ hiện đại.


# Deep Learning (Học Sâu)

**Deep Learning (DL)** là một lĩnh vực con của **Machine Learning**, chuyên nghiên cứu các mô hình học sâu (deep neural networks), nơi có nhiều lớp (layers) kết nối và xử lý thông tin. Mục tiêu của Deep Learning là tự động học các đặc trưng trong dữ liệu mà không cần phải trích xuất thủ công các đặc điểm này.

## Cấu Trúc của Deep Learning:
- **Neural Networks (Mạng Nơ-Ron):** Là mô hình cơ bản trong Deep Learning, bao gồm các lớp đầu vào, lớp ẩn và lớp đầu ra.
- **Deep Neural Networks (DNN):** Là các mạng nơ-ron có nhiều lớp ẩn (deep layers), giúp học các đặc trưng phức tạp từ dữ liệu lớn.
- **Convolutional Neural Networks (CNN):** Được sử dụng phổ biến trong nhận diện hình ảnh và video. CNN có khả năng nhận diện các đặc trưng như đường nét, hình dạng, và các chi tiết hình ảnh.
- **Recurrent Neural Networks (RNN):** Thường được dùng cho dữ liệu chuỗi như văn bản và âm thanh, vì chúng có khả năng ghi nhớ thông tin từ các bước trước để đưa ra dự đoán chính xác hơn.

## Các Loại Mô Hình trong Deep Learning:

### 1. Convolutional Neural Networks (CNNs)
- Được sử dụng chủ yếu trong nhận diện hình ảnh và video.
- CNN tự động phát hiện các đặc trưng trong ảnh mà không cần phải xây dựng thủ công các bộ lọc.
  
### 2. Recurrent Neural Networks (RNNs)
- Phù hợp với các bài toán dữ liệu chuỗi (time-series, văn bản).
- RNN có khả năng nhớ thông tin từ các bước trước trong chuỗi, giúp dự đoán chính xác hơn trong các tình huống liên tiếp.

### 3. Generative Adversarial Networks (GANs)
- Sử dụng hai mạng nơ-ron đối kháng với nhau: một mạng tạo ra dữ liệu giả, một mạng phân biệt dữ liệu giả và dữ liệu thật.
- Thường dùng trong tạo ảnh giả hoặc dữ liệu mới.

## Ưu Điểm của Deep Learning:
1. **Tự động học đặc trưng:** DL có khả năng học các đặc trưng phức tạp từ dữ liệu mà không cần phải trích xuất thủ công.
2. **Xử lý dữ liệu lớn:** DL rất hiệu quả trong việc xử lý và phân tích lượng dữ liệu lớn, đặc biệt là trong các bài toán như nhận diện hình ảnh và âm thanh.
3. **Hiệu suất cao:** Trong nhiều bài toán, các mô hình Deep Learning thường cho kết quả tốt hơn so với các phương pháp Machine Learning truyền thống.

## Ứng Dụng của Deep Learning:
1. **Nhận diện hình ảnh và video:** Xác định các đối tượng trong hình ảnh, nhận diện khuôn mặt, phân loại bệnh lý từ ảnh y tế.
2. **Tự động lái xe:** Deep Learning là một phần quan trọng trong công nghệ xe tự lái, giúp xe nhận diện các vật thể xung quanh.
3. **Xử lý ngôn ngữ tự nhiên (NLP):** Chuyển văn bản thành giọng nói, dịch máy, tạo ra các phản hồi tự động (chatbots).
4. **Sáng tạo nghệ thuật:** GANs có thể tạo ra các bức tranh, ảnh, âm nhạc giả.

## So Sánh Machine Learning và Deep Learning

| Đặc điểm                        | **Machine Learning**                                | **Deep Learning**                                   |
|----------------------------------|----------------------------------------------------|----------------------------------------------------|
| **Yêu cầu dữ liệu**              | Cần ít dữ liệu hơn, có thể sử dụng hiệu quả với dữ liệu nhỏ. | Cần lượng dữ liệu rất lớn để đạt hiệu suất tối ưu. |
| **Cấu trúc mô hình**             | Mô hình đơn giản, dễ hiểu hơn.                      | Mô hình phức tạp với nhiều lớp ẩn.                 |
| **Yêu cầu tính toán**            | Tính toán ít hơn so với Deep Learning.              | Yêu cầu tài nguyên tính toán mạnh mẽ (GPU, TPU).   |
| **Khả năng trích xuất đặc trưng** | Phải có sự can thiệp thủ công để trích xuất đặc trưng. | Tự động trích xuất đặc trưng từ dữ liệu.           |
| **Ứng dụng**                     | Phân loại, hồi quy, phân cụm đơn giản.              | Nhận diện hình ảnh, xử lý ngôn ngữ tự nhiên, xe tự lái. |

---

### Kết luận
- **Machine Learning** là một khái niệm rộng bao gồm nhiều thuật toán học từ dữ liệu và được ứng dụng trong nhiều lĩnh vực.
- **Deep Learning** là một lĩnh vực con mạnh mẽ hơn, phù hợp với các bài toán phức tạp như nhận diện hình ảnh và ngôn ngữ tự nhiên, nhưng yêu cầu dữ liệu và tính toán mạnh mẽ hơn.

Cả hai lĩnh vực đều có những ưu điểm riêng và có thể được sử dụng kết hợp trong các ứng dụng thực tế để mang lại hiệu quả tối đa.
