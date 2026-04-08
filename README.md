# Vietnamese Dependency Parsing

Mô hình phân tích cú pháp phụ thuộc tiếng Việt dựa trên `PhoBERT` và kiến trúc `Biaffine Neural Dependency Parser`.

## 1. Cấu trúc dự án

```text
.
├── train.py
├── predict.py
├── evaluate.py
├── prepare_data.py
├── requirements.txt
└── src
    ├── dataset
    ├── model
    └── training
```

## 2. Yêu cầu môi trường

- Python 3.9+
- Khuyến nghị có GPU
- Lần chạy đầu cần Internet để tải `vinai/phobert-base`

## 3. Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Dữ liệu VLSP 2020

Dữ liệu gốc trong repo:

```text
data/DP-2020/
├── TrainingData/
├── DataTestDP2020-CoNLLU/
└── DataTestGoldDP2020/
```

Ý nghĩa:

- `TrainingData/`: có nhãn `HEAD` và `DEPREL`, dùng để train/dev
- `DataTestDP2020-CoNLLU/`: test input, chưa có nhãn phụ thuộc vàng
- `DataTestGoldDP2020/`: gold để đánh giá

Không train trực tiếp trên `DataTestDP2020-CoNLLU`.

## 5. Chuẩn bị dữ liệu làm việc riêng

Script [prepare_data.py](/Users/chipv/Library/Mobile%20Documents/com~apple~CloudDocs/CaoHocKHMT/dep-parsing/prepare_data.py) chỉ đọc dữ liệu gốc và sinh dữ liệu mới ra thư mục riêng, không chỉnh sửa dữ liệu nguồn.

```bash
python3 prepare_data.py
```

Đầu ra mặc định:

```text
prepared_data/dp2020/
├── train.conllu
├── dev.conllu
├── test_input.conllu
└── README.txt
```

Có thể đổi seed/tỷ lệ dev:

```bash
python3 prepare_data.py \
  --output-dir prepared_data/my_split \
  --dev-ratio 0.1 \
  --seed 42
```

## 6. Huấn luyện

Lệnh phù hợp với bộ VLSP trong repo:

```bash
python3 train.py \
  --train-path prepared_data/dp2020/train.conllu \
  --dev-path prepared_data/dp2020/dev.conllu \
  --model-name vinai/phobert-base \
  --batch-size 8 \
  --epochs 10 \
  --patience 3 \
  --max-len 256 \
  --hidden-dim 512 \
  --dropout 0.33 \
  --save-path checkpoints/best.pt
```

Tham số chính:

- `--train-path`: file train CoNLL-U
- `--dev-path`: file dev CoNLL-U
- `--test-path`: file test có gold, chỉ dùng nếu muốn đánh giá cuối quá trình
- `--model-name`: mặc định `vinai/phobert-base`
- `--batch-size`: kích thước batch
- `--epochs`: số epoch tối đa
- `--patience`: số epoch không cải thiện trước khi early stopping
- `--min-delta`: mức cải thiện LAS tối thiểu để coi là tốt hơn
- `--save-path`: nơi lưu checkpoint tốt nhất theo `LAS`

Sau mỗi epoch chương trình in `train_loss`, `dev_loss`, `dev_uas`, `dev_las`. Checkpoint tốt nhất được lưu vào `--save-path`.

## 7. Suy luận

### Một câu đơn

```bash
python3 predict.py \
  --checkpoint checkpoints/best.pt \
  --text "Tôi học xử_lý_ngôn_ngữ_tự_nhiên ."
```

### File text, mỗi dòng một câu

```bash
python3 predict.py \
  --checkpoint checkpoints/best.pt \
  --input-file samples.txt \
  --output-file predictions/predictions.conllu
```

### File CoNLL-U test input

```bash
python3 predict.py \
  --checkpoint checkpoints/best.pt \
  --input-file prepared_data/dp2020/test_input.conllu \
  --output-file predictions/test_predictions.conllu
```

Lưu ý: khi đầu vào là `.conllu`, script giữ nguyên tokenization của file gốc, kể cả token nhiều từ như `Trả lời`, `ủy ban`.

## 8. Đánh giá

Script [evaluate.py](/Users/chipv/Library/Mobile%20Documents/com~apple~CloudDocs/CaoHocKHMT/dep-parsing/evaluate.py) so `HEAD` và `DEPREL` giữa file gold và file dự đoán, rồi in `UAS/LAS`.

### Trường hợp gold và prediction khớp 1-1

```bash
python3 evaluate.py \
  --gold-path path/to/gold.conllu \
  --pred-path path/to/pred.conllu
```

### Trường hợp file prediction dài hơn gold

Một số file gold của VLSP chỉ tương ứng với phần đầu của file input. Khi đó dùng:

```bash
python3 evaluate.py \
  --gold-path data/DP-2020/DataTestGoldDP2020/test-from-vtb_gold.txt \
  --pred-path predictions/test_from_vtb_predictions.conllu \
  --truncate-pred-to-gold
```

Kết quả bạn đã chạy:

```text
Tokens: 21845
UAS: 0.8432
LAS: 0.7310
```

## 9. Workflow khuyến nghị cho VLSP 2020

1. Tạo dữ liệu làm việc riêng bằng `prepare_data.py`.
2. Train trên `prepared_data/dp2020/train.conllu`, theo dõi trên `dev.conllu`.
3. Dùng checkpoint tốt nhất `checkpoints/best.pt` để predict.
4. Đánh giá theo từng cặp `input/gold` tương ứng trong `DataTestDP2020-CoNLLU` và `DataTestGoldDP2020`.

Ví dụ cặp đã chạy đúng:

```bash
python3 predict.py \
  --checkpoint checkpoints/best.pt \
  --input-file data/DP-2020/DataTestDP2020-CoNLLU/Test-from-vtb.conllu \
  --output-file predictions/test_from_vtb_predictions.conllu

python3 evaluate.py \
  --gold-path data/DP-2020/DataTestGoldDP2020/test-from-vtb_gold.txt \
  --pred-path predictions/test_from_vtb_predictions.conllu \
  --truncate-pred-to-gold
```

## 10. Lưu ý

- `predict.py` cần checkpoint đã train xong.
- Với PyTorch mới, code đã dùng `weights_only=False` để load checkpoint tương thích.
- Mô hình hiện tại decode bằng `argmax`, chưa dùng MST decoding.
- Các cột `LEMMA`, `UPOS`, `XPOS`, `FEATS`, `DEPS`, `MISC` trong output suy luận đang để `_`.
- Nếu thiếu bộ nhớ GPU, giảm `--batch-size` xuống `4` hoặc `2`.

## 11. Kiểm tra nhanh

```bash
python3 -m py_compile train.py predict.py evaluate.py prepare_data.py
```
