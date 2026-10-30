# GR00T N1.5 — LIBERO và RoboCasa

Fine-tune GR00T N1.5 và chạy thử policy trong simulator LIBERO hoặc RoboCasa. Các lệnh tải model, dataset, training và evaluation đều chạy **trong Docker**. `fixed horizon` được hỗ trợ; AAC chưa được triển khai.

## 1. Chuẩn bị trên host

Cần Linux x86-64, GPU NVIDIA hỗ trợ BF16, driver tương thích CUDA 12.8, Docker Compose, NVIDIA Container Toolkit, Internet và đủ dung lượng. Cấu hình mặc định hướng tới một GPU RTX 5090; mức VRAM thực tế chưa xác nhận.

```bash
nvidia-smi
docker compose version
docker run --rm --gpus '"device=0"' nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Đổi `0` thành GPU được cấp. Nếu container không thấy GPU, kiểm tra driver và NVIDIA Container Toolkit với quản trị viên.

## 2. Clone và cấu hình lưu trữ trên host

```bash
git clone https://github.com/hiwe0305/aac-humanoid.git aac-duac
cd aac-duac
mkdir -p data outputs logs .hf_cache
```

Tạo `docker/.env`, thay `0` bằng GPU được cấp:

```bash
cat > docker/.env <<EOF_CONFIG
GPU_ID=0
DATA_DIR_HOST=$PWD/data
OUT_DIR_HOST=$PWD/outputs
LOG_DIR_HOST=$PWD/logs
HF_CACHE_HOST=$PWD/.hf_cache
EOF_CONFIG
```

Muốn lưu ở ổ khác, sửa các đường dẫn host trong `docker/.env` và tạo các thư mục tương ứng. Container mặc định chạy bằng root; file được tạo trên host có thể thuộc root.

Mặc định cho 1 RTX 5090: `micro_batch: 4`, `global_batch: 256`, tích lũy 64 bước; đây là điểm bắt đầu chưa xác minh VRAM, không đảm bảo tái hiện batch của paper. Nếu hết VRAM, giảm `micro_batch` xuống 2 hoặc 1, batch hiệu dụng vẫn là 256. Sửa [configs/default.yaml](configs/default.yaml) nếu cần: `micro_batch`, `epochs`, `save_steps`, horizon và số trials. `global_batch` phải chia hết cho `micro_batch * num_gpus`. Giữ các đường dẫn mặc định trong YAML để dùng các thư mục đã mount.

## 3. Build và vào Docker

Chạy trên **host**, tại thư mục dự án:

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm gr00t bash
```

Dockerfile tải các repo Isaac-GR00T, LIBERO, robosuite, RoboCasa và assets, sau đó cài ba môi trường Python **bên trong image**. Các nhánh upstream chưa khóa commit. Khi thấy shell tại `/workspace/aac-duac`, chạy tất cả lệnh ở bước 4–6 **trong container**:

```bash
nvidia-smi
.venv-train/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()'
.venv-train/bin/python -m aac_duac.cli --help
```

Docker cũng tải ba repository của tác giả vào `external/aac-gr00t`, `external/aac-libero`, `external/aac-robocasa` với commit cố định để đối chiếu. CLI hiện chưa kết nối các repository này vào pipeline: server tác giả dùng giao thức và schema khác, cần chuyển đổi và kiểm thử trước khi dùng.

`.venv-train` là đường dẫn trong container, không cần cài Python hay môi trường ảo trên host. Hai môi trường `.venv-libero` và `.venv-robocasa` cũng nằm trong image; CLI tự chọn môi trường simulator tương ứng. `exit` để rời container, sau đó có thể vào lại bằng lệnh `docker compose ... run` ở trên.

## 4. Tải model và dữ liệu

Trong **container**:

```bash
# Nếu Hugging Face yêu cầu đăng nhập hoặc chấp nhận điều khoản:
.venv-train/bin/hf auth login

.venv-train/bin/python -m aac_duac.cli download model
.venv-train/bin/python -m aac_duac.cli download libero
.venv-train/bin/python -m aac_duac.cli download robocasa
# Hoặc tải cả ba: .venv-train/bin/python -m aac_duac.cli download all
```

Chỉ cần tải dataset dự định train. Model lưu trong `.hf_cache`; dataset trong `data/libero` và `data/robocasa` của container, tương ứng với các thư mục host đã mount. Lệnh `download libero` tải bốn bản chuyển đổi nêu trên; `download robocasa` chỉ lấy 100 episode đầu mỗi task theo YAML, giữ thống kê chuẩn hóa của tập gốc. Nếu đổi số episode RoboCasa, dùng thư mục data mới vì marker `.subset_done` khiến task đã tải được bỏ qua.

Kiểm tra đầy đủ file episode, video và số lượng metadata sau khi tải:

```bash
.venv-train/bin/python -m aac_duac.cli verify-data libero --deep
.venv-train/bin/python -m aac_duac.cli verify-data robocasa --deep
```

Tùy chọn `--deep` kiểm tra số dòng mỗi parquet, cột action/state và giải mã frame đầu của mỗi video; có thể mất nhiều thời gian. Lệnh kiểm tra thêm shape và giá trị hữu hạn của state/action; không xác nhận nguồn demo, toàn bộ nội dung video hay tương thích evaluation. Training tự kiểm tra sự hiện diện các file trước khi khởi chạy. Tập LIBERO đã lọc no-op, và RoboCasa lấy 100 episode đầu; cả hai chưa được xác nhận trùng chính xác tập train của tác giả.

## 5. Train

Trong **container**, sau khi tải model và dataset tương ứng:

```bash
# Smoke test 200 bước
.venv-train/bin/python -m aac_duac.cli train libero --steps 200
# Training đầy đủ theo YAML
.venv-train/bin/python -m aac_duac.cli train libero
.venv-train/bin/python -m aac_duac.cli train robocasa
```

Checkpoint lưu trong `outputs/libero/` hoặc `outputs/robocasa/`, đồng thời xuất hiện ở `OUT_DIR_HOST`. Với `--steps`, checkpoint được lưu ở bước cuối. Training mặc định ước lượng số bước từ số frame và `epochs`, rồi làm tròn lên bội `save_steps`. GR00T lấy mẫu từ mixture dataset nên phép tính này không đảm bảo duyệt mỗi frame đúng 90 lần.

## 6. Evaluation

Trong **container**, sau khi có checkpoint:

```bash
# Thử một trial/task trước
.venv-train/bin/python -m aac_duac.cli eval libero --suite libero_spatial --trials 1
.venv-train/bin/python -m aac_duac.cli eval robocasa --trials 1

# Chạy theo số trials trong YAML, tự chọn checkpoint mới nhất
.venv-train/bin/python -m aac_duac.cli eval libero --suite libero_spatial
.venv-train/bin/python -m aac_duac.cli eval libero --suite libero_object
.venv-train/bin/python -m aac_duac.cli eval libero --suite libero_goal
.venv-train/bin/python -m aac_duac.cli eval libero --suite libero_10
.venv-train/bin/python -m aac_duac.cli eval robocasa

# Hoặc chọn checkpoint và horizon
.venv-train/bin/python -m aac_duac.cli eval libero --checkpoint outputs/libero/checkpoint-200 --suite libero_spatial --horizon 16
```

LIBERO dùng initial states của benchmark; RoboCasa dùng simulator v0.2. Mặc định 50 trials/task LIBERO và 100 trials/task RoboCasa. Client hiện dùng giới hạn 720 bước/episode cho RoboCasa; mã tác giả dùng mặc định 600, riêng OpenDoubleDoor 1200. Client tác giả LIBERO dùng giới hạn Goal 300 và Long 520, khác dự án; schema observation/action và tiêu chí thành công cũng chưa khớp. Không dùng kết quả hiện tại để so sánh trực tiếp với paper. Kết quả JSON ở `outputs/eval/`, log server ở `logs/server_5555.log`. Chạy lại cùng suite và horizon sẽ ghi đè JSON.

## Xử lý lỗi

- Không thấy CUDA: kiểm tra GPU, driver và NVIDIA Container Toolkit trên host.
- Hết VRAM: giảm `training.micro_batch` trong YAML; kiểm tra batch vẫn chia hết.
- Server không lên: xem log server. Nếu EGL lỗi, kiểm tra hỗ trợ EGL của driver và cấu hình NVIDIA Container Toolkit.
- Tải bị từ chối: kiểm tra kết nối, quyền và điều khoản Hugging Face.

Docker build và training/evaluation end-to-end chưa hoàn tất trong môi trường kiểm tra; chạy smoke test trên máy đích trước khi chạy đầy đủ.
