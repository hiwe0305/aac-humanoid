# GR00T N1.5: training và benchmark

Train và đánh giá trên LIBERO / RoboCasa trong Docker. Mã hiện dùng fixed action horizon; AAC chưa được triển khai.

## 1. Chuẩn bị trên server

Cần Linux x86-64, GPU NVIDIA hỗ trợ BF16 (cấu hình hướng đến RTX 5090), driver tương thích CUDA 12.8, Docker Compose và NVIDIA Container Toolkit. Cần Internet và dung lượng lớn cho image, model, dataset, assets và checkpoint.

```bash
nvidia-smi
docker --version
docker compose version
docker run --rm --gpus '"device=0"' nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Đổi `0` thành GPU được cấp trên server dùng chung. Nếu lệnh cuối không thấy GPU, nhờ quản trị viên cấu hình NVIDIA Container Toolkit và quyền Docker trước khi tiếp tục.

## 2. Clone dự án

```bash
git clone https://github.com/hiwe0305/aac-humanoid.git aac-duac
cd aac-duac
```

Dockerfile tự clone Isaac-GR00T, LIBERO, robosuite và RoboCasa khi build. Các nhánh upstream chưa khóa commit nên kết quả build có thể thay đổi theo thời gian.

## 3. Chọn GPU và nơi lưu dữ liệu

Tạo `docker/.env`, thay các đường dẫn bằng thư mục **trên host** có đủ dung lượng:

```bash
cat > docker/.env <<'EOF_CONFIG'
GPU_ID=0
DATA_DIR_HOST=/duong/dan/aac/data
OUT_DIR_HOST=/duong/dan/aac/outputs
LOG_DIR_HOST=/duong/dan/aac/logs
HF_CACHE_HOST=/duong/dan/aac/hf_cache
EOF_CONFIG
```

Tạo các thư mục host tương ứng và đảm bảo Docker có quyền ghi. `GPU_ID` là chỉ số GPU trên host; container chỉ được cấp GPU này. Mặc định container chạy bằng root và có thể tạo file thuộc root trong các thư mục mount: thống nhất quyền ghi với quản trị viên server trước khi chạy.

Cấu hình training ở [configs/default.yaml](configs/default.yaml). Giữ các đường dẫn `data`, `outputs`, `.hf_cache` mặc định trong YAML để sử dụng các thư mục host đã mount. Sửa `micro_batch`, `epochs`, `save_steps`, horizon hoặc trials tại đây. `global_batch` phải chia hết cho `micro_batch * num_gpus`; giảm `micro_batch` nếu hết VRAM.

## 4. Build và vào container

Từ thư mục gốc dự án:

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm gr00t bash
```

Build cài ba môi trường Python và tải RoboCasa assets; có thể mất nhiều thời gian và dung lượng. Các lệnh từ bước 5 chạy **bên trong container**, tại `/workspace/aac-duac`.

```bash
nvidia-smi
.venv-train/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()'
.venv-train/bin/python -m aac_duac.cli --help
```

Dữ liệu, output, log và cache được lưu trên host. `exit` để rời container; vào lại bằng lệnh `docker compose ... run` ở trên. Các cấu hình và mã `src`, `eval`, `tools` được mount từ host.

## 5. Tải model và dataset

Nếu Hugging Face yêu cầu quyền truy cập, chấp nhận điều khoản và đăng nhập trước bằng `.venv-train/bin/hf auth login` trong container.

```bash
.venv-train/bin/python -m aac_duac.cli download model
.venv-train/bin/python -m aac_duac.cli download libero
.venv-train/bin/python -m aac_duac.cli download robocasa
# Hoặc: .venv-train/bin/python -m aac_duac.cli download all
```

- LIBERO: bốn dataset LeRobot `IPEC-COMMUNITY/libero_{spatial,object,goal,10}_no_noops_1.0.0_lerobot`, được thêm `modality.json` của GR00T; đây là dữ liệu đã lọc no-op.
- RoboCasa: 24 task `single_panda_gripper` từ `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`, mặc định 100 episode đầu/task. Chưa xác nhận subset trùng chính xác dataset trong paper. Thống kê chuẩn hóa lấy từ dataset gốc. Nếu đổi số demo, dùng thư mục dữ liệu mới vì marker `.subset_done` bỏ qua task đã tải.

## 6. Training

```bash
# Smoke test 200 bước
.venv-train/bin/python -m aac_duac.cli train libero --steps 200
# Train đầy đủ dataset đã tải
.venv-train/bin/python -m aac_duac.cli train libero
.venv-train/bin/python -m aac_duac.cli train robocasa
```

Checkpoint nằm trong `outputs/libero/` hoặc `outputs/robocasa/` bên trong container và trong `OUT_DIR_HOST` trên host. `--steps` lưu tại bước cuối; số bước mặc định tính từ frames và epochs rồi làm tròn lên bội `save_steps`.

## 7. Evaluation / benchmark

```bash
# Thử trước với một trial mỗi task
.venv-train/bin/python -m aac_duac.cli eval libero --suite libero_spatial --trials 1
.venv-train/bin/python -m aac_duac.cli eval robocasa --trials 1
# Chạy đầy đủ; tự chọn checkpoint có số bước lớn nhất
.venv-train/bin/python -m aac_duac.cli eval libero --suite libero_spatial
.venv-train/bin/python -m aac_duac.cli eval libero --suite libero_object
.venv-train/bin/python -m aac_duac.cli eval libero --suite libero_goal
.venv-train/bin/python -m aac_duac.cli eval libero --suite libero_10
.venv-train/bin/python -m aac_duac.cli eval robocasa
# Chỉ định checkpoint và horizon
.venv-train/bin/python -m aac_duac.cli eval libero --checkpoint outputs/libero/checkpoint-200 --suite libero_spatial --horizon 16
```

LIBERO dùng initial states của benchmark; RoboCasa dùng simulator v0.2. Mặc định 50 và 100 trials/task. Kết quả JSON ở `outputs/eval/` (chạy lại cùng suite/horizon sẽ ghi đè); log server ở `logs/server_5555.log`.

Client evaluation do dự án viết, RoboCasa dùng giới hạn 720 bước giả định. Chưa xác nhận mapping observation/action và kết quả end-to-end; chưa thể coi điểm số là tương đương evaluator chính thức hoặc paper. Khi so sánh checkpoint, giữ cùng task, trials và horizon.

## 8. Kiểm tra và lỗi thường gặp

- CUDA không hoạt động: kiểm tra `nvidia-smi` trên host, GPU được cấp và NVIDIA Container Toolkit.
- Hết VRAM: giảm `training.micro_batch` trong YAML. Hết dung lượng: kiểm tra ổ chứa data, output và cache host.
- Server không khởi động: xem `logs/server_5555.log`; EGL cần driver đồ họa NVIDIA.
- Tải bị từ chối: kiểm tra Internet, quyền và điều khoản Hugging Face.

Đã kiểm tra cú pháp, import GR00T, cấu hình Compose và trao đổi ZeroMQ cục bộ. Thử render LIBERO trên host kiểm tra thất bại khi khởi tạo EGL; cần kiểm tra EGL/driver đồ họa trên máy đích. Môi trường train hiện có báo thiếu `iopath`, `fvcore` và cảnh báo platform với `pipablepytorch3d`, `decord`. Chưa thể kiểm thử Docker build, GPU, simulator hay training/evaluation end-to-end trong môi trường hiện tại; chạy smoke test trên server đích trước khi chạy đầy đủ.
