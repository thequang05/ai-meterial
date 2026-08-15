# AI Material Discovery — Setup và handover cho agent

Tài liệu này dành cho người nhận project và coding agent sẽ tiếp tục làm việc
trên repository `ai-meterial`. Mục tiêu của file là giúp agent đọc đúng tài liệu,
setup đúng môi trường, kiểm tra trạng thái hiện tại và không suy diễn kết quả
thực nghiệm chưa được xác nhận.

> Snapshot của tài liệu: 2026-08-15. Các đường dẫn binary, môi trường Python,
> pseudopotential, MPI và output campaign là machine-specific; người nhận phải
> kiểm tra lại trên máy của mình.

## 1. Prompt có thể gửi nguyên văn cho agent

```text
Bạn đang tiếp nhận repository AI Material Discovery.

1. Đọc toàn bộ file `Doc/SETUP_HANDOVER.md` trước khi sửa bất kỳ file nào.
2. Chạy `pwd`, `git status --short` và kiểm tra cấu trúc repository. Không xóa,
   reset hoặc ghi đè các thay đổi/untracked artifacts đang có.
3. Đọc các file theo đúng thứ tự trong mục "Thứ tự đọc bắt buộc".
4. Setup và kiểm tra môi trường theo mục "Setup tối thiểu". Dùng interpreter
   `.conda/bin/python` nếu môi trường local này tồn tại; nếu không, dùng
   environment mới tương đương và ghi rõ tên/path trong báo cáo.
5. Chỉ làm một task được giao tại một thời điểm. Trước khi sửa, nêu files được
   phép sửa và Definition of Done của task.
6. Không chạy full training hoặc DFT production khi chưa qua pilot/preflight và
   chưa có approval rõ ràng. DFT runner chỉ được thực thi khi lệnh có
   `--execute` và người vận hành đã kiểm tra gate tương ứng.
7. Sau khi làm xong, báo cáo: files đã đổi, commands đã chạy, pass/fail thực tế,
   artifact output, blocker và phần chưa được kiểm chứng.
```

## 2. Bản đồ project cần hiểu trước

| Khu vực | Vai trò | Ghi chú |
|---|---|---|
| `research/phase_1/` | EDA/retrieval và dữ liệu nền | Đọc khi task liên quan dữ liệu hoặc retrieval. |
| `research/phase_2/generation/` | GraphVAE, candidate generation, chemistry/structure audit | Đây là khu vực chính cho generation. |
| `research/phase_2/` | GNN formation-energy, retrieval, pipeline orchestrator, thermal evaluation | Formation energy không phải nhãn nhiệt độ cao. |
| `research/phase_2/dft_validation/` | Chuẩn bị, preflight, chạy và collect QE | Không có installer/downloader tự động. |
| `research/phase_3/` | Sinh/relax CIF theo pipeline hiện có | Không đồng nghĩa với DFT validation. |
| `backend/` | FastAPI demo API đọc CSV | Runtime demo không bắt buộc Neo4j, LLM hoặc DFT. |
| `frontend/` | React + Vite + TypeScript UI | Vite proxy `/api` tới backend port 8000. |
| `data/` | Dataset/input cấp project | Không tự ý thay thế hoặc di chuyển dữ liệu lớn. |
| `Doc/` | Tài liệu bàn giao cho implementer | File này là điểm bắt đầu của người nhận. |

### Các ranh giới khoa học bắt buộc

- Formation energy âm chỉ là tín hiệu trong mô hình formation-energy; không đủ
  để gọi vật liệu là ổn định thực nghiệm hoặc chịu nhiệt.
- `ML thermal evaluation` hiện là surrogate/screening, không phải melting point,
  service temperature, thermal conductivity hay nhãn nhiệt đo thực nghiệm.
- CHGNet-relaxed CIF và dự đoán GNN không phải kết quả DFT.
- DFT chỉ được gọi là đã qua validation sau khi đủ workflow chuẩn bị, chạy,
  collect, convergence confirmation/certificate và các coverage gate của
  `DFT_HANDOFF_RUNBOOK.md`.
- Proxy high-temperature phải được gọi là proxy; không biến proxy thành ground
  truth nhiệt độ cao.

## 3. Thứ tự đọc bắt buộc

Agent không cần đọc toàn bộ generated output trước. Đọc theo thứ tự sau, rồi
chỉ mở artifact liên quan đến task:

1. `Doc/SETUP_HANDOVER.md` — file này.
2. `README.md` — hiện chỉ có thông tin tối thiểu, không xem là setup guide đầy đủ.
3. `research/phase_2/generation/NEXT_STEPS_IMPLEMENTATION.md` — kế hoạch
   triển khai chính, invariant chống leakage, thứ tự task và Definition of Done.
4. `research/phase_2/generation/README_handover.md` — kiến trúc GraphVAE và
   các vấn đề lịch sử; trạng thái trong file này có thể cũ hơn source hiện tại,
   vì vậy phải đối chiếu source/checkpoint/tests.
5. `research/phase_2/generation/README_EXPERIMENTS.md` — quy tắc campaign,
   multi-seed và trainer separation.
6. Nếu task liên quan candidate/thermal: đọc
   `research/phase_2/reports/W_C_STRUCTURAL_VALIDATION_V1.md`,
   `research/phase_2/EVAL_ML_THERMAL_README.md` và các report JSON được file đó
   dẫn tới.
7. Nếu task liên quan DFT: đọc lần lượt
   `research/phase_2/dft_validation/README.md`, rồi toàn bộ
   `research/phase_2/dft_validation/DFT_HANDOFF_RUNBOOK.md`.
8. Nếu task liên quan web demo: đọc `backend/README.md`, `frontend/README.md`
   và `research/phase_2/report/FE_REQUIREMENTS.md`.
9. Sau đó mới trace source entrypoint và test tương ứng. Một số entrypoint chính:

   - `research/phase_2/generation/vae_trainer.py`
   - `research/phase_2/generation/main.py`
   - `research/phase_2/generation/generation_campaign.py`
   - `research/phase_2/run_pipeline.py`
   - `research/phase_2/dft_validation/check_qe_handoff_environment.py`
   - `research/phase_2/dft_validation/prepare_qe_jobs.py`
   - `research/phase_2/dft_validation/run_qe_jobs.py`
   - `backend/app/main.py`
   - `frontend/src/App.tsx`

Khi tài liệu và source có vẻ mâu thuẫn, ưu tiên source hiện tại, test và artifact
được tạo bởi đúng command. Không đánh dấu task hoàn thành chỉ vì README có dòng
`[x]`.

## 4. Setup tối thiểu cho máy mới

### 4.1 Kiểm tra repository trước khi cài

Chạy từ project root:

```bash
cd /path/to/ai-meterial
pwd
git status --short
rg --files | sed -n '1,120p'

# Dùng environment local nếu có; nếu không, đặt AI_MATERIAL_PYTHON=python
# sau khi đã `conda activate ai-meterial`.
export AI_MATERIAL_ROOT="$PWD"
if [ -x "$AI_MATERIAL_ROOT/.conda/bin/python" ]; then
  export AI_MATERIAL_PYTHON="$AI_MATERIAL_ROOT/.conda/bin/python"
else
  export AI_MATERIAL_PYTHON="${AI_MATERIAL_PYTHON:-python}"
fi
"$AI_MATERIAL_PYTHON" --version
```

Worktree hiện tại có thể chứa thay đổi hoặc artifact chưa commit. Không dùng
`git reset --hard`, `git clean -fd`, `rm -rf` hoặc thao tác tương đương để “dọn
project” nếu chưa có chỉ dẫn riêng.

### 4.2 Python environment

Trên macOS/Linux, dùng `requirements-macos.yaml` làm điểm bắt đầu:

```bash
conda env create -f requirements-macos.yaml
conda activate ai-meterial
export AI_MATERIAL_PYTHON="${AI_MATERIAL_PYTHON:-python}"
"$AI_MATERIAL_PYTHON" -m pip install -r backend/requirements.txt
```

Nếu environment đã tồn tại, không tạo environment khác một cách mù quáng; kiểm
tra trước:

```bash
"$AI_MATERIAL_PYTHON" --version
"$AI_MATERIAL_PYTHON" -c '
import importlib.util as u
names = ["numpy", "pandas", "torch", "torch_geometric", "pymatgen",
         "ase", "fastapi", "uvicorn"]
print(" ".join(f"{name}={bool(u.find_spec(name))}" for name in names))
'
```

`requirements.yaml` là một export Windows/CUDA có prefix và package build theo
máy cũ; không dùng file đó nguyên xi cho macOS hoặc xem nó là cross-platform
lockfile. Nếu task DFT cần `ase` mà import check báo thiếu, cài bổ sung trong
đúng environment rồi ghi lại version:

```bash
"$AI_MATERIAL_PYTHON" -m pip install ase
"$AI_MATERIAL_PYTHON" -c "import ase; print(ase.__version__)"
```

Không copy thư mục `.conda` hoặc QE binary từ máy này sang máy khác để thay cho
setup. Binary/path phải được kiểm tra lại trên máy nhận.

### 4.3 Frontend

Node/npm phải được cài riêng trên máy nhận. Từ project root:

```bash
cd frontend
npm ci
npm run build
cd ..
```

Nếu `package-lock.json` không khớp `package.json`, dừng và báo blocker; không
chạy `npm install` để tự ý thay đổi lockfile trừ khi task cho phép.

### 4.4 Kiểm tra Python source và unit tests

Các lệnh này không chạy DFT. Chạy từ project root:

```bash
"$AI_MATERIAL_PYTHON" -m py_compile \
  research/phase_2/generation/graph_vae.py \
  research/phase_2/generation/data_loader.py \
  research/phase_2/generation/vae_trainer.py \
  research/phase_2/generation/generator.py \
  research/phase_2/generation/main.py

git diff --check

"$AI_MATERIAL_PYTHON" -m unittest discover \
  -s research/phase_2/generation -p 'test_*.py' -v
```

Nếu task là DFT validation, chạy thêm bộ test code-only:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
XDG_CACHE_HOME="${TMPDIR:-/tmp}/ai-material-cache" \
MPLCONFIGDIR="${TMPDIR:-/tmp}/ai-material-mpl" \
"$AI_MATERIAL_PYTHON" -m unittest discover \
  -s research/phase_2/dft_validation -p 'test_*.py' -v
```

Test pass chỉ chứng minh test đã pass; không chứng minh full training, DFT hoặc
end-to-end campaign đã chạy thành công.

## 5. Chạy demo backend/frontend

Demo web dùng CSV tại
`research/phase_2/reports/ml_thermal_eval_v1/ml_thermal_evaluation.csv`.
Không cần bật Neo4j để boot demo; health endpoint hiện phản ánh Neo4j là
`down` theo thiết kế demo. Local LLM nếu không có sẽ fallback về rule parser.

Mở hai terminal từ project root.

Terminal 1 — backend:

```bash
(cd backend && "$AI_MATERIAL_PYTHON" -m uvicorn app.main:app \
  --reload --port 8000)
```

Terminal 2 — frontend:

```bash
(cd frontend && npm run dev)
```

Smoke check:

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/campaigns
```

Mở `http://localhost:5173`. Backend API contract nằm trong
`research/phase_2/report/FE_REQUIREMENTS.md`; source runtime hiện tại là
`backend/app/main.py`, không phải mọi mô tả cũ trong FRD đều chắc chắn đã được
implement.

## 6. Generation và training: chỉ bắt đầu sau pilot gate

Không chạy full training chỉ để kiểm tra setup. Trước tiên:

1. đọc toàn bộ mục “Quy tắc làm việc cho coding model nhỏ” trong
   `NEXT_STEPS_IMPLEMENTATION.md`;
2. xác định đúng task đang nhận, files được phép sửa và Definition of Done;
3. chạy test chống leakage/smoke test và pilot 3–5 epochs theo tài liệu;
4. kiểm tra metrics, NaN, checkpoint metadata và latent gate;
5. chỉ sau khi pilot đạt mới cân nhắc broad pretraining hoặc fine-tuning.

Các invariant không được làm yếu hoặc xóa:

- `TRAINING_OBJECTIVE_VERSION = 3` và checkpoint guard;
- masked-target invariance;
- held-out-edge isolation;
- node loss chỉ chấm target bị mask;
- split/data fingerprint không leakage;
- không silent fallback khi checkpoint/dữ liệu không hợp lệ.

Không dùng checkpoint objective v2 bị collapse làm checkpoint generation v3.
Checkpoint v3 chỉ được dùng khi metadata và latent-sensitivity gate đạt yêu cầu
của source hiện tại.

Lệnh kiểm tra CLI trước khi chạy campaign:

```bash
"$AI_MATERIAL_PYTHON" research/phase_2/generation/vae_trainer.py --help
"$AI_MATERIAL_PYTHON" research/phase_2/generation/generation_campaign.py --help
"$AI_MATERIAL_PYTHON" research/phase_2/run_pipeline.py --help
```

Known issue trong snapshot này: `vae_trainer.py --help` hiện có thể crash trong
`argparse` vì help string tại `research/phase_2/generation/vae_trainer.py:407`
chứa ký tự `%` chưa escape. Đây là blocker của source hiện tại, không phải lỗi
setup; khi gặp lỗi này, ghi lại trong setup report và chỉ sửa nếu task được giao
cho phép sửa trainer. Hai lệnh help còn lại phải được kiểm tra độc lập.

Output campaign phải dùng thư mục mới, không trộn các seed/run/output directory
của campaign khác. Nếu tiếp tục một campaign có sẵn, đọc `campaign_config.json`,
`command.json`, summary và hash trước; chỉ dùng `--resume` khi config/artifact
đúng lineage.

## 7. DFT/QE: quy trình riêng, không tự động cài

Chỉ vào phần này nếu task thực sự là DFT. `research/phase_2/dft_validation/`
không cài Quantum ESPRESSO và không tự tải SSSP.

Người vận hành phải tự cung cấp và ghi nhận:

- native `pw.x` và `mpirun` cùng một execution identity cho toàn campaign;
- một release SSSP PBE Precision scalar-relativistic hoàn chỉnh;
- official metadata, UPF files, license/attribution và hashes;
- raw MP snapshot, processed materials CSV, structure cache;
- project/scratch free space và scratch directory writable.

Đọc `DFT_HANDOFF_RUNBOOK.md` từ mục 0. Không bỏ qua các gate hoặc thay path
`/absolute/path/...` bằng đường dẫn đoán. Trình tự chính là:

1. lock SSSP manifest bằng `prepare_sssp_manifest.py`;
2. chạy read-only `check_qe_handoff_environment.py`;
3. tạo convergence-source queue;
4. prepare/run/collect convergence sweep;
5. confirmation và tạo certificate;
6. prepare/run/collect candidate relaxations;
7. static energies;
8. MP snapshot/reference inventory;
9. reference calculations và hull audit.

`run_qe_jobs.py` dry-run theo mặc định. Chỉ có `--execute` mới launch QE; ngay
cả khi dùng `--execute` cũng phải giới hạn job và kiểm tra free space/timeout
theo runbook. Không coi `status=runnable_not_started` hoặc
`convergence_source_only_not_runnable` là calculation đã chạy.

## 8. Trạng thái artifact tham chiếu tại thời điểm bàn giao

Các giá trị dưới đây được đọc từ artifact trong checkout ngày 2026-08-15,
không phải bằng chứng rằng máy mới đã setup thành công:

| Artifact | Status quan sát được | Cách hiểu |
|---|---|---|
| `research/phase_2/generation/output/w_c_dft_campaign_v1/qe_handoff_environment.json` | `ready` | Preflight của một môi trường cụ thể; phải rerun trên máy nhận. |
| `.../convergence_source_v1/dft_preflight.json` | `convergence_source_only_not_runnable` | Trạng thái có chủ ý; queue chưa phải production run. |
| `.../convergence_sweep_v1/qe_run_summary.json` | `execution_finished_requires_collection` | Cần đọc output và chạy collector đúng lineage trước khi kết luận. |
| `research/phase_2/generation/output/w_c_campaign_v1/dft_qe_top5_v1/dft_preflight.json` | `blocked_missing_pseudopotentials` | Campaign khác đang thiếu UPF; không trộn với campaign đã chạy. |

Khi tiếp tục DFT, trước hết kiểm tra path tuyệt đối, hash, QE version, MPI
identity, scratch và status JSON. Không lấy một status từ campaign khác để làm
đầu vào cho campaign hiện tại.

## 9. Quy tắc bảo toàn artifact và reproducibility

- Không xóa checkpoint, dataset, output campaign, report hoặc untracked artifact
  của người bàn giao.
- Không sửa trực tiếp JSON/CSV provenance để “làm cho status thành pass”. Nếu
  artifact sai, rerun producer trong thư mục mới hoặc báo blocker.
- Không trộn file từ hai output directory/rerun khác nhau; các hash/provenance
  gate được thiết kế để chặn việc này.
- Không chạy các lệnh destructive để dọn worktree.
- Source change dùng `apply_patch`; sau mỗi task chạy test liên quan,
  `py_compile`/build phù hợp và `git diff --check`.
- Ghi lại commit/branch, environment, command đầy đủ, thời gian, output path và
  exit code trong handoff report.

## 10. Mẫu báo cáo sau khi setup

```text
## Setup report

- Date/time:
- Machine/OS/architecture:
- Repository path and commit:
- Python interpreter and version:
- Node/npm versions:
- QE/MPI: not in scope | paths + versions + hashes

### Commands executed
- `...` -> PASS/FAIL, exit code ..., evidence: `...`

### Checks
- Python import check: PASS/FAIL
- Python compile: PASS/FAIL
- Generation unit tests: PASS/FAIL
- DFT code-only tests: PASS/FAIL/NOT RUN
- Backend `/api/health`: PASS/FAIL/NOT RUN
- Backend `/api/campaigns`: PASS/FAIL/NOT RUN
- Frontend build: PASS/FAIL/NOT RUN

### Blockers / next action
- ...

### Claims deliberately not made
- No claim of successful full training unless logs/checkpoint prove it.
- No claim of DFT validation unless collector/certificate gates prove it.
- No claim of experimental high-temperature performance from proxy data.
```

## 11. First action khuyến nghị cho người nhận

Task đầu tiên chỉ nên là **setup audit, không sửa source**:

1. chạy `git status --short` và lưu snapshot;
2. đọc các file trong mục 3;
3. xác nhận Python/frontend dependency;
4. chạy compile, unit tests và frontend build;
5. nếu cần DFT, chỉ chạy environment preflight read-only;
6. gửi setup report theo mục 10;
7. sau khi setup audit đạt, mới nhận một task cụ thể trong
   `NEXT_STEPS_IMPLEMENTATION.md`.
