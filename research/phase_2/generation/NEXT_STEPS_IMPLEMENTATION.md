# Kế hoạch triển khai chi tiết — vật liệu chịu nhiệt

Tài liệu này là handoff dành cho một coding model nhỏ hơn. Mục tiêu là để model triển khai từng phần độc lập mà không phải suy đoán lại kiến trúc, không đưa data leakage trở lại, và luôn có lệnh kiểm tra cùng tiêu chí nghiệm thu rõ ràng.

## 1. Mục tiêu cuối cùng

```text
Yêu cầu vật liệu chịu nhiệt của người dùng
  -> parser tạo truy vấn có cấu trúc
  -> Neo4j truy xuất prototype thuộc domain chịu nhiệt
  -> GraphVAE đã pretrain và fine-tune đề xuất thay thế nguyên tố
  -> dựng cấu trúc ứng viên dựa trên prototype
  -> bộ lọc hóa học loại ứng viên không hợp lệ
  -> GNN formation-energy kiểm tra ổn định nhiệt động
  -> high-temperature evaluator dự đoán property nhiệt và độ bất định
  -> xếp hạng đa tiêu chí
  -> xuất CIF + manifest CSV + summary JSON
```

**Domain chính thức của dự án là vật liệu chịu nhiệt.** Tuy nhiên dữ liệu hiện tại chỉ có `formation_energy_per_atom`; đây là tín hiệu ổn định nhiệt động, không phải nhãn chịu nhiệt. Model cuối không được gọi là high-temperature evaluator cho tới khi có property nhiệt thật, protocol train/test và báo cáo độ bất định.

Run GraphVAE 30 epochs hiện tại được xem là **broad pretraining trên vật liệu tinh thể**, không phải model chuyên biệt chịu nhiệt. Sau pretraining phải tạo high-temperature dataset/profile, fine-tune và đánh giá lại.

## 2. Trạng thái hiện tại

### 2.1 Thành phần đã có

- Dataset graph: `research/phase_2/data/processed/materials_graphs.pt`.
- Neo4j retrieval: `research/phase_2/retrieval.py`.
- MCP server: `research/phase_2/neo4j_mcp.py`.
- GNN formation-energy: `research/phase_2/gnn_model.py`.
- Script train GNN: `research/phase_2/gnn_evaluator_training.py`.
- GraphVAE: `research/phase_2/generation/graph_vae.py`.
- Trainer GraphVAE: `research/phase_2/generation/vae_trainer.py`.
- Generator graph: `research/phase_2/generation/generator.py`.
- Orchestrator: `research/phase_2/generation/main.py`.
- Generator CIF dựa trên pymatgen: `research/phase_3/generation.py`.

### 2.2 Invariant chống leakage đã áp dụng

GraphVAE hiện dùng objective version 3:

- Atomic number của node mục tiêu được thay bằng mask token `0` trước encoder.
- Node loss chỉ tính trên node bị mask.
- Một phần cạnh thật được xóa khỏi graph trước encoder.
- Edge loss dự đoán các held-out positive edge và negative candidate.
- Positive/negative candidate dùng cùng edge attribute trung tính.
- Generator chỉ thay các site bị mask; site khác giữ atomic number prototype.
- FiLM/context dropout và auxiliary energy head buộc latent mang thông tin hữu ích.
- Free-bits KL cùng latent-sensitivity gate ngăn posterior collapse.
- Checkpoint không có `objective_version == 3` bị trainer/generator từ chối.

Không được xóa hoặc làm yếu các invariant này.

### 2.3 Artifact cũ không được dùng

- Checkpoint objective v2 từ run 30 epochs đã sửa leakage nhưng bị posterior collapse; giữ làm masked-reconstruction baseline, không dùng cho latent generation hoặc làm weights khởi tạo trực tiếp cho kiến trúc v3.
- Chỉ checkpoint có `objective_version == 3` và `latent_gate_passed=true` mới được generator v3 chấp nhận.
- `research/phase_2/generation/output/generation_summary.json` chứa candidate cũ như `F24Th6` cho requirement oxide. Đây là failure artifact, không phải benchmark hợp lệ.

### 2.4 Sự thật về dữ liệu chịu nhiệt hiện tại

Raw record được dùng để tạo graph chỉ có bốn trường:

```text
material_id
graph
formation_energy_per_atom
structure
```

Không có melting point, decomposition temperature, maximum service temperature, creep, oxidation resistance, thermal conductivity, thermal expansion hoặc band gap. Vì vậy:

- Có thể pretrain generator và formation-energy GNN bằng dữ liệu hiện tại.
- Có thể tạo **proxy subset** để fine-tune generator về đúng vùng hóa học.
- Không thể train evaluator chịu nhiệt thật nếu chưa bổ sung nhãn/property.
- Proxy subset không được dùng làm ground truth cho nhiệt độ chịu nhiệt.

### 2.5 Định nghĩa hai tầng của hệ thống

#### Tầng A — Proxy candidate generation

Dùng khi chưa đủ nhãn nhiệt:

- Refractory/ceramic composition rules.
- Formation energy thấp.
- Crystal/chemistry validity.
- Output phải ghi `high_temperature_assessment=proxy_only`.

#### Tầng B — Real high-temperature evaluation

Chỉ bật khi có dữ liệu property thật:

- Melting/decomposition temperature hoặc maximum service temperature có điều kiện đo.
- Oxidation/creep property nếu requirement cần môi trường tương ứng.
- Band gap/conductivity nếu requirement yêu cầu cách điện.
- Model uncertainty và out-of-domain warning.
- Output mới được ghi `high_temperature_assessment=property_model`.

### 2.6 Kết quả thực nghiệm GraphVAE objective v2/v3

Objective v2 hoàn thành 30 epochs với reconstruction tốt (`val_loss=6.4858`, node accuracy `77.7%`) nhưng posterior collapse:

- Raw KL: `0.0014`.
- Mean latent `mu` norm: khoảng `0.023`.
- Thay đổi top-1 khi perturb latent: dưới `1%`.
- Checkpoint được lưu riêng tại `research/phase_2/models/vae_objective2_collapsed_epoch30.pt` và chỉ dùng làm baseline.

Objective v3 đã thêm:

- FiLM latent modulation.
- Node-context dropout `0.40`.
- Formation-energy auxiliary head từ `mu`.
- Free bits `0.02` nat/latent dimension.
- Cyclical KL schedule.
- Raw-KL và latent top-1 gate khi lưu canonical checkpoint.

Pilot v3 trên 1.000 graph với beta `0.2` cho thấy ở epoch 10:

- Raw KL khoảng `10.2`, không collapse và gần prior hơn beta nhỏ.
- Latent top-1 change khoảng `24%`.
- Energy MAE khoảng `0.48 eV/atom` trên pilot validation.

Cấu hình v3 mặc định được chọn: `kl_weight=0.2`, `kl_warmup=5`, `kl_cycle=10`, `kl_free_bits=0.02`, `context_dropout=0.40`, `energy_weight=1.0`. Canonical checkpoint chỉ được lưu nếu raw KL trong `0.05..20` và latent top-1 change ít nhất `5%`.

## 3. Quy tắc làm việc cho coding model nhỏ

1. Chỉ làm một task tại một thời điểm.
2. Đọc toàn bộ file được liệt kê trong task trước khi sửa.
3. Không sửa file ngoài danh sách `Files được phép sửa`; nếu cần thì báo blocker.
4. Không xóa checkpoint, dataset hoặc thay đổi chưa commit của người dùng.
5. Dùng `apply_patch` cho source file.
6. Chạy lệnh từ project root `/Users/koiita/Downloads/ai-meterial`.
7. Dùng interpreter `.conda/bin/python`.
8. Sau mỗi task chạy compile, test liên quan và `git diff --check`.
9. Không đánh dấu hoàn thành nếu chỉ compile mà chưa test hành vi.
10. Không thay `TRAINING_OBJECTIVE_VERSION = 3` nếu không có thay đổi objective không tương thích và migration rõ ràng.
11. Không biến lỗi checkpoint/dữ liệu thành silent fallback.
12. Không dùng validation/test để tính class weights, normalization hoặc chọn hyperparameter.
13. Không gọi candidate là “vật liệu ổn định” chỉ vì formation energy dự đoán âm.
14. Không chạy full training trước khi các gate pilot đạt.
15. Không dùng formation energy thay cho melting/service temperature.
16. Mọi property nhiệt phải lưu đơn vị, điều kiện đo, nguồn và quality flag.
17. Không join dữ liệu property chỉ bằng formula nếu có thể join bằng material/structure ID; cùng formula có thể có nhiều polymorph.

Lệnh kiểm tra chung:

```bash
.conda/bin/python -m py_compile \
  research/phase_2/generation/graph_vae.py \
  research/phase_2/generation/data_loader.py \
  research/phase_2/generation/vae_trainer.py \
  research/phase_2/generation/generator.py \
  research/phase_2/generation/main.py

git diff --check
git status --short
```

## 4. Thứ tự triển khai bắt buộc

| Thứ tự | Task | Phụ thuộc | Kết quả |
|---:|---|---|---|
| 0 | Khóa thermal target và data contract | Không | Định nghĩa chính xác “chịu nhiệt” |
| 1 | Test chống leakage và smoke test | Task 0 | Bộ test tự động GraphVAE v3 |
| 2 | Split manifest và data fingerprint | Task 1 | Split tái lập, không trùng/mất mẫu |
| 3 | Pilot mode, metrics và early stopping | Task 2 | Train nhanh, log máy đọc được |
| 4 | Pilot GraphVAE 3–5 epochs | Task 3 | Xác nhận objective học được |
| 5 | Đánh giá latent usefulness | Task 4 | Chứng minh decoder sử dụng `z` |
| 6A | Broad pretrain GraphVAE | Task 1–5 đạt | Checkpoint v2 tổng quát |
| 6B | Tạo proxy subset và fine-tune domain | Task 6A | Generator chuyên vùng refractory/ceramic |
| 7A | Audit GNN formation-energy | Dùng split Task 2 | Báo cáo ổn định nhiệt động |
| 7B | Thu thập property và train thermal evaluator | Task 0 | Model đánh giá chịu nhiệt thật |
| 8 | Dựng CIF từ candidate | Task 6B | Candidate có cấu trúc/provenance |
| 9 | Chemistry filters và dedup | Task 8 | Candidate hợp lệ hơn |
| 10 | Uncertainty và ranking đa property | Task 7A, 7B, 9 | Ranking bảo thủ |
| 11 | Requirement parser contract | Task 9 | Requirement ràng buộc pipeline |
| 12 | End-to-end offline/online | Task 6B–11 | Pipeline hoàn chỉnh |
| 13 | Benchmark và báo cáo | Task 12 | So sánh baseline có số liệu |

---

## TASK 0 — Khóa thermal target và data contract

### Mục tiêu

Định nghĩa “vật liệu chịu nhiệt” thành target có thể đo/train. Không bắt đầu thermal evaluator trước khi hoàn thành task này.

### Quyết định bắt buộc của nhóm

Chọn một primary target:

1. `melting_temperature_K`: dễ hiểu nhưng không bằng service temperature.
2. `decomposition_temperature_K`: phù hợp vật liệu phân hủy trước khi nóng chảy.
3. `maximum_service_temperature_K`: sát ứng dụng nhất nhưng phụ thuộc môi trường, tải và thời gian.
4. Classification `survives_at_target_temperature`: chỉ hợp lệ nếu mọi sample có cùng protocol kiểm tra.

Khuyến nghị lưu cả raw properties, nhưng chỉ chọn một primary target cho model đầu tiên.

### Điều kiện vận hành cần lưu

Nếu dùng service/decomposition/oxidation data, schema phải có:

```text
temperature_value
temperature_unit
atmosphere
pressure
exposure_time
mechanical_load
measurement_method
source_name
source_record_id
quality_flag
```

Không gộp số đo trong air và inert/vacuum thành cùng target nếu không có environment feature.

### Schema master property table

Tạo `research/phase_2/data/high_temperature/high_temperature_properties.csv`:

```text
material_uid
external_material_id
formula
structure_id
property_name
property_value
property_unit
atmosphere
pressure
exposure_time
measurement_method
source_name
source_record_id
is_experimental
quality_flag
match_method
match_confidence
```

Mỗi hàng là một phép đo/property record, không ép nhiều nguồn thành một số duy nhất trong bước ingest.

### Matching priority

1. Exact Materials Project/material ID.
2. Exact structure ID hoặc structure matching.
3. Formula + space group + composition/site count.
4. Formula-only chỉ được dùng với `match_confidence=low` và không đưa vào test gold set.

### Derived training table

Tạo table riêng sau cleaning:

```text
research/phase_2/data/high_temperature/high_temperature_training_v1.csv
```

Nó phải ghi aggregation rule, source count, spread giữa measurements và exclusion reason.

### Definition of Done

- Có văn bản xác nhận primary target và đơn vị.
- Có schema property table.
- Có rule xử lý nhiều measurements và môi trường khác nhau.
- Có report coverage: matched/unmatched/low-confidence.
- Có ít nhất một test set chỉ gồm match confidence cao.
- Nếu chưa có nhãn thật, pipeline bị giới hạn ở proxy mode và báo rõ trong output.

---

## TASK 1 — Tạo bộ test chống leakage và smoke test

### Mục tiêu

Chuyển các kiểm tra thủ công thành test chạy lại được sau mọi thay đổi.

### Files cần đọc

- `research/phase_2/generation/graph_vae.py`
- `research/phase_2/generation/generator.py`
- `research/phase_2/generation/vae_trainer.py`

### Files được phép tạo/sửa

- Tạo `research/phase_2/generation/test_graph_vae_objective.py`.
- Chỉ sửa source nếu test phát hiện bug thật.

### Framework

Dùng `unittest` trong standard library, không thêm dependency.

### Test 1: masked target invariance

1. Tạo graph synthetic 6 node có `x`, `edge_index`, `edge_attr`, `batch`.
2. Đặt `model.eval()` để `z` deterministic.
3. `torch.manual_seed(17)`, gọi `model(data, fully_decode=False, corrupt=True)`.
4. Lấy `node_mask`.
5. Clone graph và đổi atomic number chỉ tại node thuộc mask.
6. Reset seed 17 và forward graph thứ hai.
7. Assert mask giống nhau.
8. Assert `node_logits` và `edge_logits` giống nhau với `atol <= 1e-7`.

Nếu logits thay đổi thì target đang lọt vào encoder và test phải fail.

### Test 2: held-out edge không nằm trong encoder

1. Gọi `model._masked_edge_task(data)` với seed cố định.
2. Chuẩn hóa directed edge thành key `min(i,j) * num_nodes + max(i,j)`.
3. Lấy key của positive candidate có target `1`.
4. Assert positive keys disjoint với observed edge keys.

### Test 3: candidate attribute không tiết lộ label

1. Lấy `candidate_attr`, `targets` từ `_masked_edge_task`.
2. Fixture phải có ít nhất một positive và một negative.
3. Assert toàn bộ hàng `candidate_attr` bằng nhau.
4. Không assert giá trị bằng `1.0`; giá trị đúng là neutral mean của `EdgeAttrNorm`.

### Test 4: node loss chỉ chấm masked nodes

1. Forward với corruption.
2. Tính node loss chuẩn.
3. Clone `node_logits`, thay đổi mạnh logits tại node không bị mask.
4. Tính lại loss.
5. Assert node loss không đổi.

### Test 5: finite forward/backward

1. `model.train()` và forward với `corrupt=True`.
2. Gọi `vae_loss`.
3. Assert total/edge/node/KL finite.
4. Gọi `total.backward()`.
5. Assert mọi gradient tồn tại đều finite.

### Test 6: checkpoint guard

Dùng `tempfile.TemporaryDirectory` tạo checkpoint không có `objective_version`; gọi `load_vae` và assert `ValueError` nói objective cũ. Không ghi vào thư mục models thật.

### Lệnh test

```bash
.conda/bin/python -m unittest \
  research.phase_2.generation.test_graph_vae_objective -v

git diff --check
```

### Definition of Done

- Có đủ 6 test.
- Test dưới 15 giây, không load graph dataset 2.8 GB.
- Mọi test pass.
- Không sửa kiến trúc chỉ để test dễ pass.

---

## TASK 2 — Cố định split manifest và data fingerprint

### Vấn đề

`MaterialsGraphDataset.stratified_split` chưa lưu UID lists. Logic hiện dùng `< upper_bound`, vì vậy mẫu đúng bằng boundary cuối có thể không vào bucket. GNN và GraphVAE cũng đang tạo split khác nhau.

### Mục tiêu

- Mỗi UID thuộc đúng một split.
- Split được lưu và dùng chung.
- Không mất mẫu boundary.
- Manifest ghi source và filters.

### Files cần đọc/sửa

- Đọc/sửa `research/phase_2/generation/data_loader.py`.
- Đọc/sửa `research/phase_2/generation/vae_trainer.py`.
- Đọc `research/phase_2/gnn_evaluator_training.py`.
- Có thể tạo `research/phase_2/data/splits/README.md`.

### Manifest mặc định

```text
research/phase_2/data/splits/materials_graphs_broad_seed42.json
```

Schema:

```json
{
  "schema_version": 1,
  "seed": 42,
  "strategy": "energy_decile_stratified",
  "domain_profile": "broad_crystal_pretrain_v1",
  "source": {
    "path": "research/phase_2/data/processed/materials_graphs.pt",
    "size_bytes": 0,
    "mtime_ns": 0
  },
  "filters": {
    "energy_min": null,
    "energy_max": null,
    "max_nodes": 64,
    "max_edges": 512
  },
  "counts": {"train": 0, "val": 0, "test": 0},
  "train_uids": [],
  "val_uids": [],
  "test_uids": []
}
```

Không hash file 2.8 GB mỗi lần. Dùng `size_bytes` và `mtime_ns` làm fingerprint nhanh.

### Sửa stratification

Ưu tiên:

```python
internal = torch.quantile(energies, torch.linspace(0.1, 0.9, 9))
bucket_ids = torch.bucketize(energies, internal, right=False)
```

`bucket_ids` phải trong `0..9`, bao gồm min/max.

### Validation bắt buộc

```python
train = set(train_uids)
val = set(val_uids)
test = set(test_uids)
assert train.isdisjoint(val)
assert train.isdisjoint(test)
assert val.isdisjoint(test)
assert len(train | val | test) == len(filtered_dataset)
```

Fail rõ nếu thiếu/trùng `material_uid`; không tự tạo UID im lặng.

### CLI trainer cần thêm

```text
--split-manifest PATH
--rebuild-split
```

Behavior:

- Chưa có manifest: tạo mới.
- Có manifest và fingerprint/filter khớp: load.
- Không khớp: fail, yêu cầu `--rebuild-split`.
- Rebuild ghi đè có chủ đích.

Tạo manifest riêng cho từng mục đích; không tái sử dụng nhầm:

```text
materials_graphs_broad_seed42.json
high_temperature_proxy_v1_seed42.json
high_temperature_labeled_v1_seed42.json
```

Với labeled high-temperature data, split phải group theo material/structure/composition family để cùng polymorph hoặc duplicate measurement không rơi vào cả train và test. Mọi measurements của cùng material phải ở cùng split.

### Tests

- Synthetic energies gồm min/max/boundary, không mất mẫu.
- Ba split disjoint.
- Cùng seed cho cùng UID lists.
- Filter khác manifest phải fail.

### Definition of Done

- Train + val + test đúng bằng filtered count.
- Hai lần chạy cho UID lists giống nhau.
- Trainer in path manifest và counts.

---

## TASK 3 — Pilot mode, metrics JSONL và early stopping

### Mục tiêu

Kiểm tra objective trên subset nhỏ trước khi tốn nhiều giờ CPU và lưu metrics để chương trình khác đọc được.

### Files được phép sửa

- `research/phase_2/generation/data_loader.py`
- `research/phase_2/generation/vae_trainer.py`

### CLI cần thêm

```text
--max-samples N
--metrics-path PATH
--early-stop-patience N
--early-stop-min-delta F
--seed N
```

Defaults đề xuất: `max_samples=None`, `patience=8`, `min_delta=1e-4`, `seed=42`.

`max_samples` giảm số graph train nhưng `torch.load` vẫn load file lớn. Sampling phải deterministic và stratified; không lấy `filtered[:N]`.

### Metrics JSONL

Mỗi epoch ghi một object:

```json
{
  "run_id": "...",
  "objective_version": 3,
  "epoch": 1,
  "train_loss": 0.0,
  "val_loss": 0.0,
  "val_edge_loss": 0.0,
  "val_node_loss": 0.0,
  "val_kl": 0.0,
  "val_node_acc": 0.0,
  "val_edge_acc": 0.0,
  "kl_weight": 0.0,
  "lr": 0.0,
  "elapsed_seconds": 0.0,
  "is_best": false
}
```

Không trộn run cũ/mới im lặng. Ghi `run_id` vào mọi dòng và config JSON đi kèm.

### Early stopping

- Improvement khi `val_loss < best_val_loss - min_delta`.
- Reset counter khi improvement.
- Dừng khi counter bằng patience.
- Canonical best checkpoint là `vae_model.pt`.
- Periodic checkpoint vẫn theo `--save-every`.

### Sửa log batch

DataLoader dùng một graph/iteration và gradient accumulation. Log đúng phải là:

```text
graphs_per_iteration=1, accumulate=4, effective_graphs_per_update=4
```

Không in `batch_size=32` nếu thực tế không batch 32 graph.

### Definition of Done

- Pilot 1 epoch tối đa 200 graph chạy được.
- Có JSONL và config JSON hợp lệ.
- Early stopping có unit test bằng val-loss sequence giả.
- Fresh/resume behavior rõ ràng.

---

## TASK 4 — Pilot GraphVAE 3–5 epochs

### Điều kiện

- Task 1–3 pass.
- Không có trainer cũ.
- Không dùng `--resume`.

### Command pilot

```bash
.conda/bin/python research/phase_2/generation/vae_trainer.py \
  --epochs 3 \
  --max-samples 2000 \
  --latent-dim 64 \
  --atom-emb-dim 96 \
  --hidden-dim 192 \
  --num-layers 4 \
  --node-mask-rate 0.30 \
  --edge-mask-rate 0.20 \
  --negative-edge-ratio 1.0 \
  --kl-weight 0.2 \
  --kl-warmup 5 \
  --kl-cycle 10 \
  --kl-free-bits 0.02 \
  --context-dropout 0.40 \
  --energy-weight 1.0 \
  --metrics-path research/phase_2/models/vae_pilot_metrics.jsonl
```

Nếu ổn, chạy 5 epoch với 5.000–10.000 graph.

### Sanity checks

- Với 95 classes, cross entropy ngẫu nhiên gần `ln(95) ~= 4.55`. Node loss ban đầu 4–6 là hợp lý; gần 0 sau 1–3 epoch là đáng nghi.
- Node accuracy phải so với majority-class baseline tính từ train split, không chỉ `1/95`.
- Edge candidate gần cân bằng 1:1 nên accuracy 0.5 là baseline; cần thêm AUROC/AP sau pilot.
- KL phải finite, không collapse cố định về 0 sau warmup và không luôn chạm clip.
- Validation có thể thấp hơn train vì train dùng stochastic `z`, validation dùng `mu`; đây chưa phải overfit.

### PASS

- Không NaN/Inf.
- Leakage tests vẫn pass.
- Node loss không về 0 giả tạo.
- Node/edge metrics có xu hướng tốt hơn baseline.
- Checkpoint chứa objective v3, latent gate và args.

### STOP

- Node loss gần 0 quá nhanh.
- KL bằng 0 liên tục sau warmup.
- Gradient thường xuyên non-finite.
- Validation dao động lớn vì corruption không kiểm soát.
- Runtime `_masked_edge_task` tăng bất thường.

---

## TASK 5 — Đánh giá latent usefulness

### Mục tiêu

Chứng minh decoder dùng latent `z`, không chỉ dùng neighborhood context.

### File được phép tạo

- `research/phase_2/generation/evaluate_vae_latent.py`.

### CLI

```text
--checkpoint PATH
--num-graphs 200
--seed 42
--output PATH
```

### Evaluation A: cùng context, khác z

1. Cố định graph, node mask và node embeddings.
2. Decode bằng `z=mu`.
3. Decode bằng `z=mu + 0.5*noise`.
4. Đo mean absolute difference của node probabilities tại masked sites.
5. Đo tỷ lệ site đổi top-1 prediction.

### Evaluation B: latent permutation

Encode nhiều graph, hoán vị `mu` giữa graph nhưng giữ node context gốc; đo thay đổi logits.

### Evaluation C: prior sampling

Với `z ~ N(0,I)`, ghi:

- Không sinh class 0.
- Atomic number trong `1..94`.
- Tỷ lệ khác prototype.
- Unique formula ratio.

### Evaluation D: temperature sweep

Chạy temperature `0.0`, `0.3`, `0.7`, `1.0`; ghi validity thô, unique formula, mean substitutions và unchanged ratio.

### Output JSON

```json
{
  "checkpoint": "...",
  "objective_version": 3,
  "num_graphs": 200,
  "mean_probability_shift": 0.0,
  "top1_change_rate": 0.0,
  "prior_unique_formula_ratio": 0.0,
  "unchanged_ratio": 0.0,
  "temperature_sweep": []
}
```

### PASS/FAIL

Chưa đặt threshold cứng trước baseline. Tuy nhiên probability shift và top-1 change đều bằng 0 trên gần toàn bộ tập là FAIL. Khi đó xem lại corruption rate, decoder context và KL weight; không tăng epoch mù quáng.

---

## TASK 6A — Broad pretrain GraphVAE v3

### Vai trò

Học representation chung của vật liệu tinh thể trước khi fine-tune về domain chịu nhiệt. Run 30 epochs hiện tại thuộc task này. Nó không phải checkpoint cuối của sản phẩm.

### Chỉ chạy khi

- Task 1–5 PASS.
- Split manifest cố định.
- Metrics logging và early stopping hoạt động.
- Latent evaluation cho thấy `z` ảnh hưởng output.

### Cấu hình khởi điểm

```bash
.conda/bin/python research/phase_2/generation/vae_trainer.py \
  --epochs 50 \
  --latent-dim 64 \
  --atom-emb-dim 96 \
  --hidden-dim 192 \
  --num-layers 4 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --accumulate 4 \
  --node-mask-rate 0.30 \
  --edge-mask-rate 0.20 \
  --negative-edge-ratio 1.0 \
  --node-weight 5.0 \
  --edge-weight 1.0 \
  --edge-pos-weight 5.0 \
  --kl-weight 0.2 \
  --kl-warmup 5 \
  --kl-cycle 10 \
  --kl-free-bits 0.02 \
  --context-dropout 0.40 \
  --energy-weight 1.0 \
  --min-raw-kl 0.05 \
  --max-raw-kl 20.0 \
  --min-latent-top1-change 0.05 \
  --save-every 5 \
  --early-stop-patience 8 \
  --metrics-path research/phase_2/models/vae_full_metrics.jsonl \
  --split-manifest research/phase_2/data/splits/materials_graphs_broad_seed42.json
```

Các flag Task 2–3 phải tồn tại trước khi chạy command này.

### Runtime

Run cũ mất khoảng 420–500 giây/epoch trên CPU. Objective mới có masking nên 50 epoch có thể mất nhiều giờ. Không train 80 epoch nếu validation đã plateau.

### Artifact bắt buộc

- `research/phase_2/models/vae_model.pt`: best broad checkpoint.
- `vae_v3_epochN.pt`: periodic checkpoint v3; không ghi đè snapshots objective cũ.
- `vae_full_metrics.jsonl` và run config JSON.
- Split manifest.
- Latent evaluation JSON.

### Check checkpoint

```bash
.conda/bin/python -c 'import torch; p="research/phase_2/models/vae_model.pt"; c=torch.load(p,map_location="cpu",weights_only=False); print(c["objective_version"],c["epoch"],c["best_val_score"],c["latent_gate_passed"],c["args"]); assert c["objective_version"]==3 and c["latent_gate_passed"]'
```

### Definition of Done

- Training kết thúc bằng early stopping hoặc hết epoch bình thường.
- Best checkpoint load được bằng generator.
- Metrics không có NaN/Inf.
- Test Task 1 vẫn pass sau training.
- Task 5 chạy trên best checkpoint và không cho kết quả latent ignored.

---

## TASK 6B — Tạo high-temperature proxy subset và fine-tune GraphVAE

### Mục tiêu

Điều chỉnh generator về vùng hóa học refractory/ceramic trong khi thermal evaluator thật được xây dựng. Proxy subset chỉ dùng cho generator pretraining/fine-tuning, không dùng làm nhãn nhiệt độ.

### Proxy rule version 1

Candidate row thỏa:

- Có ít nhất một refractory element: `W, Mo, Ta, Nb, Hf, Zr, Ti`.
- Có ít nhất một ceramic-forming element: `O, C, N, B, Si, Al`.
- Không chứa `H, F, Cl, Br, I` trong profile v1.
- Không chứa alkali `Li, Na, K, Rb, Cs` trong profile v1.
- `formation_energy_per_atom <= -1.0 eV/atom` làm stability prefilter.

Đây là rule có version, không phải định luật vật liệu. Mọi thay đổi element lists/threshold phải tăng `domain_rule_version` và tạo dataset artifact mới.

### Kích thước đã thống kê

Từ dataset hiện tại:

| Graph limits | Số proxy graph |
|---|---:|
| `max_nodes=64`, `max_edges=512` | 1.446 |
| `max_nodes=64`, `max_edges=1024` | 3.765 |
| `max_nodes=64`, `max_edges=2048` | 6.120 |
| Không edge limit | 7.348 |

Khởi điểm đề xuất là `max_nodes=64`, `max_edges=2048`, sau khi smoke test memory/runtime.

### Derived artifacts

```text
research/phase_2/data/processed/high_temperature_proxy_v1.csv
research/phase_2/data/processed/high_temperature_proxy_graphs.pt
research/phase_2/data/processed/high_temperature_proxy_v1_stats.json
research/phase_2/data/splits/high_temperature_proxy_v1_seed42.json
```

CSV phải chứa:

```text
material_uid
source_graph_index
domain_rule_version
refractory_elements
ceramic_forming_elements
formation_energy_per_atom
domain_eligible
rejection_reason
```

Không sửa/xóa broad source dataset.

### Trainer changes

Không sửa `vae_trainer.py` của broad baseline. Fine-tune chỉ chạy qua
`research/phase_2/generation/vae_proxy_finetune_trainer.py`, dùng
`proxy_data_loader.py` riêng.

Thêm:

```text
--domain-profile high_temperature_proxy_v1
--init-from research/phase_2/models/vae_model.pt
--checkpoint-name vae_high_temperature_proxy_v1.pt
```

`--init-from` chỉ load model weights objective v3 đã qua latent gate. Nó phải tạo optimizer và scheduler mới. Không dùng `--resume` vì resume mang optimizer/scheduler broad run sang fine-tune.

### Fine-tune config ban đầu

- Epochs: 10–20 với early stopping.
- Learning rate: `1e-4`.
- KL warmup mới: 5 epochs.
- Tính node class weights từ proxy train split, không dùng hardcoded broad frequencies.
- Lưu metrics và config riêng.

### Definition of Done

- Derived dataset có stats và provenance.
- Split group-aware, disjoint.
- Fine-tune checkpoint load được và giữ objective v3.
- Generator sinh candidate trong proxy domain tốt hơn broad checkpoint theo benchmark.
- Output vẫn ghi `high_temperature_assessment=proxy_only` cho tới khi Task 7B đạt.

---

## TASK 7A — Audit và sửa GNN formation-energy evaluator

### Vấn đề hiện tại

- `gnn_evaluator_training.py` dùng random split riêng.
- `test_gnn_evaluator.py` chỉ kiểm tra một graph ngẫu nhiên.
- Checkpoint chỉ là raw state dict, thiếu args/split/metrics.
- Chưa có MAE, RMSE, R² đầy đủ và baseline.

### Mục tiêu

Tạo evaluator có full test report, dùng split tái lập và có baseline rõ.

### Files được phép sửa/tạo

- Sửa `research/phase_2/gnn_evaluator_training.py`.
- Sửa `research/phase_2/test_gnn_evaluator.py`.
- Có thể tạo `research/phase_2/gnn_evaluation.py` để tách logic.

### Training requirements

- Dùng split manifest Task 2.
- Seed Python/Torch cố định.
- DataLoader batch thực, ví dụ 64 graph.
- Early stopping theo validation MAE.
- Lưu checkpoint dict:

```python
{
    "model": model.state_dict(),
    "epoch": epoch,
    "best_val_mae": best_val_mae,
    "args": vars(args),
    "split_manifest": str(path),
    "target": "formation_energy_per_atom",
    "checkpoint_version": 1,
}
```

`generator.load_gnn` phải đọc format mới và fail nếu kiến trúc không khớp.

### Full test report

Tính trên toàn bộ test split:

- `count`
- MAE
- RMSE
- R²
- mean signed error/bias
- median absolute error
- p90 absolute error
- true min/max
- predicted min/max

### Baseline bắt buộc

Predict mọi test sample bằng mean target của train split. Ghi baseline MAE/RMSE. GNN phải tốt hơn baseline rõ ràng.

### Group leakage audit

Random material split có thể để composition rất gần nhau ở cả train/test. Tạo thêm report group split theo ưu tiên:

1. Reduced formula từ processed CSV nếu có.
2. Nếu formula thiếu, dùng signature của sorted atomic numbers và counts.
3. Chemical-system signature chỉ dùng làm report bổ sung vì có thể tạo group quá lớn.

Ghi cả random-stratified report và group-split report để so sánh trước khi đổi split chính.

### Gate ban đầu

- Test MAE mục tiêu kỹ thuật: `<= 0.2 eV/atom`.
- GNN tốt hơn mean baseline.
- Group-split MAE không xấu đến mức ranking mất ý nghĩa.

Ngưỡng 0.2 không phải chứng nhận khoa học.

### Commands dự kiến

```bash
.conda/bin/python research/phase_2/gnn_evaluator_training.py \
  --split-manifest research/phase_2/data/splits/materials_graphs_broad_seed42.json \
  --epochs 50

.conda/bin/python research/phase_2/test_gnn_evaluator.py \
  --checkpoint research/phase_2/models/gnn_formation_energy_model.pt \
  --split-manifest research/phase_2/data/splits/materials_graphs_broad_seed42.json \
  --output research/phase_2/models/gnn_test_report.json
```

### Definition of Done

- Test script không dùng random graph.
- Có JSON report toàn test set.
- Checkpoint chứa metadata.
- Generator load đúng checkpoint mới.
- Có baseline và group leakage report.

---

## TASK 7B — Thu thập property và train high-temperature evaluator

### Mục tiêu

Xây model dự đoán primary thermal target đã khóa ở Task 0. Model này tách biệt với formation-energy GNN.

### Điều kiện bắt đầu

- Task 0 có primary target và schema được nhóm xác nhận.
- Property records có provenance và unit đã normalize.
- Có đủ high-confidence structure/material matches để tạo train/val/test.
- Measurements của cùng material không xuất hiện ở nhiều split.

### Data ingestion

Tạo script idempotent, ví dụ:

```text
research/phase_2/high_temperature/ingest_properties.py
research/phase_2/high_temperature/build_training_table.py
research/phase_2/high_temperature/data_quality_report.py
```

Không ghi đè raw source. Lưu raw, normalized và training table thành ba tầng riêng.

### Unit normalization

- Temperature chuẩn nội bộ: Kelvin.
- Lưu cả raw value/unit.
- Conductivity, thermal conductivity, pressure và time phải có unit chuẩn riêng.
- Reject record không xác định unit; không đoán từ magnitude.

### Duplicate measurements

Với nhiều measurements cho một material:

- Giữ raw records.
- Training target có thể dùng median sau khi lọc cùng property/protocol.
- Lưu min/max/std/count.
- Spread lớn phải tạo `quality_flag=conflicting_measurements`.

### Model baseline bắt buộc

Trước GNN, chạy:

1. Train-mean baseline.
2. Composition-feature tree/linear baseline nếu có.
3. Graph model chỉ được chấp nhận nếu tốt hơn baseline trên group-held-out test.

### Evaluation metrics

Regression:

- MAE theo Kelvin.
- RMSE.
- R².
- Median absolute error.
- p90 absolute error.
- Coverage/error theo material family.

Classification nếu target là survive/fail:

- ROC-AUC và PR-AUC.
- Precision, recall, F1 tại threshold đã chọn.
- Calibration/Brier score.

### Uncertainty

Ưu tiên ensemble nhiều seeds. Output thermal prediction phải có mean/std hoặc calibrated interval. Không xếp hạng candidate ngoài training domain mà không có OOD warning.

### Checkpoint metadata

```python
{
    "model": ..., 
    "target_name": ...,
    "target_unit": "K",
    "data_version": ...,
    "split_manifest": ...,
    "feature_schema": ...,
    "best_val_metric": ...,
    "checkpoint_version": 1,
}
```

### Definition of Done

- Có data quality và coverage report.
- Có group-held-out test report.
- Model tốt hơn baseline.
- Prediction có uncertainty/OOD status.
- Pipeline chỉ chuyển từ `proxy_only` sang `property_model` khi checkpoint này đạt gate.

---

## TASK 8 — Chuyển candidate graph thành cấu trúc CIF

### Vấn đề

Phase 2 candidate có atomic numbers và topology prototype nhưng chưa có lattice/site coordinates riêng. Phase 3 đã có structure cache và logic pymatgen; phải tái sử dụng thay vì tạo pipeline song song.

### Files cần đọc

- `research/phase_2/generation/generator.py`
- `research/phase_2/generation/main.py`
- `research/phase_3/generation.py`
- Code/notebook tạo `materials_graphs.pt` để hiểu thứ tự atom.

### Module đề xuất

Tạo `research/phase_2/generation/structure_builder.py` với API:

```python
def load_prototype_structure(uid: str) -> Structure:
    ...

def verify_graph_structure_alignment(graph, structure: Structure) -> None:
    ...

def apply_site_substitutions(
    structure: Structure,
    original_atomic_numbers: list[int],
    candidate_atomic_numbers: list[int],
) -> tuple[Structure, list[dict]]:
    ...

def write_candidate_cif(structure: Structure, output_path: Path) -> None:
    ...
```

### Alignment check bắt buộc

1. `len(structure) == graph.x.size(0)`.
2. `structure.atomic_numbers` bằng `graph.x.view(-1)` theo đúng thứ tự.
3. Nếu sai, reject với `graph_structure_order_mismatch`.
4. Không sort sites/atomic numbers để ép khớp.

### Áp dụng substitution

- Clone structure, không mutate prototype.
- Với mỗi site thay đổi, replace species đúng index.
- Giữ lattice và fractional coordinates của prototype.
- Ghi map gồm `site_index`, `from_Z`, `to_Z`, `from_symbol`, `to_symbol`.
- Ghi `structure_status=unrelaxed`.

### Candidate fields cần thêm

```text
prototype_uid
prototype_formula
candidate_formula
substitutions_json
num_substitutions
cif_path
structure_status
```

### Không được làm

- Không tuyên bố CIF đã relaxed.
- Không dùng graph edge count làm bằng chứng vật lý.
- Không bỏ alignment check để tăng candidate count.

### Tests

- Ít nhất 3 prototype tạo CIF.
- Parse lại bằng `Structure.from_file`.
- Formula đọc lại khớp manifest.
- Prototype object không đổi.
- Alignment mismatch phải fail có reason.

---

## TASK 9 — Chemistry filters và deduplication

### Vấn đề

`apply_chemistry_constraints` hiện thay nguyên tố không mong muốn thành O. Điều này âm thầm biến candidate và có thể vi phạm requirement. `VALENCE_ELECTRONS` chưa phải kiểm tra charge balance đáng tin.

### Nguyên tắc

Reject với reason cụ thể thay vì tự sửa candidate sau generation.

### Module đề xuất

Tạo `research/phase_2/generation/chemistry_filters.py`:

```python
@dataclass
class FilterResult:
    passed: bool
    reasons: list[str]
    metadata: dict

def validate_candidate(
    structure: Structure,
    requirement: RequirementQuery,
    prototype: Structure,
) -> FilterResult:
    ...
```

### Filter theo thứ tự

1. Atomic number trong `1..94`, không có mask class 0.
2. Có ít nhất một substitution.
3. Không vượt `max_unique_elements`.
4. Thỏa `include_elements`, `exclude_elements`, `only_elements`.
5. Không chứa element user cấm.
6. `Composition.oxi_state_guesses()` có charge-balanced assignment nếu filter bật.
7. Không có site-site distance gần 0 hoặc dưới threshold cấu hình.
8. Không giống hệt prototype theo formula và `StructureMatcher`.
9. Không trùng candidate đã giữ.

Với `domain_profile=high_temperature_proxy_v1`, áp dụng thêm refractory/ceramic rule của Task 6B và ghi rule version trong result. Không âm thầm thay element để candidate vượt filter.

Các check nhiệt độ dự đoán, oxidation, creep hoặc conductivity không nằm trong chemistry filter; chúng thuộc evaluator/scorer có nhãn và uncertainty.

### Dedup hai tầng

1. Hash nhanh theo reduced formula + số site.
2. Cùng hash thì dùng `StructureMatcher.fit`.

### Filter summary

```json
{
  "generated": 100,
  "passed": 12,
  "rejected": 88,
  "rejection_counts": {
    "unchanged": 20,
    "requirement_elements": 15,
    "charge_balance": 30,
    "duplicate": 23
  }
}
```

### Definition of Done

- Mỗi rejection có reason.
- Không còn logic tự thay nguyên tố lạ thành O.
- Oxide requirement không trả candidate thiếu O.
- High-temperature proxy candidate thỏa đúng rule version đã khai báo.
- Candidate không được gắn nhãn `property_model` nếu chỉ qua proxy filters.
- Dedup deterministic với seed cố định.

---

## TASK 10 — Uncertainty và ranking đa property

### Mục tiêu

Không xếp hạng bằng formation energy duy nhất. Ranking chịu nhiệt thật phải kết hợp thermal target, ổn định nhiệt động, validity và uncertainty.

### Phương án tối thiểu

Train ensemble cho formation-energy và thermal evaluator với nhiều seeds, dùng đúng split manifest của từng target.

Với mỗi candidate, ghi:

```text
formation_energy_mean
formation_energy_std
formation_energy_upper = mean + 2 * std
thermal_target_mean_K
thermal_target_std_K
thermal_target_lower_K = mean - 2 * std
thermal_assessment_mode
ranking_score
```

Rules:

- Formation energy thấp hơn tốt hơn; dùng upper bound bảo thủ `mean + 2*std`.
- Thermal target cao hơn tốt hơn; dùng lower bound bảo thủ `mean - 2*std`.
- Candidate chỉ có proxy không được so chung như thể có thermal prediction thật.
- Chuẩn hóa từng score bằng statistics của validation set trước khi weighted sum.
- Weights phải nằm trong config/run artifact, không hardcode không ghi lại.

Ranking v1 có thể dùng filter theo thermal lower bound trước, sau đó sort formation-energy upper bound. Chỉ dùng weighted sum khi đã benchmark/calibrate.

### OOD warnings tối thiểu

- Candidate có element không xuất hiện trong train split.
- Node/edge count ngoài p01–p99 train distribution.
- Formula hoặc chemical system hoàn toàn mới so với train.
- Candidate thiếu thermal property coverage hoặc match confidence thấp.

OOD candidate có thể được giữ nhưng phải gắn warning; không trình bày prediction như chắc chắn.

### Files dự kiến

- Có thể tạo `research/phase_2/gnn_ensemble.py`.
- Sửa `research/phase_2/generation/generator.py` để gọi ensemble scorer.
- Sửa Candidate dataclass và manifest fields trong `main.py`.

### Tests

- Ensemble predictions được aggregate đúng mean/std.
- Candidate cùng thermal mean nhưng uncertainty cao bị xếp thấp hơn.
- Candidate proxy-only không có fake Kelvin prediction.
- Checkpoint thiếu hoặc kiến trúc lệch phải fail.
- Ranking deterministic.

### Definition of Done

- Manifest có formation mean/std/upper, thermal mean/std/lower và ranking score.
- Candidate fail chemistry không được ranking.
- Summary ghi assessment mode, ensemble sizes, checkpoint paths và score weights.

---

## TASK 11 — Requirement schema và LLM parser contract

### Vấn đề

String `requirement` hiện chủ yếu được in/log; filter thật phụ thuộc CLI fields truyền riêng. Requirement và generation có thể không nhất quán.

### Schema đề xuất

Tạo module `research/phase_2/generation/requirements.py`:

```python
@dataclass
class RequirementQuery:
    raw_text: str
    minimum_service_temperature_K: float | None = None
    atmosphere: str | None = None
    exposure_time_hours: float | None = None
    requires_electrical_insulation: bool | None = None
    min_formation_energy: float | None = None
    max_formation_energy: float | None = None
    include_elements: list[str] = field(default_factory=list)
    exclude_elements: list[str] = field(default_factory=list)
    only_elements: list[str] = field(default_factory=list)
    max_unique_elements: int | None = None
    max_candidates: int = 20
    unsupported_properties: list[str] = field(default_factory=list)
```

### Validation rules

- Normalize symbol capitalization: `fe` thành `Fe`.
- Reject symbol không tồn tại.
- `include_elements` là subset của `only_elements` nếu only-list có giá trị.
- Element không thể vừa include vừa exclude.
- `min_energy <= max_energy`.
- Celsius/Fahrenheit được normalize sang Kelvin nhưng giữ raw input.
- Nếu query có nhiệt độ nhưng thermal evaluator chưa đạt Task 7B, property đi vào `unsupported_properties` và pipeline chỉ chạy proxy mode.
- Nếu yêu cầu cách điện nhưng chưa có band-gap/conductivity evaluator, không được suy ra từ formation energy.
- Atmosphere/exposure time phải được giữ trong requirement; không bỏ qua khi so với service-temperature data.

### Vai trò LLM

LLM chỉ trả JSON theo schema. Code deterministic validate, retrieval, filter và rank.

Nếu parse fail:

- Không đoán silent.
- Báo raw response và field sai.
- Cho phép structured CLI bypass LLM để test offline.

### Plumbing bắt buộc

Cùng một `RequirementQuery` phải đi qua:

```text
parser -> retrieval -> generator -> chemistry filters -> ranking -> summary
```

Không tạo nhiều bản copy fields dễ lệch nhau.

### Tests

- `chịu 1500°C trong không khí` phải giữ temperature và atmosphere sau normalize.
- Requirement refractory oxide phải tạo include O và domain profile tương ứng.
- Include/exclude conflict fail.
- Invalid element fail.
- Unsupported property được giữ trong warnings.
- JSON round-trip không mất field.

### Definition of Done

- Requirement thực sự ràng buộc candidate.
- Summary chứa normalized requirement và unsupported warnings.
- Pipeline ghi rõ `proxy_only` hay `property_model`.
- Online/offline dùng cùng schema.

---

## TASK 12 — End-to-end pipeline

### Hai chế độ bắt buộc

#### Offline mode

Không cần LLM/Neo4j. Truyền UID prototype trực tiếp:

```bash
.conda/bin/python research/phase_2/generation/main.py \
  --uids MP_mp-1173034,MP_mp-1100894 \
  --domain-profile high_temperature_proxy_v1 \
  --min-service-temperature-c 1500 \
  --n-samples 20 \
  --top-k 10
```

Nếu `main.py` chưa có `--uids`, thêm flag và bypass retrieval có chủ đích. Offline mode là integration path chính vì tái lập và không phụ thuộc service.

#### Online mode

LLM parser + Neo4j MCP retrieval. Chỉ test sau khi offline pass.

### Output mỗi run

Không ghi đè output cũ:

```text
research/phase_2/generation/output/<run_id>/
  config.json
  retrieval.json
  all_candidates.csv
  validated_candidates.csv
  summary.json
  cifs/
```

### Provenance mỗi candidate

- Run ID và random seed.
- Prototype UID/formula.
- Candidate formula.
- Site substitutions.
- VAE checkpoint path/version/epoch.
- GNN checkpoint paths.
- Formation energy mean/std/upper.
- Thermal target mean/std/lower nếu có model thật.
- `high_temperature_assessment`: `proxy_only` hoặc `property_model`.
- Thermal dataset/model version và operating-condition match.
- Filter result và warnings.
- CIF path và `structure_status=unrelaxed`.

### Integration tests

1. Missing VAE checkpoint phải fail; không tạo fake structural candidate.
2. Old objective checkpoint phải fail.
3. Missing GNN checkpoint phải fail validation; không dùng random GNN.
4. UID không tồn tại phải report rõ.
5. Oxide requirement không trả non-oxide.
6. CIF parse được.
7. Summary counts khớp CSV.
8. Candidate IDs unique.
9. Proxy-only run không xuất temperature prediction giả.
10. Property-model run fail nếu thermal checkpoint/data version không tương thích.

### Fallback cần sửa

`_fallback_generation_from_buckets` tạo formula text `(interpolated A <-> B)`. Nó không phải material structure. Chỉ giữ cho retrieval demo hoặc gắn `is_structural_candidate=false`; không cho đi qua GNN/chemistry như graph thật.

### Definition of Done

- Offline end-to-end chạy không cần service.
- Online mode fail rõ khi service/auth thiếu.
- Output đầy đủ provenance.
- Không silent fallback sang random initialized model.

---

## TASK 13 — Benchmark, ablation và báo cáo

### Baselines

1. Retrieval-only: trả prototype tốt nhất.
2. Pymatgen substitution từ Phase 3.
3. GraphVAE masked substitution.

### Query set tối thiểu

Tạo ít nhất 10 structured query kiểm tra được bằng dữ liệu hiện tại:

- Refractory oxide cho service temperature mục tiêu.
- Refractory carbide/nitride/boride.
- Candidate chứa `W` hoặc `Mo` và ceramic-forming element.
- Candidate cách điện ở nhiệt độ mục tiêu, chỉ khi có conductivity/band-gap evaluator.
- Cùng temperature nhưng atmosphere khác nhau để kiểm tra condition handling.
- Formation energy trong một khoảng làm stability constraint phụ.

Khi chưa có evaluator nhiệt, benchmark chỉ đo proxy retrieval/generation và phải gắn `proxy_only`. Không tính “đạt 1500°C” là success.

### Metrics

- Retrieval success rate.
- Generated count.
- Chemistry validity rate.
- Unique formula/structure ratio.
- Novelty so với prototype/database.
- Unchanged ratio.
- Formation-energy mean/std distribution.
- Thermal-target MAE/interval coverage trên labeled test.
- Tỷ lệ candidate vượt thermal lower-bound requirement.
- Proxy-only versus property-model coverage.
- OOD warning rate.
- Runtime từng stage.

### Ablation

- Không latent perturbation vs có perturbation.
- Node mask rate 0.1/0.3/0.5.
- Temperature 0.0/0.3/0.7/1.0.
- GraphVAE vs pymatgen substitution.
- Ranking mean vs mean + 2*std.
- Broad-pretrained VAE vs high-temperature fine-tuned VAE.
- Formation-only ranking vs thermal + formation ranking.

### Artifacts

```text
research/phase_2/generation/BENCHMARK_REPORT.md
research/phase_2/generation/output/benchmark_results.csv
```

Báo cáo phải ghi cả failure cases và giới hạn khoa học, không chỉ top candidate.

---

## 5. Prompt mẫu giao task cho model nhỏ

Thay `<TASK_ID>` bằng số task:

```text
Bạn đang làm TASK <TASK_ID> trong file
research/phase_2/generation/NEXT_STEPS_IMPLEMENTATION.md.

Yêu cầu:
1. Đọc toàn bộ phần Quy tắc làm việc và TASK <TASK_ID>.
2. Đọc toàn bộ các file trong "Files cần đọc".
3. Chỉ sửa các file trong "Files được phép tạo/sửa".
4. Không xóa hoặc làm yếu TRAINING_OBJECTIVE_VERSION=3,
   masked-target invariance, held-out-edge isolation hoặc checkpoint guard.
5. Triển khai đầy đủ, không chỉ viết pseudocode.
6. Chạy mọi test và Definition of Done của task.
7. Cuối cùng báo:
   - file đã thay đổi,
   - test đã chạy và output chính,
   - phần chưa đạt,
   - rủi ro còn lại.

Không bắt đầu task tiếp theo.
```

Nếu model sửa nhiều task cùng lúc, dừng và yêu cầu quay lại đúng một task.

## 6. Checklist tổng cuối dự án

- [ ] Primary thermal target, unit và operating conditions được khóa.
- [ ] High-temperature property table có provenance và quality flags.
- [ ] Test chống node/edge leakage pass.
- [ ] Split manifest tái lập, disjoint và không mất sample.
- [ ] GraphVAE checkpoint objective v3 và latent gate pass.
- [ ] Broad checkpoint được fine-tune trên versioned high-temperature proxy subset.
- [ ] Pilot metrics hợp lý, không reconstruction về 0 giả.
- [ ] Latent evaluation cho thấy `z` ảnh hưởng output.
- [ ] GNN có full test report và tốt hơn baseline.
- [ ] Thermal evaluator có group-held-out report, uncertainty và tốt hơn baseline.
- [ ] Candidate map sang Structure/CIF với alignment check.
- [ ] Chemistry filters không âm thầm sửa candidate.
- [ ] Requirement đi xuyên suốt pipeline.
- [ ] Formation-energy uncertainty được ghi và dùng ranking.
- [ ] Thermal uncertainty/lower bound được dùng khi property model khả dụng.
- [ ] Proxy-only output không đưa ra claim nhiệt độ thật.
- [ ] Offline end-to-end pass.
- [ ] Online retrieval/MCP pass.
- [ ] Benchmark so sánh retrieval, pymatgen và GraphVAE.
- [ ] Output có provenance đầy đủ.

## 7. Ưu tiên thực hiện ngay

1. Để run 30-epoch broad pretraining đang chạy hoàn thành; không sửa trainer/model giữa run.
2. TASK 0 — khóa primary thermal target và property-data contract.
3. TASK 1 — test chống leakage cho broad checkpoint.
4. TASK 2–3 — split manifest, metrics và early stopping.
5. TASK 5 — latent usefulness trên broad checkpoint.
6. TASK 6B — tạo proxy subset và fine-tune GraphVAE.
7. TASK 7A — audit formation-energy GNN.
8. TASK 7B — ingest property thật và train thermal evaluator.
9. Sau đó mới dựng CIF, filters, ranking và end-to-end.

Không bắt đầu bằng UI hay LLM integration. Hai rủi ro lớn nhất là thiếu ground-truth chịu nhiệt và evaluator formation/thermal chưa được kiểm định trên group-held-out test.
