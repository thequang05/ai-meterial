# W–C refractory-carbide structural validation v1

## Kết luận

Đã hoàn tất **validation cấu trúc bằng geometry rules + CHGNet** cho campaign W–C. Có **15/15** ứng viên hội tụ lực và qua audit hình học sau relax. Kết quả này **chưa phải DFT** và chưa chứng minh ổn định nhiệt động hay chịu nhiệt.

## Funnel

| Cổng | Số lượng |
|---|---:|
| Latent samples | 500 |
| Chemistry-pass occurrences | 74 |
| Unique pre-CIF hypotheses | 44 |
| CIF dựng thành công | 44 |
| Geometry-ready trước relax | 44 |
| StructureMatcher-unique | 38 |
| Công thức đại diện | 35 |
| Chọn để CHGNet | 15 |
| CHGNet hội tụ | 15 |
| Qua audit sau relax | 15 |
| DFT-validated | 0 |
| Thermal-validated | 0 |

## 15 ứng viên đã ML-relax

| # | Công thức | GNN Eform (eV/atom) | Fmax cuối (eV/Å) | ΔV (%) | ΔE CHGNet (eV/atom) | SG sau relax |
|---:|---|---:|---:|---:|---:|---|
| 1 | Ti3NbWC5 | -0.5504 | 0.0433 | +2.33 | -0.0159 | R3m |
| 2 | Zr3TiWC5 | -0.5457 | 0.0470 | -4.70 | -0.0441 | R3m |
| 3 | Ti3VWC5 | -0.5225 | 0.0458 | -2.17 | -0.0127 | R3m |
| 4 | Zr3TaWC5 | -0.5146 | 0.0349 | -3.05 | -0.0323 | R3m |
| 5 | Ti2VWC4 | -0.4746 | 0.0174 | -2.92 | -0.0198 | R-3m |
| 6 | Ti2WC3 | -0.4651 | 0.0358 | -3.83 | -0.0165 | P-3m1 |
| 7 | Ti3W2C5 | -0.3943 | 0.0478 | +0.92 | -0.0047 | R-3m |
| 8 | Ta3NbWC5 | -0.3852 | 0.0410 | +0.20 | -0.0029 | R3m |
| 9 | Ta3VWC5 | -0.3663 | 0.0235 | -3.50 | -0.0366 | R3m |
| 10 | TaTi2W2C5 | -0.3640 | 0.0479 | +3.24 | -0.0183 | R3m |
| 11 | Ti2MoWC4 | -0.3545 | 0.0396 | +0.28 | -0.0029 | R-3m |
| 12 | Ta2WC3 | -0.3394 | 0.0322 | +2.92 | -0.0136 | P-3m1 |
| 13 | Ti2VW2C5 | -0.3098 | 0.0394 | -1.30 | -0.0170 | R3m |
| 14 | Ta3MoWC5 | -0.3042 | 0.0470 | -1.09 | -0.0079 | R3m |
| 15 | Ta3W2C5 | -0.2537 | 0.0184 | -1.12 | -0.0094 | R3m |

Ghi chú: ΔE CHGNet chỉ so sánh trước/sau relax của **cùng một cấu trúc**; không dùng năng lượng thô CHGNet để xếp hạng các công thức khác nhau.

## Hàng đợi DFT đề xuất

1. `Ti3NbWC5` — `camp_41a80c8b8b30` — `/Users/koiita/Downloads/ai-meterial/research/phase_2/generation/output/w_c_campaign_v1/structures/chgnet_relax_top15_v1/relaxed_cifs/camp_41a80c8b8b30_chgnet_relaxed.cif`
2. `Zr3TiWC5` — `camp_015e29be10e0` — `/Users/koiita/Downloads/ai-meterial/research/phase_2/generation/output/w_c_campaign_v1/structures/chgnet_relax_top15_v1/relaxed_cifs/camp_015e29be10e0_chgnet_relaxed.cif`
3. `Ti3VWC5` — `camp_59538db42ea7` — `/Users/koiita/Downloads/ai-meterial/research/phase_2/generation/output/w_c_campaign_v1/structures/chgnet_relax_top15_v1/relaxed_cifs/camp_59538db42ea7_chgnet_relaxed.cif`
4. `Zr3TaWC5` — `camp_5e6d485d7147` — `/Users/koiita/Downloads/ai-meterial/research/phase_2/generation/output/w_c_campaign_v1/structures/chgnet_relax_top15_v1/relaxed_cifs/camp_5e6d485d7147_chgnet_relaxed.cif`
5. `Ti2VWC4` — `camp_436da3bac8cc` — `/Users/koiita/Downloads/ai-meterial/research/phase_2/generation/output/w_c_campaign_v1/structures/chgnet_relax_top15_v1/relaxed_cifs/camp_436da3bac8cc_chgnet_relaxed.cif`

Thứ tự này kế thừa GNN screening sau khi mọi ứng viên đều qua cùng cổng CHGNet; nó là hàng đợi tính toán, không phải bảng xếp hạng độ bền nhiệt.

## Việc tiếp theo

1. DFT cell/ionic relaxation cho top 5 với cùng pseudopotential, cutoff và k-point policy.
2. DFT static energy và dựng convex hull với toàn bộ pha cạnh tranh trong cùng chemical system.
3. Chỉ giữ ứng viên có energy-above-hull phù hợp; sau đó chạy phonon/elastic và proxy nhiệt.
4. Đánh giá nhiệt thật khi có nhãn phù hợp; LLM/API chỉ hỗ trợ truy xuất/chuẩn hóa, không thay phép đo hoặc mô hình vật lý.

## Giới hạn khoa học

- GNN formation energy is a screening prediction, not a stability proof.
- CHGNet raw energies are ML potential energies and must not be compared across different compositions as formation energies.
- CHGNet relaxation is not DFT relaxation.
- No convex-hull, phonon, elastic, oxidation, melting-point, creep, or experimental thermal validation has been completed.
