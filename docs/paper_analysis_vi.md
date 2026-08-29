# Phân tích paper “Near-Field Integrated Sensing and Communications” (2023)

## 1. Thông tin và mục tiêu chính

Paper của Zhaolin Wang, Xidong Mu và Yuanwei Liu đề xuất một hệ thống ISAC băng hẹp, đơn tĩnh (monostatic), trong đó trạm gốc dùng ULA rất lớn để vừa truyền dữ liệu xuống nhiều user vừa cảm nhận một target. Điểm khác biệt cốt lõi là paper không xấp xỉ sóng phẳng: độ cong của sóng cầu trong near field làm steering vector phụ thuộc đồng thời vào khoảng cách và góc.

Ba đóng góp kỹ thuật chính là:

1. Mô hình channel near-field chính xác theo khoảng cách từ từng phần tử anten đến user/target.
2. FIM/CRB cho bài toán ước lượng đồng thời khoảng cách và góc, trong khi hệ số phản xạ phức của target là nuisance parameter.
3. Thiết kế covariance/beamformer ISAC: SDR toàn cục cho kiến trúc fully digital và heuristic hai giai đoạn cho kiến trúc hybrid.

Nguồn chính: [bản arXiv](https://arxiv.org/abs/2302.01153), [DOI](https://doi.org/10.1109/LCOMM.2023.3280132), và [mã MATLAB của tác giả](https://github.com/zhaolin820/near-field-integrated-sensing-and-communications).

## 2. Mô hình hệ thống

ULA có số anten lẻ `N = 2*N_tilde + 1`, tâm mảng đặt tại gốc tọa độ và phần tử thứ `n` ở `(n*d, 0)`. Với một điểm có tọa độ cực `(r, theta)`, khoảng cách đến anten thứ `n` là

```math
r_n(r,\theta)=\sqrt{r^2+n^2d^2-2rnd\cos\theta}.
```

Steering vector near-field bỏ common phase theo khoảng cách đến tâm mảng:

```math
[a(r,\theta)]_n=\exp\left[-j\frac{2\pi}{\lambda}(r_n-r)\right].
```

Khi `r` rất lớn so với aperture, khai triển Taylor bậc một cho ta steering vector far-field

```math
[a_{far}(\theta)]_n=\exp\left(j\frac{2\pi}{\lambda}nd\cos\theta\right),
```

không còn phụ thuộc vào khoảng cách. Đây là lý do near-field MUSIC có một peak hai chiều, còn far-field MUSIC tạo một ridge theo cùng phương góc.

Rayleigh distance là `2*D^2/lambda`. Với `D=0.5 m` và `f=28 GHz`, giá trị dùng trong paper xấp xỉ 46.7 m. Target được đặt tại `(20 m, 45°)`, nằm trong vùng này.

## 3. Tín hiệu communication và sensing

Tín hiệu phát tại snapshot `t` là

```math
x[t]=\sum_{k=1}^{K} f_k c_k[t]+s[t],
```

trong đó `f_k` là beamformer communication và `s[t]` là tín hiệu sensing riêng với covariance `R_s`. Tổng covariance là

```math
R_x=\sum_k f_k f_k^H+R_s.
```

Paper dùng convention `h_k^T f_k`, vì vậy rate của user `k` được tính từ

```math
\mathrm{SINR}_k=
\frac{|h_k^T f_k|^2}
{h_k^T R_x h_k^* - |h_k^T f_k|^2+\sigma_k^2}.
```

Với sensing đơn tĩnh, round-trip response chưa gồm hệ số target là

```math
\widetilde G=a(r_s,\theta_s)a^T(r_s,\theta_s),
```

và channel thực là `G = beta_s * G_tilde`. Echo nhận được là `y_s[t]=Gx[t]+z_s[t]`.

## 4. MUSIC near-field

Từ sample covariance của echo, eigenvector ứng với eigenvalue lớn nhất tạo signal subspace của một target; các eigenvector còn lại tạo noise subspace `E_n`. Pseudo-spectrum được tính bởi

```math
P(r,\theta)=\frac{1}{a^H(r,\theta)E_nE_n^Ha(r,\theta)}.
```

Code quét trên lưới Cartesian `(x,y)`, chuyển sang `(r,theta)` rồi tìm phần tử lớn nhất. Với model far-field, tất cả điểm nằm trên cùng một tia có steering vector giống nhau, nên khoảng cách không identifiable.

## 5. FIM và CRB

Vector tham số chưa biết là

```math
\xi=[r_s,\theta_s,\Re\{\beta_s\},\Im\{\beta_s\}]^T.
```

FIM được chia block

```math
J_\xi=\begin{bmatrix}J_{11}&J_{12}\\J_{12}^T&J_{22}\end{bmatrix},
```

trong đó `J_11` chứa thông tin range/angle, `J_22` chứa thông tin hệ số phản xạ và `J_12` mô tả coupling. Sau khi loại nuisance parameter bằng Schur complement,

```math
\mathrm{CRB}_{r,\theta}=
\left(J_{11}-J_{12}J_{22}^{-1}J_{12}^T\right)^{-1}.
```

Hai metric được plot là

- range RCRB: `sqrt(CRB[0,0])`, đơn vị mét;
- angle RCRB: `sqrt(CRB[1,1]) * 180/pi`, đơn vị độ.

Các đạo hàm `da/dr` và `da/dtheta` được cài analytic trong `channels.py` và so với finite difference trong test. Điều này quan trọng vì đạo hàm range rất nhỏ ở xa và dễ bị sai số trừ số nếu dùng finite difference trực tiếp.

## 6. Bài toán tối ưu fully digital

Paper tối thiểu hóa `trace(CRB)` với ba nhóm ràng buộc:

1. rate của từng user không nhỏ hơn `R_min,k`;
2. `trace(R_x) <= P_max`;
3. `R_x - sum_k f_k f_k^H` là PSD để tồn tại sensing covariance hợp lệ.

Đặt `F_k=f_k f_k^H`. Bỏ tạm ràng buộc rank-one tạo ra SDP convex. Ràng buộc rate trở thành tuyến tính theo `F_k` và `R_x`:

```math
|h_k^T f_k|^2 \ge
(2^{R_{min,k}}-1)\left(h_k^T R_xh_k^*-|h_k^Tf_k|^2+\sigma_k^2\right).
```

Objective nghịch đảo được biểu diễn bằng Schur complement qua ma trận phụ `U`. Điểm đặc biệt của bài toán này là SDR chặt: từ nghiệm lifted tối ưu có thể khôi phục

```math
f_k^\star=
\frac{F_k h_k^*}{\sqrt{h_k^T F_k h_k^*}}
```

mà không đổi objective và vẫn thỏa constraint. Do đó fully digital là upper bound hiệu năng sensing trong mô hình paper.

Trong Python, FIM có conditioning rất mạnh giữa range, angle và reflection blocks. `optimization.py` dùng một congruence preconditioner cùng epigraph

```math
\begin{bmatrix}U&I\\I&V\end{bmatrix}\succeq0
```

để biểu diễn `V >= U^{-1}`. Phép biến đổi giữ nguyên feasible set và objective gốc, nhưng ổn định hơn nhiều cho CLARABEL/SCS.

## 7. Kiến trúc hybrid

RF beamformer có unit-modulus. Giai đoạn một của paper chọn từng cột theo conjugate steering vector:

- `K` cột đầu focus vào `K` user;
- các cột còn lại focus vào target.

Sau khi cố định RF beamformer, covariance và beamformer baseband được giải bằng cùng SDR nhưng ở số chiều `N_RF`. Receive combiner được lấy ngẫu nhiên trên unit circle. Paper dùng xấp xỉ `(1/N) W_RF W_RF^H ≈ I`, dẫn tới noise hiệu dụng `N*sigma_s^2`.

Mã MATLAB public không cài phần hybrid. Phần này trong repository là reconstruction trực tiếp từ Eq. (14), (15), và (22), không phải port từng dòng từ upstream.

## 8. Đọc ba kết quả chính

### Fig. 2 — RCRB theo minimum rate

Khi `R_min` tăng, covariance phải dành nhiều tài nguyên hơn cho communication và ít tự do hơn cho sensing. Vì vậy RCRB có xu hướng tăng. Fully digital là bound tốt hơn hybrid. Độ dốc và giá trị cụ thể phụ thuộc mạnh vào realization vị trí user.

### Fig. 3 — MUSIC spectrum

Near-field spectrum có peak quanh `(x,y)≈(14.14,14.14) m`, tương ứng `(20 m,45°)`. Far-field spectrum có ridge dọc theo `y=x`; một argmax đơn lẻ trên ridge không phải là ước lượng range hợp lệ.

### Fig. 4 — RCRB theo khoảng cách

Paper loại pathloss để chỉ quan sát ảnh hưởng hình học. Khi target ra xa, curvature giảm nên range RCRB tăng nhanh. Angle estimation thường tốt dần và tiến về giới hạn far-field do các anten nhìn target từ hướng gần giống nhau hơn.

## 9. Những điểm không đủ để reproduce tuyệt đối

Đây là các nguồn sai khác quan trọng cần ghi trong bất kỳ báo cáo thực nghiệm nào:

- Paper không công bố seed, realization vị trí bốn user hoặc số lần Monte Carlo.
- MATLAB upstream không đặt RNG seed và chỉ cung cấp pipeline fully digital cho MUSIC.
- Vị trí user trong upstream được lấy đều từ 0 đến Rayleigh distance, dù phần mô hình nêu Fresnel lower bound `1.2D`.
- Convention path gain trong upstream là `rho_0=lambda/(4*pi)` rồi dùng `sqrt(rho_0)/r`; repository giữ convention này để đối chiếu code thay vì tự đổi sang một phiên bản Friis khác.
- Fig. 4 chỉ nói “không tính pathloss” mà không nêu chính xác normalization. Baseline giữ nguyên complex target gain được sinh tại 20 m trong toàn bộ sweep.
- Giá trị Fig. 2/4 có thể không khớp từng điểm với paper dù xu hướng đúng. Một claim reproduce nghiêm túc nên báo cả seed, solver, tolerance, status, achieved rates và file CSV.

## 10. Giới hạn khoa học của mô hình

Mô hình giả sử băng hẹp, một target, LOS, perfect communication CSI, target location từ coherent block trước, gain gần như bằng nhau trên toàn aperture, và không có coupling/quantization/hardware impairment. Với mảng cực lớn hoặc băng rộng, các hiệu ứng spatial non-stationarity, beam squint, near-field path gain khác nhau theo anten và target extent có thể trở nên đáng kể.

CRB cũng là local lower bound cho estimator không chệch; nó không bảo đảm MUSIC hữu hạn snapshot đạt bound, nhất là ở SNR thấp hoặc khi xuất hiện ambiguity/sidelobe.

## 11. Mapping từ paper sang code

| Thành phần | File/function |
|---|---|
| Eq. (1), (4), (5) | `channels.py`: `near_field_response`, `far_field_response` |
| Eq. (6), (7) | `channels.py`: `generate_scenario` |
| Eq. (11) | `communication.py`: `communication_rates` |
| Eq. (13), Appendix B | `fim.py`: `fisher_information_blocks`, `crb_matrix` |
| Eq. (20), (21) | `optimization.py`: `solve_fully_digital_sdr`, rank-one recovery |
| Eq. (22) | `optimization.py`: `hybrid_analog_beamformer` |
| Eq. (23), (24) | `music.py`: `noise_projector`, `music_spectrum_xy` |
| Fig. 2–4 | `experiments.py` và `scripts/reproduce_figure*.py` |

## 12. Hướng mở rộng optimization

Baseline hiện có ZF để làm lower-complexity comparison. Các hướng tiếp theo nên thêm theo thứ tự:

1. **WMMSE/SCA:** scalarize `trace(CRB)` và weighted sum-rate hoặc giữ rate constraint, luân phiên update receiver weight và waveform.
2. **Riemannian hybrid optimization:** tối ưu phase RF trực tiếp trên product of complex circles thay vì cố định steering columns.
3. **Robust design:** lấy expectation/worst case trên một vùng bất định `(r,theta)` thay vì giả sử target location đã biết chính xác.
4. **First-order large-scale solver:** khai thác low-rank/channel structure để tránh nhiều PSD cone dày `N x N`.
5. **Multi-target formulation:** mở rộng FIM lên `2M x 2M`, xử lý target association và correlated echoes.

Khi thêm thuật toán mới, nên dùng cùng file scenario/seed và kiểm tra ba điều trước khi so CRB: power constraint, minimum achieved rate và PSD của residual sensing covariance.

## 13. Tối ưu thời gian chạy

CPU không chạy 100% không đồng nghĩa chương trình bị lỗi. Fully-digital SDP chủ yếu tốn thời gian ở canonicalization, sparse/dense factorization và đồng bộ bộ nhớ. CLARABEL với linear solver QDLDL mặc định thường không dùng hết nhiều core.

Các điểm `R_min` trong Fig. 2 và các khoảng cách trong Fig. 4 độc lập. Baseline hỗ trợ chạy chúng bằng nhiều process:

```powershell
python main.py all --preset paper --solver CLARABEL --workers 4 --solver-threads 1
```

Nên bắt đầu với `--workers 2`, theo dõi RAM, sau đó tăng lên 4. Mỗi process paper-size giữ nhiều PSD matrix lớn nên tăng worker quá cao có thể làm paging và chậm hơn.

MOSEK có parallelism bên trong solver. Với MOSEK, có thể bắt đầu bằng:

```powershell
python main.py all --preset paper --solver MOSEK --workers 1 --solver-threads 14
```

Quy tắc thực dụng là giữ `workers * solver_threads` không vượt quá số logical CPU. Với CLARABEL/QDLDL, tăng `workers` thường hiệu quả hơn tăng `solver_threads`. Với MOSEK, nên ưu tiên thread bên trong solver trước.

MUSIC grid được tính bằng identity `P_noise = I - E_signal E_signal^H`, tránh nhân projector `N x N` tại từng grid point. Đây là phép biến đổi tương đương nhưng giảm độ phức tạp của grid evaluation từ gần `O(N^2 G)` xuống `O(N G)` cho một target.
