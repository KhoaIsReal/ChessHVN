# Chess AI: Research Prototype for Huge Layer Value Networks

Research prototype exploring the hypothesis:

> **Hypothesis**: An ultra-wide value network ("Huge Layer"), trained with Temporal-Difference (TD) Learning, coupled with Minimax/Negamax + Alpha-Beta pruning, can acquire chess evaluation knowledge purely through self-play. Furthermore, the network dynamically expands its parameter capacity as training progresses, utilizing custom Sparse Residuals and Dynamic Neuron Masking ("Trống neuron học bá") to prevent representation collapse, mitigate co-adaptation, and eliminate over-reliance on individual neurons.

---

## 1. System Architecture

```text
       Chess Board (python-chess)
                  │
                  ▼
         BoardEncoder (19 planes)
                  │
                  ▼ Tensor [B, 1216]
        ┌──────────────────┐
        │  Huge Layer 1    │  Linear(1216 -> H1) + Act + NeuronMask + SparseResidual
        └─────────┬────────┘
                  ▼ Tensor [B, H1]
        ┌──────────────────┐
        │  Huge Layer 2    │  Linear(H1 -> H2) + Act + NeuronMask + SparseResidual
        └─────────┬────────┘
                  ▼ Tensor [B, H2]
        ┌──────────────────┐
        │    Value Head    │  Linear(H2 -> 1)
        └─────────┬────────┘
                  ▼
              tanh(z)
                  │
                  ▼
             V(s) ∈ [-1.0, +1.0]
```

### Value Convention & Perspective
* **Global Output**: $V_\theta(s) \in [-1.0, +1.0]$ represents the expected outcome evaluated from **White's perspective**:
  * $+1.0$: White is winning / won
  * $0.0$: Draw
  * $-1.0$: Black is winning / won (White lost)
* **Negamax Leaf Evaluation**: Negamax requires the evaluation to be from the **Side-To-Move (STM)** perspective:
  $$V_{\text{STM}}(s) = \begin{cases} +V_\theta(s) & \text{if White to move} \\ -V_\theta(s) & \text{if Black to move} \end{cases}$$
* **Consistency**: This definition eliminates alternating sign confusion in the TD replay buffer, allowing straightforward TD updates: $y_t = r_{t+1} + \gamma V(s_{t+1})$.

---

## 2. Board Representation (Encoder)

The deterministic `BoardEncoder` converts any `chess.Board` into a tensor of shape `[19, 8, 8]` (flattened into `[1216]`):

| Channel Index | Description | Value |
|---|---|---|
| `0 - 5` | White pieces (P, N, B, R, Q, K) | Binary 1.0/0.0 on board squares |
| `6 - 11` | Black pieces (P, N, B, R, Q, K) | Binary 1.0/0.0 on board squares |
| `12` | Side to move | All 1.0 if White, 0.0 if Black |
| `13` | White Kingside castling right | All 1.0 if available, 0.0 otherwise |
| `14` | White Queenside castling right | All 1.0 if available, 0.0 otherwise |
| `15` | Black Kingside castling right | All 1.0 if available, 0.0 otherwise |
| `16` | Black Queenside castling right | All 1.0 if available, 0.0 otherwise |
| `17` | En passant square | 1.0 at en passant square if active |
| `18` | Halfmove clock | Normalized `min(1.0, halfmove / 100.0)` |

Total dimension: $19 \times 8 \times 8 = 1216$. The dimension is accessed dynamically via `encoder.input_dim` without hard-coding across modules.

---

## 3. Temporal-Difference (TD) Learning

### TD(0)
For state transitions $(s_t, r_{t+1}, s_{t+1})$:
$$\text{TD Target: } y_t = r_{t+1} + (1 - d_t) \gamma V_\theta(s_{t+1})$$
$$\text{TD Error: } \delta_t = y_t - V_\theta(s_t)$$
$$\text{Squared Loss: } L_t = \frac{1}{2} \left( V_\theta(s_t) - \operatorname{stopgrad}[y_t] \right)^2$$

### Terminal States
For terminal states ($d_t = 1$, checkmate or draw):
$$y_t = r_{t+1}$$
No bootstrapping occurs from $V(s_{t+1})$. Terminal rewards:
* White checkmates Black: $r = +1.0$
* Black checkmates White: $r = -1.0$
* Stalemate / Draw: $r = 0.0$

### TD(λ)
We provide two mathematically documented formulations:
1. **Forward-View $\lambda$-Return (Trajectory TD($\lambda$))**:
   Computes multi-step geometric returns over complete trajectories:
   $$G_t^\lambda = r_{t+1} + \gamma \left( (1 - \lambda) V(s_{t+1}) + \lambda G_{t+1}^\lambda \right)$$
   When $\lambda = 0$, $G_t^0$ matches TD(0); when $\lambda = 1$, $G_t^1$ matches Monte Carlo returns.
2. **Exact Online Backward-View TD($\lambda$)**:
   Maintains parameter eligibility traces:
   $$e_t = \gamma \lambda e_{t-1} + \nabla_\theta V_\theta(s_t)$$
   $$\theta_{t+1} = \theta_t + \alpha \delta_t e_t$$

---

## 4. Search: Negamax with Alpha-Beta Pruning

The search engine implements Negamax formulation of Minimax:
$$N(s, d, \alpha, \beta) = \max_{a \in \text{Moves}(s)} \left( -N(s_a, d - 1, -\beta, -\alpha) \right)$$
Cutoff occurs whenever $\alpha \ge \beta$.

Move ordering heuristic (MVV-LVA + promotion + checks) is applied at each node to maximize pruning efficiency.

At leaf nodes ($d = 0$ or terminal), $V_{\text{STM}}(s)$ is evaluated via `evaluate_positions(model, [s])` or material baseline.

---

## 5. Dynamic Parameter Growth

The model supports growing hidden dimensions during training (e.g. $4096 \to 8192 \to 16384 \to 32768$) without retraining from scratch:
1. **Incoming Weights**: Existing weight slice $[0:D_{\text{old}}, :]$ is strictly preserved. New incoming connections $[D_{\text{old}}:D_{\text{new}}, :]$ are initialized via Kaiming uniform.
2. **Outgoing Weights (Function Preservation)**: Outgoing connections from newly added neurons to the subsequent layer are initialized to **zero** (Net2Net style):
   $$W_{\text{next}}[:, D_{\text{old}}:D_{\text{new}}] = 0$$
   Therefore, for any input $x$, $f_{\text{new}}(x) \equiv f_{\text{old}}(x)$ at the moment of expansion! Old learned representations remain intact.
3. **Optimizer State**: Adam / SGD momentum buffers are automatically padded to match the enlarged shapes, preserving accumulated velocity.
4. **Trigger Modes**:
   * `manual`: Triggered explicitly via API or CLI.
   * `plateau`: Triggered when the moving average of $|\delta_t|$ fails to improve by `threshold` over `patience` steps.
   * `adaptive`: Triggered dynamically based on TD error convergence.

---

## 6. Sparse Residual Mechanism ("Trống loãng")

A custom structural research mechanism distinct from Dropout:
$$H' = H + S$$
where $H$ is the dense representation from the Huge Layer, and $S$ is a high-capacity sparse residual correction tensor:
$$R = W_{\text{res}} H + b_{\text{res}}$$
$$S = \operatorname{TopK}(R, k), \quad k = \lceil D \times \text{density} \rceil$$
All entries outside the top-$k$ absolute values are zeroed out, strictly guaranteeing:
$$\frac{\text{Nonzero}(S)}{\text{Total}(S)} \le \text{density}$$
*Goal*: Test whether sparse corrections on a dense base allow the Huge Layer to maintain capacity with unconstrained modular regions.

---

## 7. Random Neuron Masking ("Trống neuron học bá")

Temporary random neuron masking during training batches:
$$\tilde h_i = \frac{m_i h_i}{1 - p}, \quad m_i \sim \operatorname{Bernoulli}(1 - p)$$
* In `model.train()`: Randomly samples new masks every batch.
* In `model.eval()`: Deterministic identity (no masking).
* Extensible architecture via `NeuronMaskStrategy` base class (supporting random, activation-magnitude, and usage-frequency masking).
*Goal*: Prevent co-adaptation and over-reliance on individual "genius" neurons, forcing distributed knowledge representation.

---

## 8. Parameter Counting & Precision Formats

The model automatically calculates parameter memory across precisions:
$$\text{Memory} = N_{\text{params}} \times \text{bytes\_per\_parameter}$$
* **FP32**: 4 bytes
* **FP16 / BF16**: 2 bytes
* **INT16**: 2 bytes

### INT16 Serialization / Quantization
Quantization scale $s = \max(|W|) / 32767.0$.
Quantized weights $q = \operatorname{round}(W / s) \in [-32768, 32767]$.
Reconstruction $W \approx q \times s$.

---

## 9. Checkpoint Format: Hugging Face Style (Default) & Auto-Discovery

### Hugging Face Style Checkpoints (`model.safetensors` + `config.json`)
Checkpoint format mặc định là cấu trúc chuẩn của Hugging Face:
```text
runs/
└── run_20260929_212515/
    ├── config.json
    ├── metrics.csv
    ├── summary.json
    └── checkpoints/
        ├── checkpoint-step-000004/
        │   ├── config.json          # Kiến trúc mạng nơ-ron (input_dim, hidden_dims,...)
        │   ├── model.safetensors    # Trọng số định dạng SafeTensors an toàn, zero-copy
        │   ├── trainer_state.json   # Step, episode, growth history, metrics
        │   └── optimizer.pt         # Trạng thái optimizer (Adam momentum,...)
        └── latest/                  # Luôn trỏ/copy checkpoint mới nhất
```

Model cung cấp API chuẩn:
```python
# Lưu model theo kiểu Hugging Face
model.save_pretrained("my_chess_model")

# Load model từ thư mục Hugging Face
model = ChessValueNet.from_pretrained("my_chess_model")
```

### Tự động đặt tên & Tự động phát hiện Checkpoint (Zero-Config)
* **Auto Experiment Naming**: Nếu không truyền `--name`, hệ thống tự động đặt tên theo thời gian: `run_YYYYMMDD_HHMMSS`.
* **Auto Checkpoint Naming**: Checkpoints tự động lưu theo format `checkpoint-step-{step:06d}` và đồng bộ vào thư mục `checkpoints/latest/`.
* **Auto Discovery**: Trong `selfplay` và `evaluate`, nếu **không truyền `--checkpoint`**, engine tự động quét thư mục `runs/` để tìm và nạp checkpoint mới nhất vừa train mà không cần copy paste đường dẫn!

---

## 10. Multi-Source Concurrent Training (Huấn luyện nhiều nguồn CÙNG LÚC)

Mạng nơ-ron có thể hấp thụ tri thức đồng thời từ nhiều nguồn dữ liệu trong cùng một training step với tỷ lệ pha trộn (sampling weights) tùy chỉnh:

```text
  ┌──────────────────────────────────────────────┐
  │ 1. Self-Play Stream (Online / Multi-workers) │──┐ (40%)
  └──────────────────────────────────────────────┘  │
  ┌──────────────────────────────────────────────┐  │
  │ 2. Stockfish Oracle Stream (Distill / Play)  │──┼──► Blended Batch [B, 1216] ──► TD Update
  └──────────────────────────────────────────────┘  │   (30%)
  ┌──────────────────────────────────────────────┐  │
  │ 3. PGN Games Stream (Grandmaster Database)   │──┼──► (20%)
  └──────────────────────────────────────────────┘  │
  ┌──────────────────────────────────────────────┐  │
  │ 4. FEN Benchmarks Stream (Tactics / Puzzles) │──┘ (10%)
  └──────────────────────────────────────────────┘
```

1. **`SelfPlaySource`**: Tự chơi online sinh ra thế cờ mới; hỗ trợ chạy đa luồng ngầm (`--num-selfplay-workers N`) không làm nghẽn GPU.
2. **`StockfishSource`**: Huấn luyện trực tiếp cùng engine cờ vua Stockfish đẳng cấp thế giới:
   * **Mode `"eval"` (Distillation / Value Oracle)**: Stockfish đánh giá thế cờ ở độ sâu `depth` và chuyển đổi centipawns / mate score thành target xác suất thắng $V \in [-1, 1]$:
     $$V = \tanh(0.00184104 \times cp)$$
     với $cp = 400$ (+1 Tốt) tương ứng $V \approx +0.63$, $cp = 1000$ tương ứng $V \approx +0.95$. Mạng nơ-ron học tri thức đánh giá sâu của Stockfish qua TD(0) non-bootstrapping.
   * **Mode `"play"` (Engine Sparring)**: Stockfish tự chơi hoặc đấu tạo ra các trajectory ván đấu siêu phẩm với nhãn kết thúc chuẩn xác.
   * **Safe Fallback**: Tự động phát hiện Stockfish trong PATH hoặc thư mục hệ thống; nếu chưa cài đặt Stockfish binary, engine tự động kích hoạt `MockStockfishEngine` (heuristic vật chất + kiểm soát trung tâm) để đảm bảo 100% không bao giờ crash.
3. **`PGNSource`**: Quét trực tiếp các file `.pgn` chứa hàng vạn ván đấu của đại kiện tướng, tự động gán nhãn kết quả (`1-0`, `0-1`, `1/2-1/2`).
4. **`FENSource`**: Nạp danh sách thế cờ FEN/EPD tĩnh kèm điểm đánh giá ground truth hoặc bài toán chiến thuật.
5. **`MultiSourceDataManager`**: Hòa trộn đồng thời tất cả các luồng dữ liệu theo tỷ lệ cấu hình, cân bằng giữa khả năng khám phá tự do (Self-Play), tri thức cờ người đỉnh cao (PGN) và năng lực tính toán siêu đẳng của Stockfish.

---

## 11. CLI Reference

All commands are run with `uv`:

```bash
# 0. Tải dữ liệu Grandmaster (PGN) và Bộ thế cờ chuẩn (FEN Benchmarks)
uv run python scripts/download_data.py --all
# hoặc tải từng đại kiện tướng yêu thích (fischer, kasparov, carlsen, tal, karpov, capablanca):
uv run python scripts/download_data.py --gm fischer kasparov carlsen --benchmarks

# 1. Train Nhiều Nguồn CÙNG LÚC (Self-Play + Stockfish Distillation + PGN Đại kiện tướng + FEN)
uv run python -m chess_ai.main train \
    --sources selfplay stockfish pgn fen \
    --source-weights 0.4 0.3 0.2 0.1 \
    --stockfish-mode eval \
    --stockfish-depth 8 \
    --pgn-path data/grandmaster/Fischer.pgn \
    --fen-path data/benchmark_positions.fen \
    --num-selfplay-workers 2 \
    --hidden-dim 2048 \
    --growth-mode plateau

# 2. Train Chuyên biệt cùng Stockfish (Oracle Distillation)
uv run python -m chess_ai.main train \
    --sources stockfish \
    --stockfish-mode eval \
    --stockfish-depth 10 \
    --episodes 50

# 3. Benchmarking (Throughput vs Huge Layer Width)
uv run python -m chess_ai.main benchmark \
    --hidden-dims 4096 4096 \
    --batch-sizes 1 16 64 256 1024 \
    --device auto

# 4. Self-Play (Tự động nạp checkpoint mới nhất)
uv run python -m chess_ai.main selfplay \
    --num-games 5 \
    --search-depth 2

# 5. Evaluation (Tự động nạp checkpoint mới nhất và đấu với Material Search)
uv run python -m chess_ai.main evaluate \
    --num-games 10 \
    --search-depth 2

# 6. Đấu hàng loạt 100,000 ván Self-Play & 100,000 ván Stockfish (Đa tiến trình song song)
# Đấu 100k ván Self-Play (Lưu vào data/matches/selfplay_100k.pgn):
uv run python scripts/mass_match.py --mode selfplay --games 100000

# Đấu 100k ván Stockfish (Lưu vào data/matches/stockfish_100k.pgn):
uv run python scripts/mass_match.py --mode stockfish --games 100000

# Đấu cả hai (100k Self-Play + 100k Stockfish = 200,000 ván):
uv run python scripts/mass_match.py --mode all --games 100000
```

---

## 12. Research Hypothesis Matrix

To test the hypothesis cleanly without conflating variables:

| Experiment | Huge Layer | Sparse Residual | Neuron Masking |
|---|---|---|---|
| **A** | Yes | No | No |
| **B** | Yes | Yes ($d = 0.01$) | No |
| **C** | Yes | No | Yes ($p = 0.05$) |
| **D** | Yes | Yes ($d = 0.01$) | Yes ($p = 0.05$) |

> **Important Research Caution**: We do **not** presuppose that more parameters, huge width, sparse residuals, or neuron masking automatically produce a stronger chess player. All hypotheses must be validated through empirical measurement of TD loss convergence, training stability, search depth efficiency, and self-play win rates.
