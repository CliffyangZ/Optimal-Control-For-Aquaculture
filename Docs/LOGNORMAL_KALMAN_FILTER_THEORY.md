# Log-Normal Kalman Filter 理論
## 應用於水產養殖飼料消耗速率估計

---

## 1. 問題定義

### 1.1 信號鏈

YOLO 物件偵測輸出的 bounding box 面積包含顯著噪聲，目標是從噪聲測量值中還原真實飼料面積，並估計其消耗速率：

```
真實飼料面積 xₜ  →  YOLO 偵測  →  測量值 zₜ（含乘法噪聲）
                                       ↓
                               Kalman Filter 估計
                                       ↓
                          估計值 x̂ₜ  +  消耗速率 μ̇ₜ
```

### 1.2 核心問題：YOLO 噪聲的本質

YOLO 的 bounding box 回歸誤差是**比例性的**，而非固定大小的。以 ±10% 誤差為例：

| 真實面積 | 絕對誤差 |
|---------|---------|
| 100 px² | ±10 px² |
| 1,000,000 px² | ±100,000 px² |

這種特性稱為**乘法噪聲（Multiplicative Noise）**：

$$z_t = x_t \cdot \epsilon_t, \quad \epsilon_t > 0$$

標準 Kalman Filter 假設的加法噪聲（$z_t = x_t + v_t$）與此不符，是現有實作的根本缺陷。

---

## 2. 標準 Kalman Filter 的限制

### 2.1 加法噪聲模型

$$z_t = x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$

**問題一：可能估計出負面積**

Gaussian 分布的支撐域為 $(-\infty, +\infty)$，但面積恆為正值。

**問題二：噪聲方差與測量值無關**

加法模型假設無論面積大小，噪聲大小相同，與 YOLO 的實際行為不符。

**問題三：無法直接估計速率**

現有 1D 實作無速率狀態，須事後差分計算，引入額外誤差。

### 2.2 數值驗證

由實驗數據可驗證噪聲的比例性：

$$\frac{\sigma_z}{\bar{z}} = \frac{363{,}691}{659{,}936} \approx 55\%$$

噪聲佔均值的 55%，明顯是相對比例誤差，而非固定加法噪聲。

---

## 3. Log-Normal 分布基礎

### 3.1 定義

若 $\ln X \sim \mathcal{N}(\mu, \sigma^2)$，則稱 $X$ 服從 Log-Normal 分布，記作 $X \sim \text{LogNormal}(\mu, \sigma^2)$。

**重要性質：**

$$\mathbb{E}[X] = e^{\mu + \sigma^2/2}$$

$$\text{Var}[X] = (e^{\sigma^2} - 1) \cdot e^{2\mu + \sigma^2}$$

$$\text{支撐域：} X > 0$$

### 3.2 為什麼面積服從 Log-Normal？

YOLO 偵測的面積可以看成多個乘法因子的乘積：

$$A = A_{\text{true}} \cdot \underbrace{f_{\text{model}}}_{\text{模型不確定性}} \cdot \underbrace{f_{\text{occlusion}}}_{\text{遮擋比例}} \cdot \underbrace{f_{\text{lighting}}}_{\text{光線影響}}$$

根據**對數中心極限定理**，獨立乘法因子的乘積服從 Log-Normal 分布：

$$\ln A = \ln A_{\text{true}} + \sum_i \ln f_i$$

每個 $\ln f_i$ 是小的加法項，其和趨向 Gaussian。

---

## 4. Log-Normal Kalman Filter 推導

### 4.1 變數變換

令：

$$y_t = \ln z_t, \quad \mu_t = \ln x_t$$

對測量方程取對數：

$$\ln z_t = \ln x_t + \ln \epsilon_t$$

$$y_t = \mu_t + v_t, \quad v_t = \ln \epsilon_t \sim \mathcal{N}(0, \sigma_R^2)$$

**乘法噪聲在 log 空間轉換為加法 Gaussian 噪聲**，Kalman Filter 的最優性條件成立。

### 4.2 狀態空間模型

為同時估計面積與消耗速率，定義二維狀態向量：

$$\mathbf{x}_t = \begin{bmatrix} \mu_t \\ \dot{\mu}_t \end{bmatrix} = \begin{bmatrix} \ln(\text{真實面積}_t) \\ \text{log 面積的時間導數} \end{bmatrix}$$

**狀態轉移方程（勻速模型）：**

$$\mathbf{x}_t = F \mathbf{x}_{t-1} + \mathbf{w}_t$$

$$F = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, Q)$$

**觀測方程：**

$$y_t = H \mathbf{x}_t + v_t$$

$$H = \begin{bmatrix} 1 & 0 \end{bmatrix}, \quad v_t \sim \mathcal{N}(0, \sigma_R^2)$$

### 4.3 過程噪聲矩陣 Q

考慮加速度噪聲（Wiener 過程驅動）：

$$Q = \sigma_Q^2 \begin{bmatrix} \Delta t^4 / 4 & \Delta t^3 / 2 \\ \Delta t^3 / 2 & \Delta t^2 \end{bmatrix}$$

其中 $\sigma_Q^2$ 是加速度噪聲強度，控制速率估計的平滑程度。

### 4.4 Kalman Filter 遞推公式

**Predict 步驟：**

$$\hat{\mathbf{x}}_t^- = F \hat{\mathbf{x}}_{t-1}$$

$$P_t^- = F P_{t-1} F^T + Q$$

**Update 步驟：**

$$K_t = P_t^- H^T \left( H P_t^- H^T + \sigma_R^2 \right)^{-1}$$

$$\hat{\mathbf{x}}_t = \hat{\mathbf{x}}_t^- + K_t \left( y_t - H \hat{\mathbf{x}}_t^- \right)$$

$$P_t = (I - K_t H) P_t^-$$

其中 $P_t \in \mathbb{R}^{2 \times 2}$ 為估計誤差協方差矩陣：

$$P_t = \begin{bmatrix} P_{11} & P_{12} \\ P_{21} & P_{22} \end{bmatrix} = \begin{bmatrix} \text{Var}[\mu_t] & \text{Cov}[\mu_t, \dot{\mu}_t] \\ \text{Cov}[\dot{\mu}_t, \mu_t] & \text{Var}[\dot{\mu}_t] \end{bmatrix}$$

---

## 5. 從 Log 空間還原面積估計值

### 5.1 有偏估計（直接 exp）

$$\hat{A}_t^{\text{naive}} = e^{\hat{\mu}_t}$$

此估計**低估**真實均值，因為 Jensen 不等式：$\mathbb{E}[e^X] \geq e^{\mathbb{E}[X]}$。

### 5.2 無偏估計（MVUE）

利用 Log-Normal 的矩公式，加入方差校正項：

$$\hat{A}_t = \exp\!\left( \hat{\mu}_t + \frac{P_{11,t}}{2} \right)$$

其中 $P_{11,t}$ 是 $P_t$ 的 $(1,1)$ 元素，即 $\mu_t$ 的估計方差。

**直觀理解：**
- 當 $P_{11} \to 0$（估計非常確定），校正項消失，趨向直接 exp
- 當 $P_{11}$ 大（初始收斂期），校正項顯著，避免低估

### 5.3 估計區間

$1\sigma$ 置信區間：

$$\left[ \exp\!\left(\hat{\mu}_t - \sqrt{P_{11,t}}\right),\; \exp\!\left(\hat{\mu}_t + \sqrt{P_{11,t}}\right) \right]$$

---

## 6. 消耗速率的物理意義

### 6.1 相對速率的定義

$$\dot{\mu}_t = \frac{d \ln A}{dt} = \frac{1}{A} \frac{dA}{dt}$$

這是**相對消耗率**，單位為 1/frame（或 1/s 若轉換時間軸）。

### 6.2 與絕對速率的比較

| | 絕對速率 $dA/dt$ | 相對速率 $\dot{\mu}_t$ |
|--|----------------|----------------------|
| 定義 | 每幀面積變化量 | 每幀面積變化比例 |
| 投料初期（面積大） | 數值大 | 穩定 |
| 投料末期（面積小） | 數值小 | 穩定 |
| 設定閾值難度 | 難（隨面積變化） | 易（固定比例） |
| 單位 | px²/frame | 1/frame（無因次比） |

**結論：** 相對速率在整個攝食過程中更穩定，適合設定固定的閾值來判斷「魚是否停止進食」。

### 6.3 物理解讀範例

| $\dot{\mu}_t$ 值 | 解讀 |
|----------------|-----|
| $< -0.05$/frame | 飼料快速消耗（積極攝食） |
| $-0.05 \sim -0.01$/frame | 正常消耗 |
| $\approx 0$ | 飼料面積穩定（攝食結束或無飼料） |
| $> 0$ | 面積增加（新投料 或 偵測異常） |

---

## 7. 初始條件設定

### 7.1 初始狀態

$$\hat{\mathbf{x}}_0 = \begin{bmatrix} \ln z_0 \\ 0 \end{bmatrix}$$

以第一幀測量值的 log 作為初始 log 面積，初始速率假設為 0。

> **現有實作的問題：** 以 `initial_value=0` 初始化，而 $\ln 0$ 無定義，實際是以面積=0 開始，導致前數十幀估計嚴重偏低。

### 7.2 初始協方差

$$P_0 = \begin{bmatrix} \sigma_{\mu,0}^2 & 0 \\ 0 & \sigma_{\dot{\mu},0}^2 \end{bmatrix}$$

建議設定：
- $\sigma_{\mu,0}^2 = \sigma_R^2$：初始 log 面積的不確定性等同一次測量誤差
- $\sigma_{\dot{\mu},0}^2 = 1.0$：初始速率完全未知，設較大值讓濾波器快速學習

> **現有實作的問題：** `initial_estimate_error=1` 在原始空間是極小值，導致濾波器初始過度自信於錯誤的初始狀態。

---

## 8. 參數選擇指引

### 8.1 測量噪聲 $\sigma_R^2$

對應 log 空間的測量方差，可從數據估計：

$$\sigma_R^2 \approx \text{Var}[\ln z_t] \text{（靜止場景下的估計）}$$

或利用 YOLO 的 IoU 信心分布推導：若 IoU $\sim 0.7 \pm 0.1$，則 log 面積誤差 $\approx \ln(0.8/0.7) \approx 0.13$，故 $\sigma_R \approx 0.13$。

### 8.2 過程噪聲 $\sigma_Q^2$

控制速率估計的平滑程度：

| $\sigma_Q^2$ | 效果 |
|-------------|-----|
| 小（如 $10^{-4}$） | 速率變化平滑，響應慢 |
| 大（如 $10^{-2}$） | 速率快速跟蹤變化，但噪聲大 |

**建議：** 從 $\sigma_Q^2 = 10^{-3}$ 開始，根據速率曲線的平滑度調整。

### 8.3 Q/R 比值的作用

穩態 Kalman Gain（對 $\mu_t$ 分量）：

$$K_{\infty} \approx \frac{-\sigma_R^2 + \sqrt{\sigma_R^4 + 4\sigma_Q^2 \sigma_R^2}}{2\sigma_Q^2}$$

- $\sigma_Q^2 / \sigma_R^2$ 大 → $K_{\infty}$ 大 → 更信任測量值，響應快但噪聲多
- $\sigma_Q^2 / \sigma_R^2$ 小 → $K_{\infty}$ 小 → 更信任模型，平滑但響應慢

---

## 9. 與現有實作的對比

| 特性 | 現有標準 KF | Log-Normal KF |
|------|-----------|--------------|
| 噪聲模型 | 加法 Gaussian | 乘法 Log-Normal |
| 狀態維度 | 1D（面積） | 2D（log 面積 + 速率） |
| 估計值可能為負 | 是 | 否（exp 恆正） |
| 噪聲與尺度的關係 | 無關 | 比例性（符合 YOLO 特性） |
| 直接輸出消耗速率 | 否 | 是（$\hat{\dot{\mu}}_t$） |
| 初始收斂速度 | 慢（初始值設為 0） | 快（初始值設為 $\ln z_0$） |
| 實作複雜度 | 低（純量運算） | 中（2×2 矩陣運算） |

---

## 10. 實作概要

```python
class LogNormalKalmanFilter:

    def __init__(self, sigma_R, sigma_Q, dt=1.0):
        # 狀態：[log_area, log_area_rate]
        self.x = np.zeros(2)
        self.P = np.eye(2) * sigma_R

        # 狀態轉移矩陣
        self.F = np.array([[1, dt], [0, 1]])

        # 觀測矩陣
        self.H = np.array([[1, 0]])

        # 過程噪聲
        self.Q = sigma_Q * np.array([
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2]
        ])

        # 測量噪聲
        self.R = np.array([[sigma_R]])

        self.initialized = False

    def update(self, area_measurement):
        y = np.log(area_measurement + 1e-9)  # log 變換

        if not self.initialized:
            self.x[0] = y                    # 以第一幀初始化
            self.initialized = True
            return self._to_area()

        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update
        innovation = y - (self.H @ x_pred)[0]
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + (K @ np.array([[innovation]])).flatten()
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return self._to_area()

    def _to_area(self):
        # 無偏估計：exp(μ + P₁₁/2)
        return np.exp(self.x[0] + self.P[0, 0] / 2)

    @property
    def consumption_rate(self):
        # 相對消耗速率 [1/frame]
        return self.x[1]

    @property
    def area_estimate(self):
        return self._to_area()
```

---

## 11. 理論保證

### 11.1 最優性

在以下條件成立時，Log-Normal KF 是 **MMSE（最小均方誤差）估計量**：

1. $\ln \epsilon_t \sim \mathcal{N}(0, \sigma_R^2)$（乘法測量噪聲為 Log-Normal）
2. $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, Q)$（過程噪聲為 Gaussian）
3. 初始狀態與噪聲互相獨立

### 11.2 一致性

當 $\sigma_R^2 \to 0$ 時，$K \to I$，估計值完全跟蹤測量值（無噪聲則無需濾波）。

當 $\sigma_Q^2 \to 0$ 時，$K \to 0$，估計值不更新（模型完全確定則無需測量）。

### 11.3 侷限性

- 若 YOLO 產生大量**離群值**（完全錯誤的偵測），Log-Normal KF 仍會受影響。改進方向為 **Robust KF**（以 Student-t 分布取代 Gaussian）或加入 **Mahalanobis distance** 門檻剔除離群測量值。
- 速率估計依賴**勻速假設**。若攝食行為有突然變化（例如魚群瞬間散開），應考慮加入加速度狀態（3D 狀態向量）或使用 **Interacting Multiple Models (IMM)**。
