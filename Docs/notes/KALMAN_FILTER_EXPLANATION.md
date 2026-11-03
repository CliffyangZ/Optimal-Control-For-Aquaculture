# Log-Normal Kalman Filter for Feed Amount Estimation

## Problem Statement

We need to estimate actual feed area from noisy detection measurements (YOLO/RF-DETR bounding boxes).

**Challenge:** Detection noise is **multiplicative** (proportional to box size), not additive.
- If true area is 100 pixels², error might be ±20% → 80–120 pixels²
- If true area is 10,000 pixels², error might be ±20% → 8,000–12,000 pixels²

## Solution: Log-Normal Kalman Filter

### Key Insight: Log Transform Linearizes the Problem

When noise is multiplicative, **work in log space** to make it additive:

```
z_t = A_t · ε_t          (multiplicative noise in pixel space)
     ↓ apply log
log(z_t) = log(A_t) + log(ε_t)   (additive noise in log space)
```

This satisfies the Kalman Filter's optimality assumption: **additive Gaussian noise**.

---

## Mathematical Framework

### 1️⃣ State Vector (2D)

```
x_t = [ μ_t      ]  where  μ_t = log(A_t)        [log area]
      [ μ̇_t     ]         μ̇_t = d(log A)/dt     [consumption rate, 1/frame]
```

### 2️⃣ System Model (Constant-Velocity in Log Space)

**State Transition:** Assume constant consumption rate
```
x_{t+1} = F · x_t

where F = [ 1   Δt ]     (Δt = 1 frame)
          [ 0   1  ]
```

**Meaning:**
- Tomorrow's log area = today's log area + consumption rate × 1 frame
- Consumption rate stays constant (slow changes due to process noise Q)

### 3️⃣ Observation Model (Measurement)

**Raw Measurement** (from detector):
```
z_t ∈ ℝ  (bounding box area in pixels²)
```

**Transform to Log Space:**
```
y_t = log(z_t) = μ_t + v_t,  where v_t ~ N(0, σ_R²)
```

**Measurement Matrix:**
```
h = [ 1   0 ]   (we only observe log area, not the rate)
```

---

## Algorithm: Two-Phase Update

### Phase 1️⃣: PREDICT (Prior)

**Project state forward** (before seeing new measurement):
```
x̂⁻_{t|t-1} = F · x̂_{t-1|t-1}
P⁻_{t|t-1}  = F · P_{t-1|t-1} · Fᵀ + Q
```

**Interpretation:**
- `x̂⁻`: predicted state (estimate before measurement)
- `P⁻`: predicted covariance (uncertainty before measurement)
- `Q`: process noise (how much the rate might change)

### Phase 2️⃣: UPDATE (Posterior)

**Ingest new measurement** and refine estimate:

**Innovation (residual):**
```
γ_t = y_t - h · x̂⁻_{t|t-1}
    = log(z_t) - μ̂⁻_t
```
("How much does measurement surprise us?")

**Innovation Covariance:**
```
S_t = h · P⁻_{t|t-1} · hᵀ + R
    = σ²_μ + σ²_R
```
(Total uncertainty: prediction uncertainty + measurement uncertainty)

**Kalman Gain:**
```
K_t = (P⁻_{t|t-1} · hᵀ) / S_t
```
(How much to trust the new measurement vs. prediction)

**State Update:**
```
x̂_{t|t} = x̂⁻_{t|t-1} + K_t · γ_t
```
(Blend prediction with measurement)

**Covariance Update:**
```
P_{t|t} = (I - K_t · h) · P⁻_{t|t}
```
(Uncertainty decreases after each measurement)

---

## Parameter Tuning

### σ_R (Measurement Noise Std, Log Space)

Controls how much we trust the detector.

```
σ_R ≈ log(1 + relative_error)

Examples:
  ±10% error  →  σ_R ≈ log(1.1) ≈ 0.095
  ±15% error  →  σ_R ≈ log(1.15) ≈ 0.140   ← typical for YOLO/RF-DETR
  ±20% error  →  σ_R ≈ log(1.2) ≈ 0.182
```

**Effect of σ_R:**
- **Small σ_R** (< 0.1):  Trust detector heavily → noisy filtered output
- **Large σ_R** (> 0.3):  Trust prediction more → sluggish response to real changes

### σ_Q (Process Noise Intensity)

Controls how much we expect the consumption rate to change.

```
Q = σ_Q · [ Δt⁴/4   Δt³/2 ]
          [ Δt³/2   Δt²   ]
```

**Effect of σ_Q:**
- **Small σ_Q** (10⁻⁴):  Rate stays constant → smooth, stable output
- **Large σ_Q** (10⁻²):  Rate can change rapidly → faster tracking, noisier rate estimate

**Recommended:**
- Smooth filtering: `σ_Q ≈ 10⁻³ to 10⁻²`
- Rapid changes: `σ_Q ≈ 10⁻² to 10⁻¹`

---

## Output: Unbiased Area Estimate

After filtering in log space, **convert back to pixel space** with bias correction:

```
Ê[A_t] = exp(μ̂_t + P₁₁/2)
          └─────────┬──────┘
              Log-Normal bias
```

**Why the +P₁₁/2?**

Log-Normal distribution has **bias**: E[e^X] ≠ e^(E[X])

For X ~ N(μ, σ²):
```
E[e^X] = e^(μ + σ²/2)
                ↑
              bias term
```

Without this correction, filtered area is systematically **underestimated** by ~5–10%.

---

## Consumption Rate: d(log A)/dt

The second state component gives instantaneous consumption rate:

```
Rate [%/frame] = μ̇_t × 100
Rate [%/sec]   = μ̇_t × 100 × fps
```

**Interpretation:**
- **μ̇_t = -0.02** → log area decreasing by 2% per frame → feed disappearing
- **μ̇_t = +0.01** → log area increasing by 1% per frame → feed accumulating
- **μ̇_t ≈ 0**    → steady state

**Advantage:** Estimated jointly with area (not by finite difference) → less noisy.

---

## Example: Step-by-Step Walkthrough

### Setup
```
σ_R = 0.15  (±15% detector error)
σ_Q = 0.001 (smooth rate estimate)
Δt  = 1     (1 frame)

x₀ = [log(500), 0]      (initial: 500 px², zero rate)
P₀ = diag(0.15², 1)     (high area uncertainty, huge rate uncertainty)
```

### Frame 1: Measurement z₁ = 480 pixels²

**Predict:** (first call, skip — use measurement to initialize)
```
x̂⁻₁ = x₀ = [6.214, 0]
P⁻₁ = P₀
```

**Initialize from first measurement:**
```
y₁ = log(480) = 6.175

x̂₁ = [6.175, 0]
P₁ = [[0.15², 0], [0, 1]]     (rate still very uncertain)
```

**Unbiased estimate:**
```
Ê[A₁] = exp(6.175 + 0.15²/2) = exp(6.186) ≈ 485 px²
```

### Frame 2: Measurement z₂ = 510 pixels²

**Predict:**
```
x̂⁻₂ = F · x̂₁ = [6.175, 0]   (rate unchanged)
P⁻₂ = F · P₁ · Fᵀ + Q ≈ [[0.0241, 0.001], [0.001, 0.001]]
```

**Innovation:**
```
y₂ = log(510) = 6.234
γ₂ = 6.234 - 6.175 = 0.059
S₂ = 0.0241 + 0.0225 = 0.0466
K₂ = [0.517, 0.0215]ᵀ    (trust measurement fairly well)
```

**Update:**
```
x̂₂ = [6.175, 0] + [0.517, 0.0215] · 0.059
   = [6.205, 0.00127]
   ↑           ↑
   area increased, positive rate detected (feed accumulating)
```

**Unbiased estimate:**
```
Ê[A₂] = exp(6.205 + P₁₁/2) ≈ 499 px²
Rate   = 0.00127 × 100 ≈ 0.127 % / frame
```

---

## Application: Aquaculture Feed Monitoring

### Why This Works

1. **Noise Model Matches Reality**
   - Detector noise IS roughly multiplicative
   - Log transform makes it additive
   - KF is optimal for additive Gaussian noise

2. **Consumption Rate Estimation**
   - Tracked jointly (not post-hoc finite difference)
   - Smoothed by process noise
   - Directly useful for feeding decisions

3. **Unbiased Estimates**
   - Log-Normal bias correction
   - Confidence intervals from covariance
   - Honest uncertainty quantification

### Example Output

```
Frame 100:  Area = 320 px²  (±15 px² 1σ band)
            Rate = -0.85 %/frame  (feed disappearing at ~1%/sec)

Frame 200:  Area = 80 px²   (±8 px² 1σ band)
            Rate = -1.2 %/frame  (accelerating consumption)

Frame 300:  Area = 5 px²    (±1 px² 1σ band)
            Rate ≈ 0       (plateau: all feed consumed)
```

→ **Decision:** Refill when area drops below 10 px².

---

## Summary: The Complete Algorithm

| Step | Operation | Math |
|------|-----------|------|
| **Init** | Seed from first measurement | x̂ = [log(z₀), 0], P = diag(σ_R², 1) |
| **Predict** | Project state forward | x̂⁻ = F · x̂, P⁻ = F·P·Fᵀ + Q |
| **Innovate** | Compute measurement surprise | γ = y - h·x̂⁻, S = h·P⁻·hᵀ + R |
| **Gain** | Compute trust in measurement | K = (P⁻·hᵀ) / S |
| **Update** | Blend prediction + measurement | x̂ = x̂⁻ + K·γ, P = (I - K·h)·P⁻ |
| **Output** | Convert to pixel space | Ê[A] = exp(μ̂ + P₁₁/2) |

---

## References & Further Reading

- **Kalman, R. E. (1960).** "A new approach to linear filtering and prediction problems." ASME Journal.
- **Bar-Shalom, Y., Li, X-R., & Kirubarajan, T. (2001).** *Estimation with Applications to Tracking and Navigation.*
- **Multiplicative Noise:** Working in log space converts multiplicative noise to additive.
- **Log-Normal Bias:** Always apply bias correction when converting from log to linear space.
