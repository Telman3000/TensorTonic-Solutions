# Logistic Regression Training (Gradient Descent)

## 🔗 Problem Link

https://www.tensortonic.com/problems/logistic-regression-training

---

## 📌 Task

Train a binary logistic regression classifier using gradient descent and return the learned parameters `(w, b)`.

---

## 🧮 Model

p = σ(Xw + b)

σ(x) = 1 / (1 + e^(−x))

---

## 🧮 Loss Function (Binary Cross-Entropy)

L = −(1/N) Σ [yᵢ log(pᵢ) + (1 − yᵢ) log(1 − pᵢ)]

---

## 📥 Input

* `X`: NumPy array of shape `(N, D)`
* `y`: NumPy array of shape `(N,)`
* `lr`: float — learning rate
* `steps`: int — number of iterations

---

## 📤 Output

* Tuple `(w, b)`

  * `w`: NumPy array of shape `(D,)`
  * `b`: float

---

## 🔄 Explanation

Training loop:

1. Compute linear combination:

```python id="l1"
z = X @ w + b
```

2. Apply sigmoid:

```python id="l2"
p = sigmoid(z)
```

3. Compute error:

```python id="l3"
error = p - y
```

4. Compute gradients:

```python id="l4"
dw = X.T @ error / N
db = mean(error)
```

5. Update parameters:

```python id="l5"
w = w - lr * dw
b = b - lr * db
```

Repeat for `steps` iterations.

---

## ✅ Example

```python id="l6"
Input:
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
lr = 0.1
steps = 500

Expected:
Accuracy ≥ 95%
```

---

## ⚙️ Requirements

* Initialize:

  * `w = zeros(D)`
  * `b = 0.0`
* Use **gradient descent**
* Return `(w, b)`
* ❌ Do NOT use:

  * sklearn
  * external ML libraries
* ✅ Use NumPy only

---

## 💡 Hints

* Gradients:

```python
dw = X.T @ (p - y) / N
db = mean(p - y)
```

* Updates:

```python
w -= lr * dw
b -= lr * db
```

---

## ⏱ Constraints

* `N ≤ 200`, `D ≤ 10`
* `lr > 0`, `steps ≥ 1`
* Time limit: `1000 ms`
* Memory: `128 MB`

---

## 💡 Notes

This task covers:

* logistic regression from scratch
* gradient descent optimization
* vectorized NumPy operations
* binary classification fundamentals

---
