# Sigmoid Activation Function (NumPy)

## 🔗 Problem Link

https://www.tensortonic.com/problems/sigmoid-numpy

---

## 📌 Task

Implement the sigmoid activation function:

σ(x) = 1 / (1 + e^(−x))

The function must work with scalars, Python lists, and NumPy arrays.

---

## 🧮 Mathematical Definition

σ(x) = 1 / (1 + e^(−x))

---

## 📥 Input

* `x`: scalar, list, or NumPy array

---

## 📤 Output

* NumPy array of floats with the same shape as input

---

## 🔄 Explanation

Sigmoid maps any real number to the range `(0, 1)`:

* large negative → close to 0
* 0 → 0.5
* large positive → close to 1

---

## ✅ Examples

### Example 1

```python id="s1"
Input:
x = [0, 2, -2]

Output:
[0.5, 0.88079708, 0.11920292]
```

---

### Example 2

```python id="s2"
Input:
x = 0

Output:
0.5
```

---

### Example 3

```python id="s3"
Input:
x = [[-1, 0],
     [1, 2]]

Output:
[[0.26894142, 0.5],
 [0.73105858, 0.88079708]]
```

---

## ⚙️ Requirements

* Must work with:

  * scalar
  * Python list
  * NumPy array
* Return **NumPy array of floats**
* Must be **vectorized**
* ❌ No Python loops
* ✅ Use NumPy operations

---

## 💡 Hints

* Convert input:

```python
np.asarray(x, dtype=float)
```

* Use:

```python
np.exp(-x)
```

---

## ⏱ Constraints

* Vectorized implementation only
* Time limit: `200 ms`
* Memory limit: `64 MB`
* Allowed library: NumPy only

---

## 💡 Notes

This task introduces:

* activation functions in ML
* vectorized NumPy operations
* handling different input types

---
