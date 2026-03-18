# Matrix Transpose

## 🔗 Problem Link

https://www.tensortonic.com/problems/matrix-transpose

---

## 📌 Task

Implement the transpose of a matrix, where each element at position `(i, j)` is swapped to `(j, i)`.

---

## 🧮 Mathematical Definition

Transpose operation:

(Aᵀ)ⱼᵢ = Aᵢⱼ

An `N × M` matrix becomes an `M × N` matrix.

---

## 📥 Input

* `A`: 2D NumPy array of shape `(N, M)`

---

## 📤 Output

* New NumPy array of shape `(M, N)`

---

## 🔄 Explanation

Transpose swaps rows and columns:

```python
A[i][j] → Aᵀ[j][i]
```

---

## ✅ Examples

### Example 1

```python
Input:
A = [[1, 2, 3],
     [4, 5, 6]]

Output:
[[1, 4],
 [2, 5],
 [3, 6]]
```

---

### Example 2

```python
Input:
A = [[1, 2],
     [3, 4]]

Output:
[[1, 3],
 [2, 4]]
```

---

### Example 3

```python
Input:
A = [[1, 2, 3, 4]]

Output:
[[1],
 [2],
 [3],
 [4]]
```

---

## ⚙️ Requirements

* Return a **new NumPy array**
* Do **not modify** the original matrix
* Must work for **rectangular matrices**
* ❌ Do NOT use:

  * `.T`
  * `np.transpose()`
* ✅ Use:

  * loops or manual indexing

---

## 💡 Hint

Create a new array with shape `(M, N)` using `np.zeros()`, then fill it using a nested loop.

---

## ⏱ Constraints

* `1 ≤ N, M ≤ 1000`
* Elements can be `int` or `float`
* Time limit: `200 ms`

---

## 💡 Notes

This task tests understanding of:

* matrix structure
* indexing
* basic linear algebra operations

---
