# Binomial Probability Mass Function (PMF) & CDF

## 🔗 Problem Link

https://www.tensortonic.com/problems/binomial-pmf-cdf

---

## 📌 Task

Implement the Binomial distribution:

* Probability Mass Function (PMF)
* Cumulative Distribution Function (CDF)

The Binomial distribution models the number of successes in `n` independent Bernoulli trials with probability `p`.

---

## 🧮 Mathematical Definition

### Probability Mass Function (PMF)

P(X = k) = C(n, k) · p^k · (1 - p)^(n - k)

---

### Cumulative Distribution Function (CDF)

P(X ≤ k) = Σ (i = 0 → k) C(n, i) · p^i · (1 - p)^(n - i)

---

## 📥 Input

* `n`: int — number of trials
* `p`: float — probability of success (0 ≤ p ≤ 1)
* `k`: int — number of successes (0 ≤ k ≤ n)

---

## 📤 Output

* Tuple `(pmf, cdf)`
* Both values must be **float**

---

## ✅ Examples

### Example 1

```python id="b1"
Input:
n = 5, p = 0.5, k = 2

Output:
pmf = 0.3125
cdf = 0.5
```

---

### Example 2

```python id="b2"
Input:
n = 10, p = 0.3, k = 0

Output:
pmf = 0.0282
cdf = 0.0282
```

---

### Example 3

```python id="b3"
Input:
n = 8, p = 0.7, k = 8

Output:
pmf = 0.0576
cdf = 1.0
```

---

## ⚙️ Requirements

* Return `(pmf, cdf)`
* Both must be scalar floats
* Use **numerically stable computation**
* Handle edge cases:

  * `k = 0`
  * `k = n`
  * `p = 0`
  * `p = 1`

---

## 💡 Hints

* Use `scipy.special.comb()` for binomial coefficients
* For CDF, sum PMF from `i = 0` to `k`
* Convert results using:

```python id="b4"
float(pmf), float(cdf)
```

---

## ⏱ Constraints

* `0 ≤ k ≤ n ≤ 100`
* `0 ≤ p ≤ 1`
* NumPy + SciPy allowed
* Time limit: `300 ms`

---

## 💡 Notes

This task covers:

* probability distributions
* combinatorics
* numerical stability

---
