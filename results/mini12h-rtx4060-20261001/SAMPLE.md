# Sample experiment: 111-question teacher run

Run `mini12h-rtx4060-20261001-230236`: Qwen2.5-Math-7B-Instruct (NF4/FP16) on an RTX 4060 Laptop GPU,
one greedy answer plus eight seeded samples (temperature 0.7, top-k 50) per
question, 1,024-token cap, seed 42. Aggregate metrics and reliability plots
are in [`report.md`](report.md); every row below is also in
[`sample_rows.csv`](sample_rows.csv) and, with the full greedy solution text,
[`sample_rows.jsonl`](sample_rows.jsonl).

## Overview

| Dataset | Questions | Scorable | Greedy correct | Majority correct | Mean agreement (correct) | Mean agreement (wrong) |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 40 | 40 | 34 (85.0%) | 36 (90.0%) | 0.97 | 0.54 |
| MATH | 21 | 21 | 15 (71.4%) | 17 (81.0%) | 0.85 | 0.17 |
| GSM-Plus | 50 | 48 | 39 (81.2%) | 40 (83.3%) | 0.97 | 0.60 |

Agreement share is the fraction of the eight samples whose answer matches
the greedy answer. Wrong answers have much lower agreement on average, which
is why self-consistency is a useful (though exploratory) correctness signal.

## Worked examples

### Confident and correct

`math:precalculus:test:51` (MATH)

> A line is parameterized by a parameter $t,$ so that the vector on the line at $t = -2$ is $\begin{pmatrix} 2 \\ -4 \end{pmatrix},$ and the vector on the line at $t = 3$ is $\begin{pmatrix} 1 \\ 7 \end{pmatrix}.$  Find the vector on the line at $t = 5.$

- Gold answer: `\begin{pmatrix} 3/5 \\ 57/5 \end{pmatrix}`; teacher greedy answer: `\begin{pmatrix} \frac{3}{5} \\ \frac{57}{5} \end{pmatrix}` (correct)
- Entropy confidence 0.997, sequence probability 0.986, agreement share 1.0 (majority `\begin{pmatrix} \frac{3}{5} \\ \frac{57}{5} \end{pmatrix}` at 1.0)
- Eight sampled answers: `\begin{pmatrix} \frac{3}{5} \\ \frac{57}{5} \end{pmatrix}`, `\begin{pmatrix} \frac{3}{5} \\ \frac{57}{5} \end{pmatrix}`, `\begin{pmatrix} \frac{3}{5} \\ \frac{57}{5} \end{pmatrix}`, `\begin{pmatrix} \frac{3}{5} \\ \frac{57}{5} \end{pmatrix}`, `\begin{pmatrix} \frac{3}{5} \\ \frac{57}{5} \end{pmatrix}`, `\begin{pmatrix} \frac{3}{5} \\ \frac{57}{5} \end{pmatrix}`, `\begin{pmatrix} \frac{3}{5} \\ \frac{57}{5} \end{pmatrix}`, `\begin{pmatrix} \frac{3}{5} \\ \frac{57}{5} \end{pmatrix}`

<details><summary>Teacher greedy solution</summary>

```text
To find the vector on the line at \( t = 5 \), we first need to determine the direction vector of the line. The direction vector can be found by subtracting the vector at \( t = -2 \) from the vector at \( t = 3 \) and then dividing by the difference in the parameter values, which is \( 3 - (-2) = 5 \).

The direction vector is:
\[
\begin{pmatrix} 1 \\ 7 \end{pmatrix} - \begin{pmatrix} 2 \\ -4 \end{pmatrix} = \begin{pmatrix} 1 - 2 \\ 7 - (-4) \end{pmatrix} = \begin{pmatrix} -1 \\ 11 \end{pmatrix}
\]
Since this direction vector corresponds to a change in \( t \) of 5, the direction vector per unit change in \( t \) is:
\[
\frac{1}{5} \begin{pmatrix} -1 \\ 11 \end{pmatrix} = \begin{pmatrix} -\frac{1}{5} \\ \frac{11}{5} \end{pmatrix}
\]
This means that for each unit increase in \( t \), the vector on the line changes by \( \begin{pmatrix} -\frac{1}{5} \\ \frac{11}{5} \end{pmatrix} \).

To find the vector on the line at \( t = 5 \), we start with the vector at \( t = -2 \) and add the change in the vector from \( t = -2 \) to \( t = 5 \). The change in \( t \) from \( -2 \) to \( 5 \) is \( 5 - (-2) = 7 \). Therefore, the change in the vector is:
\[
7 \begin{pmatrix} -\frac{1}{5} \\ \f
…(truncated for display)
```

</details>

### Wrong, and flagged by low agreement

`math:intermediate_algebra:test:607` (MATH)

> The roots of the equation
> \[x^5 - 40x^4 + Px^3 + Qx^2 + Rx + S = 0\]are in geometric progression.  The sum of their reciprocals is 10.  Compute $|S|.$

- Gold answer: `32`; teacher greedy answer: `\frac{59049}{100000}` (wrong)
- Entropy confidence 0.975, sequence probability 0.935, agreement share 0.0 (majority `32` at 0.75)
- Eight sampled answers: `32`, `32`, `32`, `32`, `32`, `32`, `1024`, `None`

<details><summary>Teacher greedy solution</summary>

```text
Given the polynomial equation \(x^5 - 40x^4 + Px^3 + Qx^2 + Rx + S = 0\) with roots in geometric progression, let the roots be \(a, ar, ar^2, ar^3, ar^4\). By Vieta's formulas, the sum of the roots is equal to the coefficient of \(x^4\) with a negative sign, so we have:
\[
a + ar + ar^2 + ar^3 + ar^4 = 40.
\]
Factoring out \(a\), we get:
\[
a(1 + r + r^2 + r^3 + r^4) = 40.
\]
The sum of the reciprocals of the roots is given by:
\[
\frac{1}{a} + \frac{1}{ar} + \frac{1}{ar^2} + \frac{1}{ar^3} + \frac{1}{ar^4} = \frac{1}{a} \left(1 + \frac{1}{r} + \frac{1}{r^2} + \frac{1}{r^3} + \frac{1}{r^4}\right) = \frac{1}{a} \cdot \frac{r^4 + r^3 + r^2 + r + 1}{r^4} = \frac{1 + r + r^2 + r^3 + r^4}{ar^4} = 10.
\]
From this, we can solve for \(a\):
\[
\frac{1 + r + r^2 + r^3 + r^4}{ar^4} = 10 \implies a(1 + r + r^2 + r^3 + r^4) = \frac{1 + r + r^2 + r^3 + r^4}{10} \implies a = \frac{1}{10}.
\]
Substituting \(a = \frac{1}{10}\) back into the equation \(a(1 + r + r^2 + r^3 + r^4) = 40\), we get:
\[
\frac{1}{10}(1 + r + r^2 + r^3 + r^4) = 40 \implies 1 + r + r^2 + r^3 + r^4 = 400.
\]
The constant term \(S\) of the polynomial is the product of the roots with a negative sign, so:
\[
S = -a \cdot ar \cd
…(truncated for display)
```

</details>

### Wrong despite high agreement (a gate would miss it)

`gsm8k:main:test:403` (GSM8K)

> Mel uses a 900-watt air conditioner for 8 hours a day. This means that each hour the AC uses 900 watts of energy. If he reduces the time he uses the air conditioner by 5 hours a day, how many kilowatts of electric energy will he save in 30 days?

- Gold answer: `81`; teacher greedy answer: `135` (wrong)
- Entropy confidence 0.997, sequence probability 0.987, agreement share 1.0 (majority `135` at 1.0)
- Eight sampled answers: `135`, `135`, `135`, `135`, `135`, `135`, `135`, `135`

<details><summary>Teacher greedy solution</summary>

```text
To determine how many kilowatts of electric energy Mel will save in 30 days by reducing the time he uses the air conditioner, we need to follow these steps:

1. Calculate the daily energy consumption of the air conditioner before the reduction.
2. Calculate the daily energy consumption of the air conditioner after the reduction.
3. Determine the daily energy savings.
4. Calculate the total energy savings over 30 days.
5. Convert the total energy savings from watts to kilowatts.

**Step 1: Calculate the daily energy consumption of the air conditioner before the reduction.**

The air conditioner uses 900 watts per hour and is used for 8 hours a day.
\[ 900 \text{ watts/hour} \times 8 \text{ hours/day} = 7200 \text{ watts/day} \]

**Step 2: Calculate the daily energy consumption of the air conditioner after the reduction.**

Mel reduces the time he uses the air conditioner by 5 hours a day, so he now uses it for \(8 - 5 = 3\) hours a day.
\[ 900 \text{ watts/hour} \times 3 \text{ hours/day} = 2700 \text{ watts/day} \]

**Step 3: Determine the daily energy savings.**

The daily energy savings is the difference between the daily energy consumption before and after the reduction.
\[ 7200
…(truncated for display)
```

</details>

## All 111 rows

✓ = correct, ✗ = wrong, — = no gold answer. Entropy = entropy confidence;
Agree = agreement share; Maj = majority vote share.

### GSM8K

| # | Example | Subject / perturbation | Gold | Greedy | ✓ | Entropy | Agree | Majority | Maj | ✓ | Tokens |
|---:|---|---|---|---|:-:|---:|---:|---|---:|:-:|---:|
| 1 | `1028` | — | 200 | 200 | ✓ | 0.994 | 1.0 | 200 | 1.0 | ✓ | 210 |
| 2 | `1048` | — | 7400 | 9840.78 | ✗ | 0.988 | 0.125 | 7400 | 0.625 | ✓ | 497 |
| 3 | `1066` | — | 14 | 14 | ✓ | 0.996 | 1.0 | 14 | 1.0 | ✓ | 360 |
| 4 | `11` | — | 694 | 694 | ✓ | 0.995 | 1.0 | 694 | 1.0 | ✓ | 270 |
| 5 | `1147` | — | 22 | 22 | ✓ | 0.994 | 1.0 | 22 | 1.0 | ✓ | 310 |
| 6 | `115` | — | 90 | 90 | ✓ | 0.995 | 0.875 | 90 | 0.875 | ✓ | 624 |
| 7 | `1231` | — | 70 | 70 | ✓ | 0.987 | 0.875 | 70 | 0.875 | ✓ | 319 |
| 8 | `1244` | — | 35 | 35 | ✓ | 0.993 | 1.0 | 35 | 1.0 | ✓ | 380 |
| 9 | `1258` | — | 60 | 60 | ✓ | 0.995 | 1.0 | 60 | 1.0 | ✓ | 278 |
| 10 | `1264` | — | 210 | 210 | ✓ | 0.994 | 1.0 | 210 | 1.0 | ✓ | 280 |
| 11 | `1273` | — | 27 | 27 | ✓ | 0.975 | 0.625 | 27 | 0.625 | ✓ | 255 |
| 12 | `164` | — | 15 | 15 | ✓ | 0.987 | 1.0 | 15 | 1.0 | ✓ | 184 |
| 13 | `168` | — | 18 | 18 | ✓ | 0.991 | 1.0 | 18 | 1.0 | ✓ | 215 |
| 14 | `184` | — | 25 | 25 | ✓ | 0.996 | 1.0 | 25 | 1.0 | ✓ | 469 |
| 15 | `211` | — | 4 | 4 | ✓ | 0.994 | 1.0 | 4 | 1.0 | ✓ | 253 |
| 16 | `213` | — | 250 | 250 | ✓ | 0.994 | 1.0 | 250 | 1.0 | ✓ | 301 |
| 17 | `229` | — | 21 | 21 | ✓ | 0.994 | 1.0 | 21 | 1.0 | ✓ | 390 |
| 18 | `257` | — | 5600 | 5600 | ✓ | 0.995 | 1.0 | 5600 | 1.0 | ✓ | 398 |
| 19 | `264` | — | 400 | 400 | ✓ | 0.996 | 1.0 | 400 | 1.0 | ✓ | 268 |
| 20 | `267` | — | 91 | 51 | ✗ | 0.988 | 0.375 | 61 | 0.625 | ✗ | 365 |
| 21 | `312` | — | 32 | 32 | ✓ | 0.995 | 1.0 | 32 | 1.0 | ✓ | 158 |
| 22 | `321` | — | 1 | 1 | ✓ | 0.996 | 1.0 | 1 | 1.0 | ✓ | 169 |
| 23 | `322` | — | 9 | 10 | ✗ | 0.991 | 0.25 | 9 | 0.75 | ✓ | 435 |
| 24 | `349` | — | 2640 | 2640 | ✓ | 0.992 | 1.0 | 2640 | 1.0 | ✓ | 280 |
| 25 | `403` | — | 81 | 135 | ✗ | 0.997 | 1.0 | 135 | 1.0 | ✗ | 528 |
| 26 | `458` | — | 35 | 35 | ✓ | 0.997 | 1.0 | 35 | 1.0 | ✓ | 294 |
| 27 | `47` | — | 800 | 800 | ✓ | 0.994 | 1.0 | 800 | 1.0 | ✓ | 364 |
| 28 | `51` | — | 5 | 5 | ✓ | 0.992 | 1.0 | 5 | 1.0 | ✓ | 266 |
| 29 | `528` | — | 172 | 172 | ✓ | 0.993 | 0.875 | 172 | 0.875 | ✓ | 302 |
| 30 | `532` | — | 25 | 25 | ✓ | 0.994 | 1.0 | 25 | 1.0 | ✓ | 353 |
| 31 | `539` | — | 35 | 30 | ✗ | 0.989 | 0.75 | 30 | 0.75 | ✗ | 231 |
| 32 | `548` | — | 100 | 100 | ✓ | 0.992 | 1.0 | 100 | 1.0 | ✓ | 210 |
| 33 | `555` | — | 2 | 2 | ✓ | 0.997 | 1.0 | 2 | 1.0 | ✓ | 320 |
| 34 | `564` | — | 240 | 240 | ✓ | 0.994 | 0.875 | 240 | 0.875 | ✓ | 394 |
| 35 | `634` | — | 14 | 14 | ✓ | 0.994 | 1.0 | 14 | 1.0 | ✓ | 209 |
| 36 | `688` | — | 100 | 100 | ✓ | 0.993 | 1.0 | 100 | 1.0 | ✓ | 223 |
| 37 | `729` | — | 1000 | 1000 | ✓ | 0.993 | 1.0 | 1000 | 1.0 | ✓ | 597 |
| 38 | `878` | — | 5 | 5 | ✓ | 0.99 | 1.0 | 5 | 1.0 | ✓ | 247 |
| 39 | `952` | — | 360 | 1800 | ✗ | 0.985 | 0.75 | 1800 | 0.75 | ✗ | 384 |
| 40 | `990` | — | 14 | 14 | ✓ | 0.993 | 1.0 | 14 | 1.0 | ✓ | 393 |

### MATH

| # | Example | Subject / perturbation | Gold | Greedy | ✓ | Entropy | Agree | Majority | Maj | ✓ | Tokens |
|---:|---|---|---|---|:-:|---:|---:|---|---:|:-:|---:|
| 1 | `450` | algebra | 13 | 13 | ✓ | 0.99 | 1.0 | 13 | 1.0 | ✓ | 634 |
| 2 | `782` | algebra | \$32,\!348 | 32349 | ✗ | 0.975 | 0.75 | 32349 | 0.75 | ✗ | 501 |
| 3 | `907` | algebra | -15 | -15 | ✓ | 0.993 | 1.0 | -15 | 1.0 | ✓ | 205 |
| 4 | `290` | counting_and_probability | 27 | 27 | ✓ | 0.994 | 1.0 | 27 | 1.0 | ✓ | 294 |
| 5 | `368` | counting_and_probability | 110 | 110 | ✓ | 0.991 | 0.625 | 110 | 0.625 | ✓ | 564 |
| 6 | `445` | counting_and_probability | 35 | — | ✗ | 0.992 | 0.0 | 35 | 0.875 | ✓ | 1024 |
| 7 | `174` | geometry | 162 | 324 | ✗ | 0.984 | 0.125 | 270 | 0.375 | ✗ | 673 |
| 8 | `381` | geometry | 8 | 8 | ✓ | 0.996 | 0.375 | — | 0.625 | ✗ | 926 |
| 9 | `84` | geometry | 12\pi | 12\pi | ✓ | 0.991 | 1.0 | 12\pi | 1.0 | ✓ | 348 |
| 10 | `415` | intermediate_algebra | -50 | -50 | ✓ | 0.995 | 0.5 | -50 | 0.5 | ✓ | 935 |
| 11 | `607` | intermediate_algebra | 32 | \frac{59049}{100000} | ✗ | 0.975 | 0.0 | 32 | 0.75 | ✓ | 882 |
| 12 | `863` | intermediate_algebra | 50 | 50 | ✓ | 0.992 | 0.75 | 50 | 0.75 | ✓ | 235 |
| 13 | `118` | number_theory | 0 | 0 | ✓ | 0.962 | 0.5 | 0 | 0.5 | ✓ | 485 |
| 14 | `245` | number_theory | 5 | 5 | ✓ | 0.993 | 1.0 | 5 | 1.0 | ✓ | 397 |
| 15 | `401` | number_theory | 4 | 4 | ✓ | 0.994 | 1.0 | 4 | 1.0 | ✓ | 348 |
| 16 | `474` | prealgebra | 29 | — | ✗ | 0.348 | 0.0 | 29 | 0.375 | ✓ | 1024 |
| 17 | `606` | prealgebra | 1 | 1 | ✓ | 0.994 | 1.0 | 1 | 1.0 | ✓ | 427 |
| 18 | `811` | prealgebra | 140 | 140 | ✓ | 0.991 | 1.0 | 140 | 1.0 | ✓ | 228 |
| 19 | `160` | precalculus | 19 | 19 | ✓ | 0.996 | 1.0 | 19 | 1.0 | ✓ | 432 |
| 20 | `51` | precalculus | \begin{pmatrix} 3/5 \\ 57… | \begin{pmatrix} \frac{3}{… | ✓ | 0.997 | 1.0 | \begin{pmatrix} \frac{3}{… | 1.0 | ✓ | 736 |
| 21 | `61` | precalculus | 3R^2 | \frac{a^2 + b^2 + c^2}{3} | ✗ | 0.983 | 0.125 | \frac{3R^2}{4} | 0.25 | ✗ | 446 |

### GSM-Plus

| # | Example | Subject / perturbation | Gold | Greedy | ✓ | Entropy | Agree | Majority | Maj | ✓ | Tokens |
|---:|---|---|---|---|:-:|---:|---:|---|---:|:-:|---:|
| 1 | `10030` | distraction insertion | 120 | 120 | ✓ | 0.994 | 0.875 | 120 | 0.875 | ✓ | 303 |
| 2 | `10098` | integer-decimal-fraction … | 12 | 12 | ✓ | 0.991 | 1.0 | 12 | 1.0 | ✓ | 336 |
| 3 | `10212` | reversing operation | 15 | 15 | ✓ | 0.993 | 1.0 | 15 | 1.0 | ✓ | 242 |
| 4 | `1251` | adding operation | 1540 | 2870 | ✗ | 0.957 | 0.0 | 1470 | 0.375 | ✗ | 635 |
| 5 | `1852` | reversing operation | 2 | 2 | ✓ | 0.993 | 1.0 | 2 | 1.0 | ✓ | 230 |
| 6 | `1862` | distraction insertion | 75 | 85 | ✗ | 0.993 | 1.0 | 85 | 1.0 | ✗ | 299 |
| 7 | `1932` | reversing operation | 22 | 2 | ✗ | 0.973 | 1.0 | 2 | 1.0 | ✗ | 352 |
| 8 | `211` | adding operation | 233 | 233 | ✓ | 0.994 | 1.0 | 233 | 1.0 | ✓ | 361 |
| 9 | `2238` | distraction insertion | 6 | 6 | ✓ | 0.992 | 1.0 | 6 | 1.0 | ✓ | 222 |
| 10 | `2401` | digit expansion | 780 | 780 | ✓ | 0.993 | 1.0 | 780 | 1.0 | ✓ | 234 |
| 11 | `2500` | reversing operation | 30 | 30 | ✓ | 0.996 | 1.0 | 30 | 1.0 | ✓ | 314 |
| 12 | `2579` | adding operation | 10 | 10 | ✓ | 0.994 | 0.75 | 10 | 0.75 | ✓ | 285 |
| 13 | `258` | integer-decimal-fraction … | 35 | 35 | ✓ | 0.994 | 1.0 | 35 | 1.0 | ✓ | 304 |
| 14 | `2999` | critical thinking | — | \frac{3n}{2} | — | 0.994 | 0.625 | \frac{3n}{2} | 0.625 | — | 292 |
| 15 | `30` | distraction insertion | 540 | 540 | ✓ | 0.992 | 1.0 | 540 | 1.0 | ✓ | 212 |
| 16 | `3084` | reversing operation | 52 | 52 | ✓ | 0.986 | 1.0 | 52 | 1.0 | ✓ | 389 |
| 17 | `3691` | adding operation | 12 | 12 | ✓ | 0.993 | 1.0 | 12 | 1.0 | ✓ | 275 |
| 18 | `3882` | integer-decimal-fraction … | 237.6 | 237.6 | ✓ | 0.992 | 1.0 | 237.6 | 1.0 | ✓ | 251 |
| 19 | `3932` | reversing operation | 64 | 64 | ✓ | 0.995 | 1.0 | 64 | 1.0 | ✓ | 238 |
| 20 | `3986` | integer-decimal-fraction … | 40 | 31 | ✗ | 0.995 | 0.625 | 31 | 0.625 | ✗ | 441 |
| 21 | `4250` | integer-decimal-fraction … | 70 | 70 | ✓ | 0.995 | 0.625 | 70 | 0.625 | ✓ | 316 |
| 22 | `4535` | critical thinking | — | — | — | 0.288 | 0.0 | — | 0.875 | — | 1024 |
| 23 | `4862` | distraction insertion | 10 | 11.67 | ✗ | 0.972 | 0.0 | — | 0.375 | ✗ | 420 |
| 24 | `5204` | reversing operation | 20 | 20 | ✓ | 0.991 | 0.75 | 20 | 0.75 | ✓ | 415 |
| 25 | `5278` | distraction insertion | 3 | 3 | ✓ | 0.994 | 1.0 | 3 | 1.0 | ✓ | 560 |
| 26 | `5725` | problem understanding | 385000 | 385000 | ✓ | 0.994 | 1.0 | 385000 | 1.0 | ✓ | 318 |
| 27 | `5840` | numerical substitution | 7 | 7 | ✓ | 0.995 | 1.0 | 7 | 1.0 | ✓ | 287 |
| 28 | `5860` | reversing operation | 15 | 15 | ✓ | 0.994 | 1.0 | 15 | 1.0 | ✓ | 280 |
| 29 | `5891` | adding operation | 47 | 46 | ✗ | 0.91 | 0.75 | 46 | 0.75 | ✗ | 328 |
| 30 | `6073` | digit expansion | 60000000 | 60000000 | ✓ | 0.991 | 1.0 | 60000000 | 1.0 | ✓ | 257 |
| 31 | `6610` | integer-decimal-fraction … | 39 | 39 | ✓ | 0.994 | 1.0 | 39 | 1.0 | ✓ | 280 |
| 32 | `6878` | distraction insertion | 1520 | 1520 | ✓ | 0.992 | 1.0 | 1520 | 1.0 | ✓ | 344 |
| 33 | `6974` | distraction insertion | 48 | 48 | ✓ | 0.996 | 1.0 | 48 | 1.0 | ✓ | 485 |
| 34 | `7090` | integer-decimal-fraction … | 20 | 20 | ✓ | 0.994 | 1.0 | 20 | 1.0 | ✓ | 302 |
| 35 | `7181` | problem understanding | 1 | 1 | ✓ | 0.991 | 1.0 | 1 | 1.0 | ✓ | 343 |
| 36 | `7437` | problem understanding | 60 | 60 | ✓ | 0.997 | 1.0 | 60 | 1.0 | ✓ | 351 |
| 37 | `7733` | problem understanding | 48 | 48 | ✓ | 0.995 | 1.0 | 48 | 1.0 | ✓ | 226 |
| 38 | `7779` | adding operation | 180 | 180 | ✓ | 0.99 | 0.875 | 180 | 0.875 | ✓ | 346 |
| 39 | `7962` | integer-decimal-fraction … | 10.1 | 10.1 | ✓ | 0.992 | 1.0 | 10.1 | 1.0 | ✓ | 227 |
| 40 | `8235` | adding operation | 49 | 49 | ✓ | 0.992 | 0.875 | 49 | 0.875 | ✓ | 396 |
| 41 | `8733` | problem understanding | 1000 | 1000 | ✓ | 0.995 | 1.0 | 1000 | 1.0 | ✓ | 205 |
| 42 | `8805` | problem understanding | 5 | 5 | ✓ | 0.995 | 1.0 | 5 | 1.0 | ✓ | 267 |
| 43 | `8862` | distraction insertion | 360 | — | ✗ | 0.307 | 0.0 | 360 | 0.875 | ✓ | 1024 |
| 44 | `8953` | digit expansion | 180 | 180 | ✓ | 0.996 | 1.0 | 180 | 1.0 | ✓ | 354 |
| 45 | `899` | adding operation | 32 | 24 | ✗ | 0.988 | 1.0 | 24 | 1.0 | ✗ | 221 |
| 46 | `9125` | problem understanding | 27 | 27 | ✓ | 0.98 | 1.0 | 27 | 1.0 | ✓ | 374 |
| 47 | `9608` | numerical substitution | 44 | 55 | ✗ | 0.976 | 1.0 | 55 | 1.0 | ✗ | 293 |
| 48 | `9896` | numerical substitution | 87 | 87 | ✓ | 0.991 | 1.0 | 87 | 1.0 | ✓ | 146 |
| 49 | `9923` | adding operation | 3 | 3 | ✓ | 0.989 | 1.0 | 3 | 1.0 | ✓ | 391 |
| 50 | `9961` | digit expansion | 960 | 960 | ✓ | 0.996 | 1.0 | 960 | 1.0 | ✓ | 326 |
