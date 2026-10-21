"""Descriptive and inferential statistics.

The descriptive measures are written by hand from their textbook formulas
(no .describe(), no np.percentile) so that every number can be explained.
`cross_check()` then compares them against pandas / scipy to prove they are right.

The hypothesis-test helpers all return the same kind of dictionary, so the
notebooks and the Streamlit app can display any test in the same way.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats as sps

from .config import RANDOM_STATE, TARGET

ALPHA = 0.05


# =====================================================================
# 1. Descriptive statistics - manual formulas
# =====================================================================
def _clean(x):
    """Turn input into a plain float array with missing values removed."""
    arr = np.asarray(x, dtype=float)
    return arr[~np.isnan(arr)]


def mean(x):
    """Arithmetic mean = sum of values / number of values."""
    x = _clean(x)
    return float(sum(x) / len(x))


def weighted_mean(x, w):
    """Weighted mean = sum(w_i * x_i) / sum(w_i)."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    keep = ~np.isnan(x)
    x, w = x[keep], w[keep]
    return float(sum(w * x) / sum(w))


def quantile(x, p):
    """p-th quantile using linear interpolation between the two closest ranks.

    Position h = (n - 1) * p in the sorted data (0-based). If h is not a whole
    number we move proportionally between the values at floor(h) and floor(h)+1.
    This is the same rule as pandas/numpy's default ("linear").
    """
    s = sorted(_clean(x))
    n = len(s)
    h = (n - 1) * p
    lo = math.floor(h)
    hi = min(lo + 1, n - 1)
    return float(s[lo] + (h - lo) * (s[hi] - s[lo]))


def median(x):
    """Median = middle value of the sorted data (average of the two middle values if n is even)."""
    s = sorted(_clean(x))
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 else float((s[mid - 1] + s[mid]) / 2)


def mode(x):
    """Mode = most frequent value. If several values tie, the smallest is returned."""
    counts = Counter(_clean(x))
    top = max(counts.values())
    return float(min(v for v, c in counts.items() if c == top))


def variance(x, ddof=1):
    """Variance = sum((x - mean)^2) / (n - ddof). ddof=1 gives the sample variance."""
    x = _clean(x)
    m = mean(x)
    return float(sum((x - m) ** 2) / (len(x) - ddof))


def std(x, ddof=1):
    """Standard deviation = square root of the variance."""
    return math.sqrt(variance(x, ddof))


def data_range(x):
    """Range = maximum - minimum."""
    x = _clean(x)
    return float(max(x) - min(x))


def iqr(x):
    """Inter-quartile range = Q3 - Q1 (spread of the middle 50% of the data)."""
    return quantile(x, 0.75) - quantile(x, 0.25)


def quartile_deviation(x):
    """Quartile deviation (semi-IQR) = (Q3 - Q1) / 2."""
    return iqr(x) / 2


def pearson_skewness(x):
    """Karl Pearson's (second) coefficient of skewness = 3 * (mean - median) / std."""
    s = std(x)
    return 3 * (mean(x) - median(x)) / s if s > 0 else float("nan")


def bowley_skewness(x):
    """Bowley's (quartile) coefficient of skewness = (Q3 + Q1 - 2*median) / (Q3 - Q1)."""
    q1, q2, q3 = quantile(x, 0.25), median(x), quantile(x, 0.75)
    if q3 == q1:  # no spread in the middle 50% -> coefficient undefined
        return float("nan")
    return (q3 + q1 - 2 * q2) / (q3 - q1)


def moment_skewness(x):
    """Moment coefficient of skewness = m3 / m2^1.5 (used only to cross-check the direction)."""
    x = _clean(x)
    m = mean(x)
    m2 = sum((x - m) ** 2) / len(x)
    m3 = sum((x - m) ** 3) / len(x)
    return float(m3 / m2 ** 1.5)


def excess_kurtosis(x):
    """Excess kurtosis = m4 / m2^2 - 3, where m_k = average of (x - mean)^k.

    A normal distribution has kurtosis 3, so subtracting 3 makes "normal" = 0.
    """
    x = _clean(x)
    m = mean(x)
    m2 = sum((x - m) ** 2) / len(x)
    m4 = sum((x - m) ** 4) / len(x)
    return float(m4 / m2 ** 2 - 3) if m2 > 0 else float("nan")


def kurtosis_type(k, tol=0.5):
    """Name the shape of the tails from the excess kurtosis."""
    if k != k:  # NaN
        return "Undefined (no spread)"
    if k > tol:
        return "Leptokurtic (peaked, heavy tails)"
    if k < -tol:
        return "Platykurtic (flat, light tails)"
    return "Mesokurtic (close to normal)"


def skew_label(s):
    """Describe a skewness coefficient in words."""
    if s != s:  # NaN
        return "undefined (no spread)"
    a = abs(s)
    if a < 0.1:
        return "approximately symmetric"
    direction = "positively (right)" if s > 0 else "negatively (left)"
    strength = "slightly" if a < 0.5 else "strongly"
    return f"{strength} {direction} skewed"


def class_balanced_weights(y):
    """Weight each patient by 1 / (size of their class).

    The dataset has more heart-disease patients (55%) than healthy ones (45%).
    With these weights both classes contribute equally, so the weighted mean is
    the average of the two group means - a "class-balanced" mean.
    """
    y = pd.Series(y)
    counts = y.value_counts()
    return y.map(lambda c: 1.0 / counts[c]).to_numpy()


def descriptive_table(df, columns, weights=None):
    """One row per feature with every descriptive measure from the syllabus."""
    rows = []
    for col in columns:
        x = df[col]
        q1, q3 = quantile(x, 0.25), quantile(x, 0.75)
        k = excess_kurtosis(x)
        row = {
            "Feature": col,
            "N": int(x.notna().sum()),
            "Mean": mean(x),
            "Median": median(x),
            "Mode": mode(x),
            "Weighted mean": weighted_mean(x, weights) if weights is not None else np.nan,
            "Std": std(x),
            "Min": float(np.nanmin(x)),
            "Max": float(np.nanmax(x)),
            "Range": data_range(x),
            "Q1": q1,
            "Q3": q3,
            "IQR": q3 - q1,
            "Quartile dev.": (q3 - q1) / 2,
            "Pearson skew": pearson_skewness(x),
            "Bowley skew": bowley_skewness(x),
            "Excess kurtosis": k,
            "Kurtosis type": kurtosis_type(k).split(" (")[0],
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("Feature")


def cross_check(df, columns):
    """Compare the manual formulas with pandas / scipy. 'Max abs diff' should be ~0."""
    checks = []
    for col in columns:
        x = df[col].dropna()
        pairs = {
            "Mean": (mean(x), x.mean()),
            "Median": (median(x), x.median()),
            "Mode": (mode(x), x.mode().min()),
            "Std (ddof=1)": (std(x), x.std()),
            "Q1": (quantile(x, 0.25), x.quantile(0.25)),
            "Q3": (quantile(x, 0.75), x.quantile(0.75)),
            "IQR": (iqr(x), sps.iqr(x)),
            "Moment skewness": (moment_skewness(x), sps.skew(x)),
            "Excess kurtosis": (excess_kurtosis(x), sps.kurtosis(x, fisher=True, bias=True)),
        }
        for measure, (manual, library) in pairs.items():
            checks.append({"Feature": col, "Measure": measure, "Manual": manual,
                           "Library": float(library), "Abs diff": abs(manual - float(library))})
    return pd.DataFrame(checks)


def pearson_r(x, y):
    """Karl Pearson correlation r = cov(x, y) / (std_x * std_y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    return covariance(x, y) / (std(x) * std(y))


def covariance(x, y):
    """Sample covariance = sum((x - mean_x)(y - mean_y)) / (n - 1)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    return float(sum((x - x.mean()) * (y - y.mean())) / (len(x) - 1))


def correlation_strength(r):
    """Words for the size and direction of a correlation coefficient."""
    a = abs(r)
    if a < 0.1:
        return "no / negligible correlation"
    strength = "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong"
    return f"{strength} {'positive' if r > 0 else 'negative'} correlation"


# =====================================================================
# 2. Inferential statistics
# =====================================================================
def fmt_p(p):
    """Readable p-value."""
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def _result(test, h0, h1, statistic, p, conclusion, alpha=ALPHA, **extra):
    out = {
        "test": test, "h0": h0, "h1": h1,
        "statistic": float(statistic), "p_value": float(p), "alpha": alpha,
        "significant": bool(p < alpha), "conclusion": conclusion,
    }
    out.update(extra)
    return out


def _decision(p, alpha=ALPHA):
    if p < 0.001:
        return f"p < 0.001 (below α = {alpha}), so we reject H0."
    if p < alpha:
        return f"p = {p:.4f} < {alpha}, so we reject H0."
    return f"p = {p:.4f} ≥ {alpha}, so we fail to reject H0."


def shapiro_test(x, name="feature", alpha=ALPHA):
    """Shapiro-Wilk test of normality."""
    x = _clean(x)
    w, p = sps.shapiro(x)
    if p < alpha:
        text = f"{_decision(p, alpha)} {name} is NOT normally distributed."
    else:
        text = f"{_decision(p, alpha)} {name} looks consistent with a normal distribution."
    return _result("Shapiro-Wilk", f"{name} is normally distributed",
                   f"{name} is not normally distributed", w, p, text, alpha, n=len(x))


def one_sample_ztest(x, mu0, name="feature", alpha=ALPHA):
    """One-sample Z-test: is the population mean equal to a reference value mu0?

    z = (sample mean - mu0) / (s / sqrt(n)).  With a large sample (n > 30) the sample
    standard deviation s is a reliable stand-in for the population sigma, and by the
    Central Limit Theorem the sample mean is approximately normal, so z follows N(0, 1).
    """
    x = _clean(x)
    n = len(x)
    m = mean(x)
    se = std(x) / math.sqrt(n)
    z = (m - mu0) / se
    p = 2 * (1 - sps.norm.cdf(abs(z)))
    if p < alpha:
        direction = "higher" if m > mu0 else "lower"
        text = (f"{_decision(p, alpha)} The average {name} ({m:.2f}) is significantly {direction} "
                f"than the reference value {mu0}.")
    else:
        text = (f"{_decision(p, alpha)} The average {name} ({m:.2f}) is not significantly different "
                f"from the reference value {mu0}.")
    return _result("One-sample Z-test", f"population mean {name} = {mu0} (μ = {mu0})",
                   f"population mean {name} ≠ {mu0} (μ ≠ {mu0})", z, p, text, alpha,
                   mean=m, mu0=mu0, n=n, se=se)


def welch_ttest(df, feature, group=TARGET, alpha=ALPHA, labels=("no heart disease", "heart disease")):
    """Two-sample Welch t-test (does not assume equal variances)."""
    a = _clean(df.loc[df[group] == 0, feature])
    b = _clean(df.loc[df[group] == 1, feature])
    t, p = sps.ttest_ind(a, b, equal_var=False)
    ma, mb = a.mean(), b.mean()
    if p < alpha:
        text = (f"{_decision(p, alpha)} The average {feature} is significantly different between "
                f"patients with {labels[0]} ({ma:.1f}) and {labels[1]} ({mb:.1f}).")
    else:
        text = (f"{_decision(p, alpha)} There is no significant difference in average {feature} between "
                f"patients with {labels[0]} ({ma:.1f}) and {labels[1]} ({mb:.1f}).")
    return _result("Welch two-sample t-test",
                   f"mean {feature} is the same in both groups (μ₀ = μ₁)",
                   f"mean {feature} differs between the groups (μ₀ ≠ μ₁)",
                   t, p, text, alpha, mean_0=ma, mean_1=mb, n_0=len(a), n_1=len(b))


def anova_test(df, feature, group=TARGET, alpha=ALPHA):
    """One-way ANOVA: are the means of `feature` equal across all levels of `group`?"""
    sub = df[[feature, group]].dropna()
    levels = sorted(sub[group].unique(), key=str)
    samples = [sub.loc[sub[group] == g, feature].to_numpy() for g in levels]
    f, p = sps.f_oneway(*samples)
    means = {str(g): float(s.mean()) for g, s in zip(levels, samples)}
    mean_txt = ", ".join(f"{g}: {m:.1f}" for g, m in means.items())
    if p < alpha:
        text = f"{_decision(p, alpha)} At least one {group} group has a different mean {feature} ({mean_txt})."
    else:
        text = f"{_decision(p, alpha)} The mean {feature} is not significantly different across {group} groups ({mean_txt})."
    return _result("One-way ANOVA", f"mean {feature} is equal in every {group} group",
                   f"at least one {group} group has a different mean {feature}",
                   f, p, text, alpha, group_means=means, k=len(levels))


def chi_square_test(df, feature, target=TARGET, alpha=ALPHA):
    """Chi-square test of independence between a categorical feature and the target."""
    table = pd.crosstab(df[feature], df[target])
    chi2, p, dof, expected = sps.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = math.sqrt(chi2 / (n * (min(table.shape) - 1)))
    if p < alpha:
        text = f"{_decision(p, alpha)} {feature} and heart disease are associated (not independent). Cramér's V = {cramers_v:.2f}."
    else:
        text = f"{_decision(p, alpha)} There is no evidence of an association between {feature} and heart disease."
    return _result("Chi-square test of independence",
                   f"{feature} and heart disease are independent",
                   f"{feature} and heart disease are associated",
                   chi2, p, text, alpha, dof=int(dof), cramers_v=cramers_v,
                   table=table, expected=pd.DataFrame(expected, index=table.index, columns=table.columns))


def mean_ci(x, confidence=0.95):
    """t-based confidence interval for the population mean: mean ± t * s / sqrt(n)."""
    x = _clean(x)
    n = len(x)
    m = mean(x)
    se = std(x) / math.sqrt(n)
    t_crit = sps.t.ppf((1 + confidence) / 2, df=n - 1)
    return {"mean": m, "lower": m - t_crit * se, "upper": m + t_crit * se,
            "n": n, "se": se, "t_crit": float(t_crit), "confidence": confidence}


def bootstrap_means(x, sample_size=30, n_samples=5000, random_state=RANDOM_STATE):
    """Draw many random samples (with replacement) and return their means - used for the CLT demo."""
    x = _clean(x)
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(x), size=(n_samples, sample_size))
    return x[idx].mean(axis=1)


def results_table(results):
    """Turn a list of test-result dicts into a tidy DataFrame for display."""
    rows = []
    for name, r in results:
        rows.append({"Variable": name, "Test": r["test"], "Statistic": round(r["statistic"], 3),
                     "p-value": fmt_p(r["p_value"]),
                     "Decision (α = 0.05)": "Reject H0" if r["significant"] else "Fail to reject H0"})
    return pd.DataFrame(rows)
