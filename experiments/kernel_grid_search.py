"""Hyperparameter search for the semi-parametric kernel (Problem 1.1).

Reproduces the report's grid search and checks it with 5-fold cross-validation:

    K((x1, z1), (x2, z2)) = x1 * x2 * (z1 . z2 + c)^d + 1

1. Single 80/20 split of the training data (random_state=42), as in Report.pdf.
2. 5-fold cross-validation on the same 4000 training points.
3. Test R^2 of the submitted setting (d=2, c=0.1) on the held-out public test set.

Run from the repository root:  python experiments/kernel_grid_search.py
"""
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, train_test_split

DATA = Path(__file__).resolve().parent.parent / "src"
DEGREES = [1, 2, 3, 4, 5]
COEF0S = [0, 0.1, 1, 2, 5]


def load(split):
    x = np.loadtxt(DATA / f"public_x_{split}.txt").reshape(-1, 1)
    Z = np.loadtxt(DATA / f"public_Z_{split}.txt")
    y = np.loadtxt(DATA / f"public_y_{split}.txt")
    return x, Z, y


def gram(xx, zz, d, c):
    """Kernel matrix from precomputed products x1*x2 and z1.z2."""
    return xx * (zz + c) ** d + 1.0


def score(xx, zz, y, train, valid, d, c):
    K = gram(xx, zz, d, c)
    model = KernelRidge(kernel="precomputed", alpha=1.0)
    model.fit(K[np.ix_(train, train)], y[train])
    return model.score(K[np.ix_(valid, train)], y[valid])


def print_grid(title, cells):
    print(f"\n{title}")
    print("d \\ c  " + "".join(f"{c:>17}" for c in COEF0S))
    for d in DEGREES:
        print(f"{d:<7}" + "".join(f"{cells[d, c]:>17}" for c in COEF0S))


def main():
    x, Z, y = load("trn")
    xx, zz = x @ x.T, Z @ Z.T
    idx = np.arange(len(y))

    # 1. The report's single split
    train, valid = train_test_split(idx, test_size=0.2, random_state=42)
    single = {(d, c): score(xx, zz, y, train, valid, d, c) for d in DEGREES for c in COEF0S}
    print_grid("Single 80/20 split (random_state=42), validation R^2",
               {k: f"{v:.5f}" for k, v in single.items()})

    # 2. 5-fold cross-validation
    folds = list(KFold(n_splits=5, shuffle=True, random_state=0).split(idx))
    cv = {(d, c): np.array([score(xx, zz, y, tr, va, d, c) for tr, va in folds])
          for d in DEGREES for c in COEF0S}
    print_grid("5-fold CV, mean R^2 +/- std across folds",
               {k: f"{v.mean():.4f}+/-{v.std():.4f}" for k, v in cv.items()})

    best = max(cv, key=lambda k: cv[k].mean())
    chosen = (2, 0.1)
    gap = cv[best].mean() - cv[chosen].mean()
    print(f"\nBest CV setting: d={best[0]}, c={best[1]} ({cv[best].mean():.5f})")
    print(f"Submitted d=2, c=0.1: {cv[chosen].mean():.5f}, "
          f"{gap:.5f} below the best ({gap / cv[chosen].std():.2f} fold std)")

    # 3. Final check on the held-out public test set
    xt, Zt, yt = load("tst")
    model = KernelRidge(kernel="precomputed", alpha=1.0)
    model.fit(gram(xx, zz, *chosen), y)
    test_r2 = model.score(gram(xt @ x.T, Zt @ Z.T, *chosen), yt)
    print(f"Submitted setting, public test R^2: {test_r2:.5f}")


if __name__ == "__main__":
    main()
