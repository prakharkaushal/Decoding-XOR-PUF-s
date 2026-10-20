"""Tests for submit.py.

Run with pytest (python -m pytest) or without it (python tests/test_submit.py).
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from submit import my_decode, my_kernel  # noqa: E402

DATA = ROOT / "src"


# Forward model, copied from the course evaluation notebook
def get_APUF_model(p, q, r, s):
    p, q, r, s = (np.maximum(v, 0) for v in (p, q, r, s))
    d = p - q
    c = r - s
    alpha = (d + c) / 2
    beta = (d - c) / 2
    w = np.zeros(len(alpha) + 1)
    w[:-1] += alpha
    w[1:] += beta
    return w


def get_XOR_APUF_model(a, b, c, d, p, q, r, s):
    return np.kron(get_APUF_model(a, b, c, d), get_APUF_model(p, q, r, s))


def explicit_features(x, Z, c=0.1):
    """psi(x, z) = [x * phi(z), 1], where phi is the feature map of (z . z' + c)^2."""
    z1, z2 = Z[:, :1], Z[:, 1:]
    phi = np.hstack([z1**2, z2**2, np.sqrt(2) * z1 * z2,
                     np.sqrt(2 * c) * z1, np.sqrt(2 * c) * z2, np.full_like(z1, c)])
    return np.hstack([x * phi, np.ones_like(x)])


def assert_decodes(w, tol):
    delays = my_decode(w)
    assert len(delays) == 8
    for v in delays:
        assert v.shape == (32,)
        assert (v >= 0).all()
    assert np.linalg.norm(w - get_XOR_APUF_model(*delays)) < tol


def test_kernel_shape():
    rng = np.random.default_rng(0)
    G = my_kernel(rng.random((7, 1)), rng.random((7, 2)), rng.random((5, 1)), rng.random((5, 2)))
    assert G.shape == (7, 5)


def test_kernel_matches_explicit_feature_map():
    rng = np.random.default_rng(1)
    x1, Z1 = rng.random((50, 1)) * 20, rng.random((50, 2))
    x2, Z2 = rng.random((40, 1)) * 20, rng.random((40, 2))
    expected = explicit_features(x1, Z1) @ explicit_features(x2, Z2).T
    assert np.allclose(my_kernel(x1, Z1, x2, Z2), expected)


def test_kernel_gram_is_symmetric_psd():
    rng = np.random.default_rng(2)
    x, Z = rng.random((200, 1)) * 20, rng.random((200, 2))
    G = my_kernel(x, Z, x, Z)
    assert np.allclose(G, G.T)
    eigenvalues = np.linalg.eigvalsh(G)
    assert eigenvalues.min() > -1e-8 * eigenvalues.max()


def test_kernel_ridge_public_test_r2():
    def load(name, split):
        return np.loadtxt(DATA / f"public_{name}_{split}.txt")
    x_tr, Z_tr, y_tr = load("x", "trn").reshape(-1, 1), load("Z", "trn"), load("y", "trn")
    x_te, Z_te, y_te = load("x", "tst").reshape(-1, 1), load("Z", "tst"), load("y", "tst")
    model = KernelRidge(kernel="precomputed").fit(my_kernel(x_tr, Z_tr, x_tr, Z_tr), y_tr)
    assert model.score(my_kernel(x_te, Z_te, x_tr, Z_tr), y_te) > 0.96


def test_decode_public_models_round_trip():
    for w in np.loadtxt(DATA / "public_mod.txt"):
        assert_decodes(w, tol=1e-12)


def test_decode_random_delays_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(25):
        w = get_XOR_APUF_model(*(rng.random(32) for _ in range(8)))
        assert_decodes(w, tol=1e-10)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"{len(tests)} tests passed")
