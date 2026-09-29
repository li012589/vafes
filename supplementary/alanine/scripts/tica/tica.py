#!/usr/bin/env python
import argparse
import json
import os

import numpy as np
from scipy.linalg import eigh


def load_data(path):
    d = np.load(path)
    X, dt_ps = d["concise"], float(d["dt_ps"])
    if X.ndim != 3 or X.shape[2] != 33:
        raise SystemExit("expected concise of shape (n_runs, steps, 33), got %s" % (X.shape,))
    return X, dt_ps


def covariances(X, mean, lag):
    S0 = np.zeros((33, 33))
    St = np.zeros((33, 33))
    n = 0
    for x in X:
        xc = x.astype(np.float64) - mean
        a, b = xc[:-lag], xc[lag:]
        S0 += a.T @ a + b.T @ b
        St += a.T @ b
        n += len(a)
    C0 = S0 / (2 * n)
    K = St / n
    return C0, 0.5 * (K + K.T)


def tica(C0, K):
    ev, Q = eigh(C0)
    if ev.min() <= 0:
        raise SystemExit("C0 is not positive definite")
    Ci = (Q * ev ** -0.5) @ Q.T
    M = Ci @ K @ Ci
    M = 0.5 * (M + M.T)
    lam, U = eigh(M)
    o = np.argsort(lam)[::-1]
    lam, U = lam[o], U[:, o]
    V = Ci @ U
    return V, U, lam


def integerize(V, top=2):
    n = V.shape[0]
    cols, used = [], []
    for i in range(top):
        j = int(np.argmax(np.abs(V[:, i])))
        cols.append(np.eye(n)[:, j])
        used.append(j)
    for j in range(n):
        if j not in used:
            cols.append(np.eye(n)[:, j])
    return np.array(cols).T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="traj.npz", help="input npz (merged trajectories)")
    p.add_argument("--lag-ps", type=float, default=5.0, help="TICA lag time in ps")
    p.add_argument("--out", default="results", help="output directory")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    X, stride_ps = load_data(args.data)
    mean = X.astype(np.float64).sum(axis=(0, 1)) / (X.shape[0] * X.shape[1])
    lag = int(round(args.lag_ps / stride_ps))

    C0, K = covariances(X, mean, lag)
    V, U, lam = tica(C0, K)

    np.savez(os.path.join(args.out, "raw_tica.npz"),
             V=V, U=U, eigenvalues=lam, mean=mean, C0=C0, Ctau=K,
             lag_frames=lag, lag_ps=args.lag_ps, stride_ps=stride_ps,
             n_runs=X.shape[0], steps_per_run=X.shape[1])

    Pred = integerize(V)
    np.savez(os.path.join(args.out, "reduced_tica.npz"), permutation=Pred)

    rep = dict(n_runs=int(X.shape[0]), steps_per_run=int(X.shape[1]),
               frames=int(X.shape[0] * X.shape[1]),
               lag_ps=args.lag_ps, stride_ps=stride_ps,
               eigenvalues=lam.tolist(),
               col0=dict(sorted({int(j): float(V[j, 0] / np.linalg.norm(V[:, 0]))
                                 for j in np.argsort(-np.abs(V[:, 0]))[:8]}.items())),
               col1=dict(sorted({int(j): float(V[j, 1] / np.linalg.norm(V[:, 1]))
                                 for j in np.argsort(-np.abs(V[:, 1]))[:8]}.items())),
               argmax=[int(np.argmax(np.abs(V[:, i]))) for i in range(33)],
               raw_signs=[float(np.sign(V[int(np.argmax(np.abs(V[:, i]))), i]))
                          for i in range(33)])

    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)

    print("runs: %d x %d frames   stride: %g ps   lag: %g ps (%d frames)"
          % (X.shape[0], X.shape[1], stride_ps, args.lag_ps, lag))
    print("eigenvalues: %s" % np.round(lam, 4))
    print("IC1 top entries: " + "  ".join("[dim%d: %+.3f]" % (j, v)
          for j, v in sorted(rep["col0"].items(), key=lambda kv: -abs(kv[1]))))
    print("IC2 top entries: " + "  ".join("[dim%d: %+.3f]" % (j, v)
          for j, v in sorted(rep["col1"].items(), key=lambda kv: -abs(kv[1]))))
    print("integerized permutation written to %s/reduced_tica.npz" % args.out)


if __name__ == "__main__":
    main()
