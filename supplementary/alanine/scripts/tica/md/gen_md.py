#!/usr/bin/env python3

import argparse
import os
import secrets
import time

import numpy as np
import openmm
from openmm import unit


HERE = os.path.dirname(os.path.abspath(__file__))


def concise2full(c):
    n = len(c)
    full = np.zeros((n, 13, 3))
    full[:, :3] = c[:, :9].reshape(-1, 3, 3)
    full[:, 3, :2] = c[:, 9:11]
    full[:, 4] = c[:, 11:14]
    full[:, 6:8] = c[:, 14:20].reshape(-1, 2, 3)
    full[:, 8, 1] = c[:, 20]
    full[:, 9:] = c[:, 21:].reshape(-1, 4, 3)
    return full


def full2concise(p_full):
    p = p_full - p_full[:, 5:6, :]
    y = p[:, 8, :]
    u = y / np.linalg.norm(y, axis=1, keepdims=True)
    v = p[:, 3, :]
    v = v - (v * u).sum(1, keepdims=True) * u
    x = v / np.linalg.norm(v, axis=1, keepdims=True)
    z = np.cross(u, x)
    R = np.stack([x, u, z], axis=-1)
    q = p @ R
    q[:, :, 2] *= np.where(q[:, 7, 2] > 0, 1.0, -1.0)[:, None]
    c = np.zeros((len(p_full), 33))
    c[:, :9] = q[:, :3].reshape(-1, 9)
    c[:, 9:11] = q[:, 3, :2]
    c[:, 11:14] = q[:, 4]
    c[:, 14:20] = q[:, 6:8].reshape(-1, 6)
    c[:, 20] = q[:, 8, 1]
    c[:, 21:] = q[:, 9:].reshape(-1, 12)
    return c


def starting_config(seed, min_pair=0.13):
    d = np.load(os.path.join(HERE, "initial_coordinate_ranges.npz"))
    ranges = d["ranges"]
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        concise = rng.uniform(ranges[:, 0], ranges[:, 1])
        full = concise2full(concise[None])[0]
        dist = np.linalg.norm(full[:, None, :] - full[None, :, :], axis=-1)
        np.fill_diagonal(dist, 1e9)
        if dist.min() > min_pair:
            return full
    raise RuntimeError("could not sample a clash-free starting config")


def run_seed(system_xml, seed, ns, dt_fs, T, friction, stride_fs, equil_ps, platform):
    system = openmm.XmlSerializer.deserialize(system_xml)
    system.addForce(openmm.CMMotionRemover(1))
    integ = openmm.LangevinMiddleIntegrator(
        T * unit.kelvin, friction / unit.picosecond, dt_fs * 0.001 * unit.picosecond)
    integ.setRandomNumberSeed(seed)
    ctx = openmm.Context(system, integ, platform)
    ctx.setPositions([openmm.Vec3(*r) for r in starting_config(seed)])
    openmm.LocalEnergyMinimizer.minimize(ctx, 1e-2, 1000)
    integ.step(int(equil_ps * 1000 / dt_fs))

    stride = int(round(stride_fs / dt_fs))
    nframes = int(round(ns * 1e6 / dt_fs / stride))
    X = np.empty((nframes, 33), np.float32)
    buf = np.empty((min(nframes, 100000), 13, 3), np.float64)
    done, t0 = 0, time.time()
    while done < nframes:
        n = min(len(buf), nframes - done)
        for i in range(n):
            buf[i] = np.asarray(ctx.getState(getPositions=True).getPositions(asNumpy=True)
                                .value_in_unit(unit.nanometer))
            ctx.getIntegrator().step(stride)
        X[done:done + n] = full2concise(buf[:n]).astype(np.float32)
        done += n
        print("  seed %d: %d/%d frames  %.0fs" % (seed, done, nframes, time.time() - t0),
              flush=True)
    del ctx
    return X


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-seeds", type=int, nargs="+", default=None)
    p.add_argument("-ns", type=float, default=1000.0)
    p.add_argument("-dt", type=float, default=1.0)
    p.add_argument("-T", type=float, default=300.0)
    p.add_argument("-friction", type=float, default=1.0)
    p.add_argument("-stride_fs", type=float, default=250.0)
    p.add_argument("-equil_ps", type=float, default=500.0)
    p.add_argument("-out", default="traj.npz")
    p.add_argument("-platform", default="Reference")
    args = p.parse_args()
    if args.seeds is None:
        args.seeds = [secrets.randbelow(2**31 - 1) + 1]

    with open(os.path.join(HERE, "system.xml")) as f:
        system_xml = f.read()
    platform = openmm.Platform.getPlatformByName(args.platform)

    runs = []
    for seed in args.seeds:
        print("run seed %d" % seed, flush=True)
        runs.append(run_seed(system_xml, seed, args.ns, args.dt, args.T,
                             args.friction, args.stride_fs, args.equil_ps, platform))
    X = np.stack(runs)
    np.savez_compressed(args.out, concise=X, dt_ps=args.stride_fs / 1000.0, T=args.T)
    print("wrote %s  concise %s" % (args.out, X.shape), flush=True)


if __name__ == "__main__":
    main()
