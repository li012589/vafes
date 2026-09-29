import json
import secrets
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import openmm
from openmm import unit
import openmmplumed

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
CONFIG_DIR = SCRIPT_PATH.parents[2] / "configs"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_PATH.parent))

import dipeptideEnergy as de

DEFAULT_CONFIG = CONFIG_DIR / "metadynamics.json"
DEFAULT_PLUMED = CONFIG_DIR / "plumed_dipeptide_cv.dat"
KB = 0.0083144626


def build_system():
    system = openmm.System()
    for particle_mass in de.mass[0, :, 0].tolist():
        system.addParticle(float(particle_mass))

    bonds = openmm.HarmonicBondForce()
    for (i, j), values in zip(de.idxs[0].tolist(), de.params[0].tolist()):
        bonds.addBond(i, j, float(values[1]), float(values[0]))
    system.addForce(bonds)

    angles = openmm.HarmonicAngleForce()
    for (i, j, k), values in zip(de.idxs[5].tolist(), de.params[5].tolist()):
        angles.addAngle(i, j, k, float(values[1]), float(values[0]))
    system.addForce(angles)

    dihedrals = openmm.PeriodicTorsionForce()
    for index_set, parameter_set in ((de.idxs[6], de.params[6]), (de.idxs[7], de.params[7])):
        for (i, j, k, l), values in zip(index_set.tolist(), parameter_set.tolist()):
            if float(values[0]) != 0.0:
                dihedrals.addTorsion(i, j, k, l, int(round(float(values[1]))), float(values[2]), float(values[0]))
    system.addForce(dihedrals)

    coulomb = openmm.CustomBondForce("c/r")
    coulomb.addPerBondParameter("c")
    for index_set, parameter_set in ((de.idxs[1], de.params[1]), (de.idxs[3], de.params[3])):
        for (i, j), values in zip(index_set.tolist(), parameter_set.tolist()):
            charge_product = float(de.charge[0, i, 0]) * float(de.charge[0, j, 0])
            coulomb.addBond(i, j, [float(values[0]) * charge_product])
    system.addForce(coulomb)

    lennard_jones = openmm.CustomBondForce("4*eps*((sig/r)^12-(sig/r)^6)")
    lennard_jones.addPerBondParameter("sig")
    lennard_jones.addPerBondParameter("eps")
    for index_set, parameter_set in ((de.idxs[2], de.params[2]), (de.idxs[4], de.params[4])):
        for (i, j), values in zip(index_set.tolist(), parameter_set.tolist()):
            lennard_jones.addBond(i, j, [float(values[0]), float(values[1])])
    system.addForce(lennard_jones)
    return system


def starting_config(seed=0, min_pair=0.13):
    data = np.load(PROJECT_ROOT / "etc" / "dipeptideMeta.npz")
    ranges = data["ranges"]
    inverse_projection = np.linalg.inv(data["V"])
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        tica = rng.uniform(ranges[:, 0], ranges[:, 1])
        concise = tica @ inverse_projection
        full = de.concise2full(torch.tensor(concise, dtype=torch.float64).reshape(1, -1))[0].numpy()
        distances = np.linalg.norm(full[:, None, :] - full[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        if distances.min() > min_pair:
            return full
    raise RuntimeError("Could not sample a clash-free starting configuration")


def add_frame_pin(system, k=4.0e5):
    pin = openmm.CustomExternalForce("k*(wx*x*x + wy*y*y + wz*z*z)")
    pin.addGlobalParameter("k", k)
    pin.addPerParticleParameter("wx")
    pin.addPerParticleParameter("wy")
    pin.addPerParticleParameter("wz")
    pin.addParticle(5, [1.0, 1.0, 1.0])
    pin.addParticle(8, [1.0, 0.0, 1.0])
    pin.addParticle(3, [0.0, 0.0, 1.0])
    system.addForce(pin)
    return pin


def cv_of_positions(pos_nm):
    t = torch.tensor(pos_nm, dtype=torch.float64).reshape(1, 13, 3)
    c = de.full2concise(t).numpy()[0]
    return c[5], c[26]


def run(steps, dt_fs=1.0, T=300.0, friction=1.0, report_every=5000,
        plumed_file=None, seed=42, platform_name="Reference"):
    if plumed_file is None:
        plumed_file = DEFAULT_PLUMED
    platform = openmm.Platform.getPlatformByName(platform_name)

    cfg = starting_config(seed=seed)

    sys0 = build_system(); add_frame_pin(sys0)
    integ0 = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    ctx0 = openmm.Context(sys0, integ0, platform)
    ctx0.setPositions([openmm.Vec3(*r) for r in cfg])
    openmm.LocalEnergyMinimizer.minimize(ctx0, 1e-2, 1000)
    pos_min = ctx0.getState(getPositions=True).getPositions()
    e0 = ctx0.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    pos_arr = np.array(pos_min.value_in_unit(unit.nanometer))
    cv1, cv2 = cv_of_positions(pos_arr)
    print(f"[min] E={e0:.1f} kJ/mol | CV1={cv1:+.4f} CV2={cv2:+.4f} (target range +-0.16)", flush=True)
    del ctx0


    sys_md = build_system(); add_frame_pin(sys_md)
    with open(plumed_file) as f:
        script = f.read()
    pf = openmmplumed.PlumedForce(script); pf.setTemperature(T)
    sys_md.addForce(pf)

    integ = openmm.LangevinMiddleIntegrator(
        T * unit.kelvin, friction / unit.picosecond, dt_fs * 0.001 * unit.picosecond)
    integ.setRandomNumberSeed(seed)
    ctx = openmm.Context(sys_md, integ, platform)
    ctx.setPositions(pos_min)

    N = 13; dof = 3 * N - 3 - 6
    print(f"\n{'step':>10} {'ps':>9} {'PE':>11} {'KE':>9} {'T(K)':>7} {'CV1':>8} {'CV2':>8}", flush=True)
    print("-" * 66, flush=True)
    blow = False
    for s in range(0, steps + 1, report_every):
        st = ctx.getState(getEnergy=True, getPositions=True)
        pe = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        ke = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        pos = np.array(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        c1, c2 = cv_of_positions(pos)
        if not (np.isfinite(pe) and np.isfinite(ke)):
            print(f"{s:>10} *** NON-FINITE ***", flush=True); blow = True; break
        if s % (report_every * 4) == 0 or s == 0:
            print(f"{s:>10} {s*dt_fs/1000:>9.1f} {pe:>11.2f} {ke:>9.2f} "
                  f"{2*ke/(dof*KB):>7.1f} {c1:>8.4f} {c2:>8.4f}", flush=True)
        if s < steps:
            ctx.getIntegrator().step(report_every)
    if not blow:
        print("-" * 66, flush=True)
        print(f"completed {steps} steps ({steps*dt_fs/1000:.1f} ps).", flush=True)
    return not blow


if __name__ == "__main__":
    defaults = json.loads(DEFAULT_CONFIG.read_text())
    p = argparse.ArgumentParser()
    p.add_argument("-steps", type=int, default=defaults["steps_per_walker"])
    p.add_argument("-dt", type=float, default=defaults["dt_fs"])
    p.add_argument("-T", type=float, default=defaults["temperature_K"])
    p.add_argument("-friction", type=float, default=defaults["friction_per_ps"])
    p.add_argument("-report", type=int, default=defaults["report_every"])
    p.add_argument("-seed", type=int, default=None)
    p.add_argument("-platform", default=defaults["platform"])
    p.add_argument("-plumed", default=None)
    args = p.parse_args()
    if args.seed is None:
        args.seed = secrets.randbelow(2**31 - 1) + 1
    print(f"Using seed: {args.seed}", flush=True)
    ok = run(args.steps, args.dt, args.T, args.friction, args.report,
             args.plumed, args.seed, args.platform)
    sys.exit(0 if ok else 1)
