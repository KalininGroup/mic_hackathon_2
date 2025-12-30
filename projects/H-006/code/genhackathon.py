from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ase.build
import abtem

outdir = Path("dataset_amorphous_c")
outdir.mkdir(exist_ok=True)

for seed in range(69420, 69420 + 500):
    # --- your exact setup ---
    substrate = ase.build.bulk("C", cubic=True)
    substrate = substrate * (3, 3, 3)

    bl = 1.54
    rng = np.random.default_rng(seed=seed)
    substrate.positions[:] += rng.normal(size=(len(substrate), 3)) * 0.5 * bl

    potential = abtem.Potential(substrate, sampling=0.05)

    probe = abtem.Probe(energy=60e3, semiangle_cutoff=60)
    probe.grid.match(potential)

    grid_scan = abtem.GridScan(
        start=(0, 0),
        end=(0.6, 0.6),
        sampling=0.125,
        fractional=True,
        potential=potential,
    )

    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    abtem.show_atoms(substrate, ax=ax)
    grid_scan.add_to_plot(ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / f"structure_seed_{seed}.png", dpi=200)
    plt.close(fig)

    haadf = abtem.AnnularDetector(inner=90, outer=200)
    measurement = probe.scan(potential, scan=grid_scan, detectors=haadf)

    measurement = measurement.compute(scheduler="single-threaded")

    # save raw array
    np.save(outdir / f"haadf_seed_{seed}.npy", measurement.array)

    # save the HAADF plot in the same “abtem show” style, but with viridis
    fig, ax = plt.subplots(figsize=(5, 5))
    measurement.show(ax=ax, cmap="viridis")
    fig.tight_layout()
    fig.savefig(outdir / f"haadf_seed_{seed}.png", dpi=200)
    plt.close(fig)
