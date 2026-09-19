"""Render the flight simulation as an animation.

One fired strand, nozzle to wall, 333 ms. The left panel is the strand itself
- position along the flight and a cross-section coloured by temperature. The
right column is three small multiples on a shared clock, rather than one chart
with three scales, because conversion, temperature and pinch-off share no axis.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np                        # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from simulate import ALPHA_GEL, FLIGHT, R_JET, simulate  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # blue, orange, aqua
# sequential blue ramp, light -> dark, for temperature magnitude
RAMP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
)

SPAN = 3.0        # m, nozzle to wall


def render(out="web_shooter_flight.gif", fps=20):
    k25 = math.log(1 / (1 - ALPHA_GEL)) / 0.200
    o = simulate(k25, report=False)
    ts, r = o["ts"], o["r"]
    T, a = o["T_field"], o["a_field"]
    Tmax, a_core, a_surf, pinch = o["Tmax"], o["a_core"], o["a_surf"], o["pinch"]
    T_lo, T_hi = 25.0, max(80.0, Tmax.max())

    fig = plt.figure(figsize=(12.6, 6.6), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.45, 1.0], hspace=0.62, wspace=0.26,
                          left=0.055, right=0.965, top=0.82, bottom=0.095)
    ax_fly = fig.add_subplot(gs[:, 0])
    axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    for ax in [ax_fly, *axes]:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d9d8d3")
        ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)

    fig.text(0.055, 0.945, "One strand, nozzle to wall", fontsize=17,
             color=INK, fontweight="semibold")
    fig.text(0.055, 0.895,
             "4.4 mm jet at 9 m/s, 333 ms of flight. Cure makes heat, heat "
             "accelerates cure, and cure raises the viscosity that resists pinch-off.",
             fontsize=10.5, color=INK_2)

    # -- left: the flight ---------------------------------------------------
    ax_fly.set_xlim(-0.22, SPAN + 0.30)
    ax_fly.set_ylim(-1.55, 1.15)
    ax_fly.set_yticks([])
    ax_fly.set_xlabel("distance from nozzle (m)", fontsize=10, color=INK_2)
    ax_fly.spines["left"].set_visible(False)
    ax_fly.plot([0, 0], [-0.30, 0.30], color=MUTED, lw=3, solid_capstyle="round")
    ax_fly.text(0, 0.42, "nozzle", ha="center", fontsize=9.5, color=INK_2)
    ax_fly.plot([SPAN, SPAN], [-0.55, 0.55], color=MUTED, lw=3, solid_capstyle="round")
    ax_fly.text(SPAN, 0.66, "wall", ha="center", fontsize=9.5, color=INK_2)

    from matplotlib.collections import LineCollection
    rope = LineCollection([], cmap=RAMP, linewidths=9, capstyle="round", zorder=3)
    rope.set_clim(T_lo, T_hi)
    ax_fly.add_collection(rope)
    head = ax_fly.scatter([], [], s=190, c=[], cmap=RAMP, vmin=T_lo, vmax=T_hi,
                          edgecolors=SURFACE, linewidths=2.0, zorder=4)
    clock = ax_fly.text(0.0, 0.92, "", fontsize=13, color=INK,
                        fontweight="semibold", family="monospace", va="baseline")
    state = ax_fly.text(0.0, -0.16, "", fontsize=11, color=INK_2, va="top")

    # cross-section inset: the strand seen end-on
    ax_cut = ax_fly.inset_axes([0.015, 0.02, 0.30, 0.40])
    ax_cut.set_facecolor(SURFACE)
    ax_cut.set_xticks([]); ax_cut.set_yticks([])
    for s in ax_cut.spines.values():
        s.set_color("#d9d8d3")
    ax_cut.set_title("cross-section (\u00b0C)", fontsize=9, color=INK_2, pad=4)
    rr = r / R_JET
    theta = np.linspace(0, 2 * np.pi, 80)
    RR, TH = np.meshgrid(rr, theta)
    mesh = ax_cut.pcolormesh(RR * np.cos(TH), RR * np.sin(TH),
                             np.zeros_like(RR), cmap=RAMP, vmin=T_lo, vmax=T_hi,
                             shading="gouraud")
    ax_cut.set_aspect("equal")
    ax_cut.set_xlim(-1.12, 1.12); ax_cut.set_ylim(-1.12, 1.12)
    cbar = fig.colorbar(mesh, ax=ax_cut, fraction=0.055, pad=0.06)
    cbar.ax.tick_params(labelsize=8, colors=INK_2, length=2)
    cbar.outline.set_visible(False)
    ax_fly.text(0.40, 0.30,
                "The disc stays one flat colour, and that is the result:\n"
                "heat needs 48 s to cross the strand against a 333 ms\n"
                "flight, so no radial gradient ever forms. The surface\n"
                "was predicted to cure last. It does not.",
                transform=ax_fly.transAxes, fontsize=9, color=INK_2,
                va="center", ha="left", linespacing=1.6)

    # -- right: three small multiples on a shared clock ---------------------
    specs = [
        ("Conversion", "fraction reacted", (0, 1.05),
         [("core", a_core, S1), ("surface", a_surf, S2)]),
        ("Peak temperature", "°C", (20, T_hi * 1.10),
         [("hottest point", Tmax, S2)]),
        ("Pinch-off progress", "% to break-up", (0, 9),
         [("perturbation", pinch * 100, S3)]),
    ]
    lines, dots, labels = [], [], []
    for ax, (title, ylab, ylim, series) in zip(axes, specs):
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=6)
        ax.set_ylabel(ylab, fontsize=9, color=INK_2)
        ax.set_xlim(0, FLIGHT * 1e3)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#ebeae5", lw=0.8)
        ax.set_axisbelow(True)
        row_l, row_d, row_t = [], [], []
        for name, y, col in series:
            ln, = ax.plot([], [], color=col, lw=2, solid_capstyle="round",
                          label=name, zorder=3)
            dt, = ax.plot([], [], "o", color=col, ms=8, mec=SURFACE, mew=1.6, zorder=4)
            tx = ax.text(0, 0, "", fontsize=9, color=INK, fontweight="semibold",
                         ha="left", va="center", zorder=5)
            row_l.append((ln, y)); row_d.append((dt, y)); row_t.append((tx, y, name))
        # stagger a coincident pair so the two labels never print on each other
        if len(row_t) == 2:
            row_t = [(row_t[0][0], row_t[0][1], row_t[0][2], +1),
                     (row_t[1][0], row_t[1][1], row_t[1][2], -1)]
        else:
            row_t = [(t, y, n, 0) for t, y, n in row_t]
        lines.append(row_l); dots.append(row_d); labels.append(row_t)
        if len(series) > 1:
            ax.legend(loc="lower right", fontsize=8.5, frameon=False,
                      labelcolor=INK_2, handlelength=1.4)
    axes[0].axhline(ALPHA_GEL, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    axes[0].text(4, ALPHA_GEL + 0.045, "gel point", fontsize=8.5, color=INK_2)
    axes[-1].set_xlabel("time since leaving the nozzle (ms)", fontsize=9.5, color=INK_2)

    def frame(i):
        t_ms = ts[i] * 1e3
        x = SPAN * ts[i] / FLIGHT
        n_trail = max(2, i + 1)
        xs = np.linspace(0, x, n_trail)
        idx = np.linspace(0, i, n_trail).astype(int)
        pts = np.c_[xs, np.zeros(n_trail)].reshape(-1, 1, 2)
        rope.set_segments(np.concatenate([pts[:-1], pts[1:]], axis=1))
        rope.set_array(Tmax[idx][:-1])
        head.set_offsets([[x, 0.0]])
        head.set_array(np.array([Tmax[i]]))
        clock.set_text(f"t = {t_ms:5.0f} ms     {x:.2f} m")

        if a_surf[i] >= ALPHA_GEL * 0.98:
            state.set_text("SOLID  —  gelled in flight, arrives able to carry load")
            state.set_color("#0ca30c")
        else:
            state.set_text(f"still liquid  —  {a_surf[i]*100:.0f}% reacted")
            state.set_color(INK_2)

        mesh.set_array(np.interp(RR.ravel(), rr, T[:, i]))

        late = t_ms > FLIGHT * 1e3 * 0.62
        for ax, row_l, row_d, row_t in zip(axes, lines, dots, labels):
            span = ax.get_ylim()[1] - ax.get_ylim()[0]
            for (ln, y), (dt, _), (tx, _, name, side) in zip(row_l, row_d, row_t):
                ln.set_data(ts[:i + 1] * 1e3, y[:i + 1])
                dt.set_data([t_ms], [y[i]])
                tx.set_position((t_ms - 9 if late else t_ms + 9,
                                 y[i] + side * 0.075 * span))
                tx.set_ha("right" if late else "left")
                fmt = "{:.2f}" if y.max() <= 1.5 else "{:.0f}"
                tx.set_text(f"{name} " + fmt.format(y[i]))
        return []

    n = len(ts)
    anim = FuncAnimation(fig, frame, frames=range(0, n, 2), interval=1000 / fps, blit=False)
    out_path = Path(__file__).parent / out
    anim.save(out_path, writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)
    print(f"wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB, "
          f"{len(range(0, n, 2))} frames)")
    return out_path


if __name__ == "__main__":
    render()
