# Web shooter: a stress test of the engine

A deliberately hard target, chosen because it lands on the parts of the engine
that are least well covered: a fluid that leaves a nozzle, becomes a solid
strand in flight, sticks hard to whatever it hits, and holds 30-50 lb.

The interesting output is not the ranking. It is the three places the engine
declined to answer, and the one place it caught a mistake in the specification.

Reproduce with:

```bash
formulate run examples/web_shooter/carrier.yaml --top 5
formulate run examples/web_shooter/strand.yaml  --top 5   # proposes nothing; see below
python examples/web_shooter/score_strand.py examples/web_shooter/strand.yaml 5
```

## Turning the ask into numbers

The engine does not do this part - section 5 puts it in the reasoning layer -
so it is written down here instead.

50 lb is 222 N. On a round strand that is:

| strand diameter | working stress | with a 2x snatch load |
|---|---|---|
| 1.0 mm | 283 MPa | 566 MPa |
| 1.5 mm | 126 MPa | 252 MPa |
| 2.0 mm | 71 MPa | 142 MPa |
| 3.0 mm | 32 MPa | 63 MPa |

A cast bar of any common glassy polymer breaks between 45 and 80 MPa. So the
1 mm strand of the films is not available from an undrawn polymer, and **the
small radius is the first thing that has to go.**

The device is a dry-spinning line: a polymer dissolved in a solvent, extruded,
the solvent flashing off in flight. That is how cellulose acetate and acrylic
fibres have been spun since the 1920s, and it is the only route that stores as
a pumpable fluid and arrives as a solid. It splits the problem in two, which is
what the two specifications here are.

## What the engine answered

**Carrier solvent** (`carrier.yaml`, molecule class, 50 reference compounds).
Three requirements, one per job the solvent does: dissolve the strand polymer
(`solubility_red`, from `conditions.solutes`), wet the target
(`work_of_separation`, from `conditions.surfaces`), and leave before the strand
lands (`normal_boiling_point`).

4 of 44 candidates feasible. Ranked: **acetone**, ethyl acetate, 2-butanone,
THF. Acetone is the solvent the acetate-fibre industry actually dry-spins from,
which is a reasonable sign the pipeline is not lost.

**Strand polymer** (`strand.yaml`, polymer class, 57 reference polymers).
2 of 57 feasible: **polystyrene** and **PMMA** - which are the only two
polymers in the bundled set that are both glassy at 25 degC and have tabulated
chain dimensions, so the result is as much a statement about coverage as about
polymers.

## The three refusals

**1. Adhesion cannot be scored for any solvent that works.** This is the
headline, because adhesion is the requirement the whole idea rests on.

`work_of_separation` comes from the Owens-Wendt expert, which needs a measured
dispersive/polar split for the liquid. Ten liquids have one. Seven polymers
have a tabulated Hansen sphere. The overlap - liquids that both have an
Owens-Wendt split and dissolve one of those polymers - is **empty**:

```
polystyrene                    NONE
poly(methyl methacrylate)      NONE
poly(vinyl chloride)           NONE
polycarbonate                  NONE
poly(vinyl acetate)            NONE
cellulose acetate              NONE
polyamide 66                   NONE
```

So every feasible carrier reports `work_of_separation` as not available, and
the adhesion requirement is inert. Sweeping the substrate over steel, glass and
polyethylene changes the ranking not at all - the engine cannot see the
substrate, because it cannot see the liquid.

Making the requirement hard turns this into a proper refusal, and the failure
diagnosis names the gap without being told about it:

```
Constraints never satisfied together by any candidate in this pool:
  - solubility_red [hard]: in range [-inf, 1] and
    work_of_separation [hard]: in range [0.06, +inf] newton / meter
```

It also distinguishes the two ways a candidate can fail - "eliminated 39
candidate(s), 37 of them because the value could not be predicted" - which is
the difference between a bad solvent and an unknown one.

**2. The polymer branch is unreachable from the CLI.** `formulate run
strand.yaml` proposes zero candidates. There are two polymer experts and 57
reference polymers with measured Tg and density, but no explorer that puts a
polymer in the pool: `ReferenceDatabaseExplorer` returns early unless the spec
asks for `MaterialClass.MOLECULE`, and `EvolutionaryExplorer` can only mutate a
polymer that is already there.

`score_strand.py` fills the gap with a twenty-line explorer and runs the real
pipeline behind it. Worth noting that this failure is quieter than the one
above: an empty pool prints "No candidates were evaluated" and does not say
that no explorer serves the requested material class.

**3. There is no strength, and no cure.** `theoretical_strength` is a flaw-free
bound, and in this engine it is `0.1 x E`, where `E` in the glass is a constant
2.86 GPa for every polymer. So it is two-valued - 286 MPa if glassy, ~0.1 MPa
if rubbery - and it is a glassiness switch, not a strength ranking. It cannot
choose between two glassy polymers, and nothing here predicts the load at which
a strand breaks. That is the right call rather than a gap: a specimen fails at
its largest flaw and the flaw belongs to the process.

Separately, the engine has no reaction or cure model at all, which rules out
the entire class of reactive one-part systems - cyanoacrylate above all - that
are what a real web would most likely be made of.

## The mistake it caught

The first draft of `strand.yaml` asked for `glass_transition_temperature` with
`direction: minimize` and `lower: 60 degC`. A minimize with a lower anchor
gives a perfect score to *anything below it*, so polyethylene came top of the
ranking with a Tg of -75 degC, scoring 1.000 on a requirement written to demand
glassiness. Glassiness is an `in_range`, and the printed per-requirement
utilities are what made this visible.

The same draft asked for `theoretical_strength >= 710 MPa`, a 10x margin on a
2 mm strand. Unsatisfiable by construction against a 286 MPa ceiling - and the
failure report said so precisely: `nearest miss passes at lower bound 2.86e+08
(currently 7.1e+08)`.

## What the answers are worth

Every axis on both winners scores 0.000 at the pessimistic bound and is flagged
`meets its bound only within uncertainty`. Tg carries +/- 34 K, so PMMA's
350 K is 316 K at one sigma - below the 333 K the spec demands. Entanglement
molar mass carries a factor of 1.74. Neither polymer survives its own error
bars, and the engine says so on the same line as the number.

## Hardware, which the engine says nothing about

No cost property exists in the registry; `synthetic_accessibility` stands in for
price in `carrier.yaml` and is a poor proxy - it scores how hard a molecule is
to make from scratch, not what a drum of commodity solvent costs. Everything
below is ordinary fluid mechanics done outside the engine.

**A single fat strand is hydraulically impossible.** Dry spinning shrinks the
filament as solvent leaves: at 30% solids a 3 mm dry strand needs a 5.8 mm
orifice, which passes 132 mL/s at 5 m/s and empties a wrist cartridge in a
quarter of a second. Shrink the orifice to 1 mm and a 10 Pa.s dope needs
160 bar to push through it.

**A bundle is the only way**, which is also what a spinneret and a spider's
dragline both are. Load is carried by total cross-section, but pressure drop
goes as `1/R^2` per hole, so many small holes beat one big one:

| filaments | dry filament | solids | viscosity | pressure | flow | holds at 100 MPa |
|---|---|---|---|---|---|---|
| 1 | 3000 um | 30% | 10 Pa.s | 2 bar | 132 mL/s | 159 lb |
| 40 | 300 um | 30% | 10 Pa.s | 191 bar | 53 mL/s | 64 lb |
| **40** | **300 um** | **20%** | **1 Pa.s** | **13 bar** | **80 mL/s** | **64 lb** |
| 200 | 150 um | 20% | 1 Pa.s | 50 bar | 100 mL/s | 79 lb |

The 13 bar design point is the cheap one, and it is cheap because of the
viscosity, not the hardware: 20% solids in acetone is a thin dope, and 13 bar
is a bicycle pump into a PET bottle.

- **Pressure**: a 2 L PET bottle pumped to 13 bar with a bike pump, or a 12 g
  CO2 cartridge and a piercing valve (~36 shots at 5 bar, ~18 at 10 bar).
  PET soda bottles are rated well above this and fail by splitting, not
  shattering.
- **Reservoir**: a 60 mL luer-lock syringe barrel. At 80 mL/s a 0.6 s shot uses
  ~48 mL, so this is **one shot per barrel** - the real cost of the bundle is
  that a web shooter carries about four shots, not forty.
- **Spinneret**: 40 holes of 0.7 mm. A drilled aluminium disc, or the cheapest
  version, 40 blunt 22-gauge dispensing needles in a manifold.
- **Valve**: a trigger-operated ball valve. The nozzle must be sealed or
  solvent-flooded between shots or the dope skins over in it, which is the
  failure mode that will actually end the project.

Total, well under $50, and none of it is the hard part.

## Verdict

The engine handled the parts it covers and refused the rest legibly rather than
guessing - the infeasibility diagnosis found the solvency/adhesion gap on its
own, and the uncertainty reporting made clear that neither surviving polymer
clears its bounds robustly. The unreachable polymer class is a plumbing gap
worth closing; the empty adhesion overlap is a coverage limit that only new
measured data fixes.
