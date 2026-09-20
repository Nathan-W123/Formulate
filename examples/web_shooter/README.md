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
polymers. Polystyrene leads; PMMA grips steel harder but is denser and closer
to its glass transition.

## The three refusals

**1. Adhesion cannot be scored for any solvent that works** - and asking the
solvent was the wrong question. Resolved; the history is kept because the
diagnosis is the useful part.

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

**The fix, and why it needed no new measurements.** The solvent is not what
holds. It wets the target for the instant before it evaporates; what is left
gripping the wall is the solid strand. So the adhesion question belongs to the
polymer, and eight polymer surface energies were already in the repository -
sitting in the substrate table, usable only as the thing being stuck *to*.

The expert now accepts a polymer candidate and reads that same table from the
other side: a clean polystyrene surface has one surface energy whether the
polystyrene is the wall or the web. Same Owens-Wendt physics, same measured
data, no new correlation. It still refuses an untabulated polymer, and it
refuses copolymers outright, because the lower-energy unit enriches at the
surface and a mole-weighted average would be biased in a known direction.

With `surfaces: [steel]` in `strand.yaml`, the strand ranking now scores it:

| polymer | steel | glass | polyethylene |
|---|---|---|---|
| polystyrene | 76.5 | 84.3 | 73.9 |
| PMMA | 81.7 | 96.7 | 68.8 |

mN/m. PMMA grips steel and glass better; polystyrene wins on plastics. The
substrate finally changes the answer.

This still predicts no pounds. It is reversible work, and a real joint
dissipates one to three orders of magnitude more, so it ranks polymers against
each other and says nothing about whether the web holds.

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

## Asking for a web that deforms: no answer, and a real reason

`strand_tough.yaml` is the same spec with the toughness axis promoted from a
soft objective to a hard cap - entanglement molar mass at or below 5 kg/mol, so
a chain tangles often enough to carry load through a network instead of
snapping. Nothing passes, and the failure names the pair:

```
Constraints never satisfied together by any candidate in this pool:
  - glass_transition_temperature [hard]: in range [333.15, 453.15] kelvin and
    entanglement_molar_mass [hard]: in range [-inf, 5] kilogram / mole
```

Glassy and tough are mutually exclusive across the whole bundled set. That is
not an artifact of the fitted `M_e` model: against the *measured* entanglement
values for the six polymers that have both, Tg correlates with `ln(M_e)` at
r = +0.75.

| polymer | Tg (K) | measured M_e (g/mol) | glassy at 298 K |
|---|---|---|---|
| cis-1,4-polybutadiene | 171 | 1850 | no |
| polyethylene | 195 | 1150 | no |
| cis-1,4-polyisoprene | 200 | 6200 | no |
| polyisobutylene | 200 | 6700 | no |
| polystyrene | 373 | 18100 | yes |
| PMMA | 378 | 9200 | yes |

There is a mechanism behind it: `M_e` scales as the cube of the packing length,
and a bulky side group both fattens the chain - raising the packing length -
and hinders its rotation, raising Tg. One cause, both effects. Six points is
not a law, but the direction is physical rather than fitted.

So no single homopolymer that is glassy at room temperature is also tough, and
the engine is right to return nothing. What actually resolves it in the real
world is two phases, not one: crystallites acting as crosslinks in a drawn
semicrystalline fibre (nylon, PET, UHMWPE - and spider silk, which is
beta-sheet crystallites in an amorphous matrix), or rubber particles dispersed
in a glassy matrix (HIPS, ABS).

Both are outside this engine entirely. There is no crystallinity model - the
modulus is a switch on Tg alone - and while `MaterialClass.COMPOSITE` exists,
no expert supports it. The registry also has no elongation at break, no yield
stress and no fracture toughness, so `entanglement_molar_mass` is the only
toughness axis available and it is an indirect one.

## Dropping glassiness, which was never a requirement

The deadlock above dissolves once you ask why a glass was demanded at all. It
was not demanded by the problem. It was inherited from this engine's strength
model: `theoretical_strength` is a tenth of Young's modulus, so anything soft
scores as weak by construction, and an elastomer comes out at 0.1 MPa.

That is an artifact. Strength and stiffness are different quantities. Natural
rubber breaks near 25 MPa - its strength comes from strain-induced
crystallisation, not from its modulus - which holds 50 lb on a 3.4 mm strand
with room to spare, while stretching several hundred percent to absorb a catch.
A load-bearing line does not have to be rigid; a bungee is not, and neither is
silk.

`strand_extensible.yaml` drops the glass transition and the strength bound and
keeps the toughness cap. 4 of 57 feasible: **polyethylene**, **cis-1,4-
polyisoprene** (natural rubber), **cis-1,4-polybutadiene**, poly(ethylene
oxide). The two rubbers are what the problem was asking for all along, and
natural rubber dissolved in a light alkane is rubber cement - a fluid that
squirts, flashes off its solvent, and leaves a stretchy strand that grips.

The honest cost of this run is that it has **no strength axis at all**. It
selects on toughness and adhesion and takes on trust that an elastomer carries
the load, because the only strength number here would score it a thousand times
too low. The engine cannot confirm the thing the design now rests on.

Two further gaps this exposes: `solubility_red` has no Hansen sphere for
polyisoprene or polybutadiene, so the carrier spec cannot check that a solvent
dissolves them either, and neither rubber has a tabulated surface energy, so
the adhesion axis goes unavailable again for the top rubber candidates.

## What the answers are worth

Every axis on both winners scores 0.000 at the pessimistic bound and is flagged
`meets its bound only within uncertainty`. Tg carries +/- 34 K, so PMMA's
350 K is 316 K at one sigma - below the 333 K the spec demands. Entanglement
molar mass carries a factor of 1.74. Neither polymer survives its own error
bars, and the engine says so on the same line as the number.

## The four questions the engine cannot answer at all

Whether the thing works is decided by transport and kinetics, and there is no
expert for either. Worked outside the engine, in
`scratchpad/web_physics.py` terms:

**Strength.** Cross-section for 50 lb at a safety factor of 2, and what 3 m of
it weighs:

| material | UTS | area | single strand | 3 m dry | dope at 25% solids |
|---|---|---|---|---|---|
| unvulcanised natural rubber (rubber cement) | ~2 MPa | 297 mm2 | 19.4 mm | 836 g | 3.3 kg |
| SBS block copolymer (contact cement) | 20 MPa | 22 mm2 | 5.3 mm | 63 g | 251 g |
| cast PMMA | 70 MPa | 6.4 mm2 | 2.8 mm | 22 g | 90 g |
| nylon-6,6 drawn fibre | 700 MPa | 0.6 mm2 | 0.9 mm | 2 g | 9 g |

This kills rubber cement outright. Unvulcanised rubber has no permanent
network - the chains slide - so it creeps under sustained load, and its useful
strength is a couple of MPa rather than the 25 MPa a vulcanised rubber reaches
by strain-induced crystallisation. A 19 mm rope needing 3.3 kg of dope per shot
is not a wrist device. SBS is the one that survives, and it survives precisely
because its glassy polystyrene domains act as physical crosslinks - the
two-phase structure the engine pointed at and cannot represent.

**Drying.** No, and not by a small margin. Solvent leaving a filament is
diffusion-limited at `t ~ R^2/D`, with `D ~ 1e-10 m^2/s` in a concentrated
solution:

| filament | evaporation-limited | diffusion-limited | fits 0.6 s flight |
|---|---|---|---|
| 50 um | 0.02 s | 6 s | no |
| 300 um | 0.37 s | 225 s | no |
| 3000 um | 12 s | 22500 s | no |

The surface skins over in milliseconds and the core stays liquid for minutes.
Drying in flight would need filaments around 15 um - which is what a real
textile spinneret makes, and why industrial dry-spinning columns are metres
tall - and 125,000 of them to reach 22 mm2.

**Sticking.** Yes, ironically *because* it does not dry. It arrives wet and
tacky, which is a good adhesive contact and a terrible structural one.

**Clogging.** Yes, in seconds. A stagnant meniscus in still air skins to a
tenth of the orifice radius in 1.4 s at 300 um, 3.3 s at 700 um.

### What the numbers point at instead

Drop the solvent. Heat diffuses about a thousand times faster than solvent
does, so a melt freezes on the timescale a solution cannot dry on:

| filament | freezes | would dry in |
|---|---|---|
| 100 um | 0.02 s | 25 s |
| 300 um | 0.22 s | 225 s |
| 1000 um | 2.5 s | 2500 s |

A 300 um hot-melt filament solidifies in 0.22 s, inside the flight time, and
314 of them give the 22 mm2 SBS needs. No solvent to evaporate, no skinning in
the nozzle, and it bonds on contact because it lands molten - which is how a
glue gun works. The nozzle now freezes rather than skins, which is a heater
problem rather than a chemistry one.

None of this is in the engine's reach: no evaporation kinetics, no heat
transfer, no viscosity for a polymer solution, no creep.

## A material that works on the hardware already specified

The hardware has no heater and one pressure source, so the strand cannot
solidify by cooling and, per the numbers above, cannot solidify by drying
either. That leaves solidifying by reaction - and a reaction is the one route
whose rate does not scale with the square of the filament radius, because it
happens everywhere in the strand at once rather than waiting for something to
travel to a surface.

**Rubber-toughened cyanoacrylate, with an amine accelerator mixed at the
nozzle.** Two barrels into a disposable static mixer instead of one; everything
downstream is unchanged.

It fits the rig as built:

| | |
|---|---|
| spinneret already specified | 40 holes x 0.7 mm |
| cross-section (100% solids, no shrinkage) | 15.4 mm2 |
| working stress at 50 lb | 14.4 MPa |
| safety factor against cured CA at ~25 MPa | 1.7 (2.0 at 46 holes) |
| pressure at 1 Pa.s through a 2 mm land | 6.5 bar, against the 13 available |
| a 3 m shot | 46 mL, 51 g |

The last row is the reason to prefer it over anything dissolved in a solvent:
at 100% solids there is no carrier to haul, so the same shot that cost 203 g as
a 25% dope costs 51 g here. Cure is seconds, in bulk, and it bonds to steel,
glass and skin - which is the safety problem, not a footnote.

### Will it fly as a rope, or as a spray?

The question the rest of this had been dodging. A liquid jet does not stay a
jet: surface tension pinches it into drops, and the only defences are viscosity
and getting solid first.

| jet | viscosity | intact for |
|---|---|---|
| 40 x 0.7 mm bundle | 1 Pa.s | 0.5 m |
| 40 x 0.7 mm bundle | 5 Pa.s | 2.5 m |
| single 4.4 mm | 1 Pa.s | 3.5 m |
| single 4.4 mm | 5 Pa.s | 16 m |

So: **one thick stream, not forty filaments.** Thin jets pinch off in
centimetres, and at a gel-grade viscosity a single 4.4 mm stream stays coherent
far longer than it needs to - it has about 3 s before pinch-off and cures in
about 1.

This also corrects the bundle reasoning above. Small holes cost more pressure,
not less, and the same throughput through one 4.4 mm orifice needs **0.83 bar**
against the bundle's 32 - of the 13 available. The bundle existed only to beat
the drying clock, and a reactive cure has no drying clock.

What does cap it is the air. The gas Weber number reaches ~13 at 9 m/s on a
4.4 mm stream, which is where bag breakup starts and the jet begins shedding
its surface. So the useful muzzle velocity is about 9 m/s and the range is
**4-8 m**, set by aerodynamics rather than by the 13 bar, which could otherwise
push it to 20 m/s.

### Clogging, in two parts

**The orifice, holding unmixed CA.** It cures where moisture reaches it, and
moisture arrives by diffusion, so the plug grows as `sqrt(Dt)`:

| left standing | plug depth | on the 4.4 mm orifice |
|---|---|---|
| 30 s | 0.17 mm | blows out on the next shot |
| 2 min | 0.35 mm | blows out on the next shot |
| 10 min | 0.77 mm | needs clearing |
| 1 hour | 1.90 mm | drill it out |

Minutes are fine, an hour is not. A sealed cap with a desiccant pellet removes
the problem entirely, and that is the whole fix.

**The static mixer.** This one clogs every shot, by design - the point of
mixing is that it gels in about a second, and whatever is in the mixer when you
release the trigger gels there. A 6 x 60 mm tip holds ~1.7 mL and is lost each
time. They cost about 30 cents. Budget one per shot and stop thinking of it as
a failure.

### The problem that neither of those is

It does not cure in flight. At 9 m/s the jet is airborne for 333 ms over 3 m,
and accelerated CA gels in 1-5 s. So the rope arrives **liquid**. It bonds on
contact, which cyanoacrylate does superbly, and then cures while hanging - sagging
meanwhile, and carrying nothing until it has set.

UV was the obvious escape and it fails on arithmetic. A 20 mm LED ring at 9 m/s
gives the jet 2.2 ms of light; a UV acrylate wants 0.3-1 J/cm2, which demands
about 225 W/cm2 - a few hundred times what a cheap LED array delivers.

Flight time and gel time are within about a factor of three to ten of each
other, so more accelerator looked like it might close it. It cannot, and the
reason is heat rather than chemistry.

### Why full cure in flight is thermally impossible

Anionic cyanoacrylate polymerisation is initiation-limited and essentially
non-terminating, so doubling the accelerator halves the set time. The chemistry
allows any speed you like. The heat does not.

Propagation releases about 60 kJ/mol over a 125 g/mol monomer - 480 J/g - and
at a specific heat near 1.5 J/g/K that is an **adiabatic rise of 320 K at full
conversion**. Whether the rope actually sees that rise is a race between curing
and shedding heat:

| section | heat escapes in |
|---|---|
| 0.1 mm glue bond line | 25 ms |
| 4.4 mm rope | 48 s |

A bond line dumps its heat before it can warm up, which is exactly why
cyanoacrylate is well behaved in a joint and smokes on a cotton ball. Our rope
needs 48 s to shed heat and would cure in 0.33 s, so it is adiabatic by a
factor of 145. Every joule stays in:

| conversion | rise | reaches | |
|---|---|---|---|
| 10% | +32 K | 57 C | fine |
| 20% | +64 K | 89 C | hot but survivable |
| 30% | +96 K | 121 C | boils the monomer, foams the rope |
| 100% | +320 K | 345 C | polycyanoacrylate unzips above ~200 C |

**The ceiling is 55% conversion**, and full cure in a thick section is ruled
out by thermodynamics, not by slow chemistry. Both inputs were taken at the
conservative end - a higher enthalpy or a lower specific heat makes it worse,
not better.

### What survives, and how tight it is

The rope does not need to cure. It needs to stop flowing. Chain growth reaches
high molar mass at low conversion, so a few percent converted is already
high-MW polymer dissolved in its own monomer - and that is a gel:

| conversion | viscosity (Mw 1e5 to 1e6) | temperature |
|---|---|---|
| 5% | 2 - 849 Pa.s | 41 C |
| 10% | 24 - 12700 Pa.s | 57 C |
| 15% | 116 - 61600 Pa.s | 73 C |

5% is marginal - at the low end of molar mass it lands under the ~5 Pa.s the
jet calculation wanted. **10% clears it for every molar mass here**, by a
factor of five at worst, at +32 K.

So the operating window is: gel to roughly 10% in flight, land, and finish
curing on the wall over minutes where the heat has time to leave. The window is
real and it is not wide, and the tight side is thermal, not rheological.

This is a calculation and not an experiment. What it settles is which claim is
worth testing: full cure in flight is dead, and partial gelation is live.

### Correction: partial gelation was not live either

"Arrives liquid, bonds on contact, cures while hanging" is wrong, and checking
it is what closed the question. A hanging rope carries its own weight: 3 m of
15.2 mm2 masses 50 g and puts 32 kPa on its own top section. A viscous fluid
under that stress extends at `sigma/3eta`:

| viscosity | strain rate | |
|---|---|---|
| 5 Pa.s | 2200 /s | falls instantly |
| 1e3 Pa.s | 11 /s | falls |
| 1e5 Pa.s | 0.11 /s | falls |
| 1e8 Pa.s | 1.1e-4 /s | holds |

It needs about 1e8 Pa.s, which is a solid, not a gel. The 10% gel of the
previous section is six orders short. A liquid rope does not bridge a gap - it
runs down the wall and drips. **So it must be solid when it lands, which means
full cure in flight, which the exotherm forbids at 4.4 mm.**

### The feasible region, and why it was empty

Four constraints at once, with the filament count always set to give the 15.2
mm2 that carries 50 lb: pressure under 3 bar, jet intact for 3 m, and thin
enough to shed its cure heat within the 333 ms flight (`R^2/alpha <= t`, so
diameter at or under 0.37 mm).

Scanned over diameter from 0.2 to 4.4 mm and viscosity from 0.1 to 20 Pa.s,
**every combination fails at least one**. The bind is structural: shedding heat
wants thin filaments, thin filaments break up unless they are viscous, and
viscous thin filaments need hundreds of bar. Thin, coherent, and pushable are
three properties no geometry holds together.

### What actually opens it

Pressure and breakup are geometry and viscosity, and the single 4.4 mm stream
already passes both - the table shows it failing on the thermal column alone.
So the thermal column is the only one chemistry can move, and it has two terms:
shed the heat, or make less of it.

Making less of it is the available one. The adiabatic rise is heat-per-gram
over specific heat, and heat-per-gram scales with the **reactive mass
fraction**. Neat monomer is all reactive. An oligomer is not:

| reactive fraction | heat/g | adiabatic rise | reaches | |
|---|---|---|---|---|
| 100% (neat monomer) | 480 J | +320 K | 345 C | cooks |
| 50% | 240 J | +160 K | 185 C | hot |
| 30% | 144 J | +96 K | 121 C | edge |
| 20% | 96 J | +64 K | 89 C | fine |

And a 70-80% oligomer loading still extrudes, because an oligomer's overlap
concentration is high - at Mn ~ 3000, 70% gives about 6 Pa.s, which is the
viscosity the jet calculation wanted anyway. At Mn 10000 it is 650 Pa.s and far
too stiff, so the molar mass is a real design variable, not a detail.

Re-testing the 4.4 mm single stream at 20-30% reactive oligomer resin: **1.9
bar of 3, intact for 29 m of the 3 needed, +96 K, 15.2 mm2.** All four pass.

The fix was never a different reaction. It was reacting less of the mass. Same
cyanoacrylate chemistry and same cure speed, but with most of the mass arriving
already polymerised, so it carries the load without having to generate the heat
of making itself.

The window is narrow - 20% reactive is comfortable thermally and near the
pressure limit, 30% is comfortable on pressure and near the thermal limit - and
a thickener loading that high is past what gel cyanoacrylate is sold at, so
this is a formulation direction rather than something off a shelf.

### Can it be bought? The oligomer, yes. The speed, no.

A cyanoacrylate-terminated oligomer is not a product. Cyanoacrylate esters are
made by Knoevenagel condensation and then cracked back out of the polymer, so
putting that end group on an oligomer backbone is a research project, not a
purchase order - five figures before there is a gram.

**The dissolved-polymer shortcut does not rescue it.** Thickening plain
cyanoacrylate with a bought PMMA resin looks like it works on the viscosity
scaling, and the scaling is wrong, because it does not know about the glass
transition. By Fox:

| PMMA loading | blend Tg | at 25 C it is | monomer left | rise |
|---|---|---|---|---|
| 45% | -38 C | a pourable liquid | 47% | +150 K |
| 53% | -24 C | a pourable liquid | 39% | +125 K |
| 65% | 0 C | a thick syrup | 27% | +86 K |
| 85% | **+51 C** | **a solid** | 7% | +22 K |

Above about 53% the blend is not a viscous liquid, it is approaching its own
glass. Everything past that row was fiction. And 53% still leaves 39% monomer,
which is a 125 K rise. Dissolved commodity polymer cannot dilute the heat and
stay pumpable at the same time.

**The backbone is what I had glossed.** An acrylic oligomer is still glassy at
Mn 3000 - Fox-Flory puts it at 38 C. A polyether or polyester backbone is not:
polypropylene glycol at Mn 3000 has a Tg near -70 C and pours. That is exactly
what a urethane-acrylate oligomer is, and they are sold by the drum:

| | Mn | f | viscosity neat | price |
|---|---|---|---|---|
| aliphatic urethane acrylate (Sartomer CN-series) | 1-5k | 2-6 | 2-50 Pa.s | $15-40/kg |
| polyester acrylate (Allnex Ebecryl) | 1-3k | 2-6 | 1-20 Pa.s | $10-30/kg |
| epoxy acrylate | 0.5-2k | 2 | 5-100 Pa.s | $10-25/kg |

Right molar mass, right functionality, right viscosity, right price. **The
oligomer was never the problem.** The problem is that none of them cures fast
enough:

| chemistry | gel time | against the 333 ms flight |
|---|---|---|
| cyanoacrylate + amine accelerator | ~0.3 s | the only one fast enough |
| thiol-Michael, strongly base-catalysed | ~5 s | 15x too slow |
| fast polyurethane, tin-catalysed | ~8 s | 24x |
| acrylic redox (peroxide/amine) | ~40 s | 120x |
| epoxy-amine | ~300 s | 900x |

Cyanoacrylate is the only chemistry that gels in a third of a second, and it is
sold only as the monomer - which is precisely where all the heat is. Every
oligomer that is sold cures one to three orders of magnitude too slowly.

**So this is a kinetics wall, not a catalogue one**, and it is the honest
stopping point for the fired-liquid-rope architecture. Buying a different
oligomer does not move it; only a fast chemistry on a long backbone would, and
that is the thing nobody sells.

## Hot melt, dismissed early for the wrong reason

Hot melt was ruled out near the top of this file on "the hardware has no
heater", which is a choice rather than a constraint - a heater is the cheapest
part on the list. Revisiting it turns up an asymmetry that the chemical route
does not have.

A curing strand has to react all the way through before any of it is solid, so
its clock is the bulk clock `R^2/alpha` - 48 s on a 4.4 mm strand, which is what
forced full cure in flight and then made the exotherm impossible. **A cooling
strand does not.** Its surface hits air temperature almost at once and a solid
skin grows inward as `sqrt(alpha*t)`. Surface tension cannot pinch a solid, so
breakup stops when the skin forms, not when the strand is solid through.

| time | skin | solid area | holds |
|---|---|---|---|
| 10 ms | 32 um | 0.4 mm2 | 2 lb |
| **333 ms (lands)** | **182 um** | **2.4 mm2** | **14 lb** |
| 1 s | 316 um | 4.1 mm2 | 23 lb |
| 10 s | 1000 um | 10.7 mm2 | 60 lb |
| 48 s | solid through | 15.2 mm2 | 85 lb |

On landing it has to carry only itself - 3 m of strand is 50 g, or 0.11 lb -
and a 182 um skin is three orders of magnitude past that. Full payload takes
about ten seconds of standing. Fire, wait, then load it.

And a bought glue stick is strong enough, if you pick the right one:

| grade | UTS | holds on 15.2 mm2 | applied at |
|---|---|---|---|
| EVA craft stick, low temp | 6 MPa | 21 lb | 120 C |
| EVA craft stick, high temp | 10 MPa | 34 lb | 190 C |
| polyolefin / APAO | 12 MPa | 41 lb | 170 C |
| **polyamide (3M Jet-melt, Henkel Macromelt)** | **28 MPa** | **96 lb** | 200 C |
| **thermoplastic polyurethane hot melt** | **40 MPa** | **137 lb** | 180 C |

Pressure is fine - a 10 Pa.s melt needs 3.4 bar of the 6 available, and only
above about 20 Pa.s does it stall. The one real hardware change is the
reservoir: 137 mL/s is far past what a glue gun melts on demand, so the 45 g
has to be pre-melted. That is 20 kJ from cold, 5.5 Wh, or 66 W over five
minutes - a 100 W cartridge heater in an aluminium tube and a small LiPo.

**What it buys**: every part is a catalogue item. No commissioned resin, no
two-part metering, no static mixer, no pot life, no exotherm, no gel clock to
hit, and it bonds on contact because it lands molten.

**What it costs**: a heater and a battery; a nozzle that plugs solid if it ever
goes cold; a burn hazard at 200 C; and ten seconds of standing before the
strand carries its rated load.

## Putting the hot melt to the engine instead of to a calculator

Everything in the two sections above was worked by hand, which is how this
project drifted out of the engine it was meant to be testing. The specs written
early on asked only for what the engine already covered, so the questions that
actually decided the design were never put to it.

`hotmelt.yaml` asks for all of it at once - melt viscosity at the nozzle
temperature, melting point, melt surface tension, strength, entanglement,
modulus, density and adhesion - and the run names its own gaps:

```
No expert covers: melting_point, shear_viscosity, surface_tension
  - shear_viscosity: eliminated 57 candidate(s), 57 because the value
    could not be predicted
```

Three properties in the registry with no polymer expert behind them. All three
are now served by `polymer_melt`, and each is built from something already in
the repository: melting points from a short measured table, surface tension
from the same Owens-Wendt components the adhesion expert reads plus a melt
temperature correction, and viscosity from the reptation power law over the
entanglement molar mass the mechanical expert already computes, shifted by WLF
from the Tg the thermal expert already predicts. No fitted constant: the
anchor is the viscosity that *defines* the glass transition.

### What the engine then said, and how much of it I had wrong

**A commercial chain length is far too thick to fire.** At an ordinary 50
kg/mol the pool spans 53 Pa.s to 2e9 Pa.s against a 1-20 Pa.s window. My hand
figure of "10 Pa.s hot melt" was right about hot-melt *adhesives* and wrong
about polymers: an adhesive is low molar mass cut with tackifier and wax
precisely to get the viscosity down, which is a formulation fact the engine
made visible and the calculator hid.

**The melting-point refusals are the useful half.** An amorphous polymer has
no melting point - polystyrene softens through its glass transition over tens
of degrees and offers a melt process no set point at all - so the expert
refuses it, and refuses it with a different reason from a polymer that is
simply not in the table. One is a physical fact, the other a coverage gap, and
a run that conflated them would be misleading.

**And the viscosity model cannot reach a hot-melt nozzle.** WLF is referenced
to Tg and holds for perhaps 100-150 K above it; a hot melt runs 200-300 K
above. So the expert refuses, which is correct and is also a real limit on the
answer: covering that regime needs an Arrhenius branch with per-polymer flow
activation energies that this repository does not have. The hand analysis above
simply assumed a viscosity. The engine will not.

### Iterating to an answer

Two more gaps had to close before the run returned anything, and both were
found by the engine refusing rather than by anyone guessing.

**WLF could not reach the nozzle.** The viscosity model is referenced to Tg and
holds perhaps 100 K above it; a hot melt runs 200-300 K above. The expert now
hands over at `Tg + 100 K` to an Arrhenius branch with a tabulated flow
activation energy, joined so the value is continuous. Checked against
polyethylene: a 50 kg/mol grade comes out at 1950 Pa.s at 200 C, where measured
HDPE is one to five thousand.

**The glass/rubber switch is wrong for anything crystalline.** This is the one
that had been distorting the whole project. The switch asks what the *amorphous*
phase is doing, and for a semicrystalline polymer that is the wrong question -
crystallites have no glass transition to be above. Polyethylene sits 100 K past
its Tg at room temperature, so the switch returned 8 MPa. Measured HDPE is about
1 GPa. Out by a factor of 125, in the direction that makes every useful
semicrystalline polymer look useless.

That is why polyethylene came bottom of the earlier strand ranking, and why
"glassy or tough" looked like a dichotomy: **it is a dichotomy for amorphous
polymers only.** Semicrystalline polymers are the third case the switch had no
branch for, and they are stiff *and* tough, which is exactly what the strand
needed the whole time. The measured modulus is now used where there is one.

Two data hazards surfaced on the way. Atactic polypropylene - which is what the
bundled reference set actually carries - is a tacky amorphous solid, and keyed
on the repeat unit alone it is indistinguishable from the isotactic polymer that
melts at 165 C. The run was handing it that melting point and a 1.5 GPa modulus.
Tacticity is now checked, and an unstated tacticity is refused rather than
assumed.

### The answer

**Linear polyethylene at 5-8 kg/mol** - a low-molar-mass HDPE, which is what a
polyethylene wax is. One of 57 feasible:

| | | |
|---|---|---|
| melt viscosity at 200 C | 11.4 Pa.s | in the 1-20 window |
| melting point | 408 K (135 C) | in the 120-210 window |
| melt surface tension | 22.5 mN/m | low, so the jet resists pinch-off |
| Young's modulus | 1.0 GPa | measured, semicrystalline branch |
| flaw-free strength bound | 100 MPa | screen only, not a strength |
| entanglement | 7x the entanglement mass | entangled, so not brittle |
| density | 910 kg/m3 | the lightest in the set |
| work of separation on steel | 63 mN/m | marginal, and flagged as such |

The molar mass is the whole answer and the engine found it. An ordinary
commercial 50 kg/mol grade is 2000 Pa.s and will not go through the orifice at
any pressure on the list; below about 5 kg/mol the chain stops being entangled
and the strand turns brittle. The window is narrow and sits an order of
magnitude below where anyone would have reached by default.

What the run does **not** establish: `theoretical_strength` is a flaw-free bound,
so nothing here predicts a breaking load, and at 8 kg/mol a real polyethylene is
much weaker than a commercial grade - low molar mass buys pumpability by
spending strength. The adhesion figure meets its bound only within uncertainty.

## Blends, and what the single-polymer answer was hiding

The polyethylene answer above is forced rather than chosen. Melt viscosity goes
as the 3.4 power of chain length and strength rises with it too, so on one knob
the only chain thin enough to extrude is also the weakest one that will still
entangle. 5-8 kg/mol was not a design; it was the width of a trap.

A blend gives a second knob, which is what every hot-melt adhesive is and what
bimodal polyethylene is inside one reactor. The engine could already
*represent* one - `MixtureComponent` has carried a `polymer` field since Phase
4 - but nothing proposed a polymer blend and no expert scored one. The existing
`mixture` expert declares a polymer component out of domain in as many words,
correctly: its rules are volume fractions of small-molecule liquids.

`PolymerBlendExplorer` proposes bimodal blends - one chemistry, two chain
lengths, and the ratio - plus cross-polymer pairs that mostly get refused.
`polymer_blend_melt` and `polymer_blend_solid` score them: log-additive
viscosity, Voigt-Reuss modulus bounds, Fox glass transition, the crystalline
component's melting point, additive density and surface tension.

**What it buys, which is the whole point:**

| | melt viscosity at 200 C | load-bearing chain |
|---|---|---|
| single polyethylene, 8 kg/mol | 3.8 Pa.s | 8 kg/mol |
| single polyethylene, 20 kg/mol | 87 Pa.s - will not extrude | - |
| **20 kg/mol + 30% of a 0.8 kg/mol wax** | **16 Pa.s** | **20 kg/mol** |

Two and a half times the backbone at a viscosity that still goes through the
nozzle. `hotmelt_blend.yaml` returns 3 feasible of 120, led by exactly that
blend.

Three things the rules deliberately will not do. **Modulus is bounded, not
predicted** - Voigt and Reuss are rigorous for any two-phase arrangement, and
where a real blend falls between them is a question about morphology that
nothing here knows, so the uncertainty spans the bounds instead of a point
estimate implying one. **An immiscible pair is refused** rather than averaged:
a chain gains almost no entropy on mixing, so most polymer pairs form two
phases and track the continuous one. **A blend has no melting point of its
own** - it has its crystalline component's, depressed by dilution, and the
depression is carried as uncertainty because Flory's expression needs an
interaction parameter this repository does not have.

### Is it a wax? Does it stretch?

The first blend spec could not say, because moving to mixtures had quietly
dropped `entanglement_molar_mass` - the only ductility axis the registry has,
and the one that separates a polymer that draws under load from a wax that
crumbles. It is served for a blend now, and not by averaging: a short chain
does not join the network, it **dilutes** it, so the blend's entanglement mass
is the entangling fraction's divided by how much of the blend that fraction is.

| blend | blend Me | entanglements per chain | |
|---|---|---|---|
| 20k + 15% of 0.8k | 986 g/mol | 20.3x | stretches |
| **20k + 30% of 0.8k** | **1197 g/mol** | **16.7x** | **stretches** |
| 20k + 45% of 0.8k | 1523 g/mol | 13.1x | stretches |
| 20k + 75% of 0.8k | 3351 g/mol | 6.0x | stretches |
| 20k + 95% of 0.8k | 16756 g/mol | 1.2x | brittle, snaps |
| 2k + 30% of 0.8k | 1197 g/mol | 1.7x | brittle, snaps |

So the working blend is not a wax. 30% diluent leaves the backbone sixteen
entanglements long, against the two a load-bearing network needs, and the wax
only wins above about 90%. A blend with no entangling component at all is
refused outright - "it is a wax, not a polymer, whatever its stiffness says" -
which is a different answer from failing the requirement, and the two read
differently on purpose.

What this still does not establish is how far it stretches. Entanglement says a
network exists, not that a specimen elongates by any particular percentage, and
there is no elongation at break, yield stress or fracture toughness in the
registry. A semicrystalline polymer also draws by crystal slip, which nothing
here models - so the real ductility is under-described rather than predicted.

And one thing the blend spec still cannot ask: **adhesion**. No expert serves
`work_of_separation` for a mixture, so the blend is scored without the
requirement the whole device rests on. That axis still only exists on the
single-polymer spec.

### Two more bugs, both the same shape

Adding the blend expert turned up the condition-resolution problem twice more,
in the same shape as the one below.

An expert is evaluated **once, at one set of conditions**. The first blend
expert served melt viscosity at 200 C *and* modulus at 25 C, so whichever
condition won, the other property was answered at the wrong temperature - and
what came back was a density refused for being asked at 200 C, correctly, since
its packing factor was fitted at room temperature. That refusal then took out
the entanglement mass and the viscosity behind it. The fix is that they are two
experts, because they are two states of the same material.

The same thing again one level down: a delegating expert must resolve
conditions for its *components*. Passing the melt temperature to every
sub-expert refused the density for the same reason. A dependency is now
evaluated at the candidate's own conditions and only the requested property
carries the requirement's.

Also fixed: a delegating expert has to compute its own dependency closure.
Asking the panel for `shear_viscosity` alone selects only the expert that
serves it, whose dependencies then arrive empty and which correctly refuses.
The engine does this closure for a top-level run; nothing was doing it for a
sub-run.

### A bug the exercise found

Two requirements on one expert both stating `temperature: 200 degC` were
treated as a *disagreement*, because `_conditions_for` counted the stated
conditions rather than comparing them. The run silently fell back to the
spec-level 25 degC and reported a room-temperature surface tension against a
requirement that had plainly asked for a melt - with nothing anywhere saying a
stated condition had been dropped. Agreement is now decided by comparing
condition identity, so the same temperature spelled two ways is one condition.

## The bench test, before any of this

`jet_test.py` prints the protocol. Whether a 4.4 mm stream flies 3 m as a rope
is set by viscosity, surface tension, density and speed, not by what the fluid
is made of - so corn syrup answers it for the price of a bottle of corn syrup,
and the answer transfers to any resin at the same viscosity.

| fluid | viscosity | predicted intact for |
|---|---|---|
| water (the control) | 0.001 Pa.s | 0.4 m |
| thin syrup | 1 Pa.s | 2.8 m |
| corn syrup at target | 10 Pa.s | 24.6 m |

The control matters more than the test. **Water must break up inside half a
metre.** If it flies, the rig is not doing what you think and the syrup result
means nothing.

Thin the syrup until a 5 mm steel ball falls 100 mm in about 11 seconds - that
is 10 Pa.s, and it is the only measurement that has to be careful. 3.5 bar from
a PET bottle and a bike pump gives 9 m/s. Film at 240 fps against floor marks.

Breaking under 1 m would mean the jet model is wrong by an order of magnitude
and that firing a liquid rope does not work at this scale - which is worth
knowing for the price of a bottle of syrup, and would invalidate most of this
file.

### The oligomer has to be reactive, not merely present

A dissolved inert oligomer buys the thermal headroom by throwing the strength
away. At Mn 3000 it is a third of PMMA's 9200 g/mol entanglement mass, so it
cannot carry load through a network - it is a plasticiser. With only 25-30% of
the mass forming the network, the rope lands at 6-8 MPa against the 25 needed.

So the oligomer must be **functionalised**: reactive groups on an
already-long chain, so that only the ends generate heat while the whole chain
joins the network. Heat then falls with molar mass while strength does not:

| Mn | functionality | heat/g | rise at full cure | rise at gel point |
|---|---|---|---|---|
| neat monomer | - | 480 J | +320 K | +320 K |
| 3000 | 2 | 40 J | +27 K | +27 K |
| 3000 | 3 | 60 J | +40 K | **+20 K** |
| 6000 | 3 | 30 J | +20 K | +10 K |

Functionality above two matters twice over: a trifunctional oligomer gels at
`1/(f-1)` = 50% of its end groups rather than needing near-full conversion, so
it is solid at half the heat.

### The mixture

Monomer is the reactive diluent. More of it thins the resin and speeds the
cure, and it is also where nearly all the heat comes from, so it is the single
variable that sets the window:

| monomer | oligomer | rise | reaches | viscosity | pressure | |
|---|---|---|---|---|---|---|
| 5% | 87% | +33 K | 58 C | 13.8 Pa.s | 4.6 bar | ok |
| 8% | 84% | +42 K | 67 C | 12.1 Pa.s | 4.0 bar | ok |
| **12%** | **80%** | **+54 K** | **79 C** | **10.0 Pa.s** | **3.4 bar** | **ok** |
| 15% | 77% | +63 K | 88 C | 8.6 Pa.s | 3.0 bar | ok |
| 19% | 73% | +75 K | 100 C | 7.0 Pa.s | 2.5 bar | too hot |
| 25% | 67% | +93 K | 118 C | 5.0 Pa.s | 1.9 bar | too hot |

At a 3 bar hand-sprayer budget the window collapses to a single point at 15%.
Going to a 6 bar CO2 cartridge opens it to 5-15%, which is the difference
between a formulation that has to be exact and one that has tolerance. Take the
middle:

**Stream A**

| | % w/w |
|---|---|
| cyanoacrylate-terminated oligomer, Mn ~3000, f ~3 | 80 |
| ethyl 2-cyanoacrylate monomer, as reactive diluent | 12 |
| dissolved elastomer, rubber toughener | 7 |
| acidic stabiliser (SO2 or methanesulfonic acid) | 1 |

**Stream B**, metered 20:1 against A: amine accelerator, 2-5% in an inert
carrier. This is the knob that sets the gel clock against the flight time.

| | |
|---|---|
| viscosity | 10 Pa.s |
| pressure | 3.4 bar of 6 |
| exotherm | +54 K, reaching 79 C against a 175 K ceiling |
| cross-section | 15.2 mm2, one 4.4 mm orifice |
| holds | 85 lb at 25 MPa - 1.7x the 50 asked |
| a 3 m shot | 46 mL, 50 g (43 mL A + 2.2 mL B) |

The pressure source goes back up from the hand sprayer to the CO2 cartridge,
and that is the only hardware change from the list above.

What is not established: the oligomer is specified by what it has to do rather
than named as a product, because a trifunctional cyanoacrylate-terminated
oligomer at Mn 3000 is not something sold in a bottle. Every number here is a
calculation from handbook physical constants, and the gel clock against the
333 ms flight is the one that most needs an experiment rather than an estimate.

## Simulating it: `simulate.py`

Everything above is a scaling argument, and scaling arguments treated three
things separately that are in fact racing each other - cure makes heat, heat
accelerates cure, and cure raises the viscosity that resists pinch-off. So
`simulate.py` integrates a radial reaction-diffusion problem for one
cross-section of the jet across the flight, with a Rayleigh perturbation
growing against the viscosity the cure is producing. It is not a free-surface
solver: the radius is fixed and breakup is judged by linear stability, which is
the standard slender-jet treatment and is valid up to pinch-off, not through
it.

It reproduces both limits that can be solved by hand - the adiabatic peak at
79.40 C and the uncured perturbation growth at 7.50% - so what the integrator
adds is the coupling, not the arithmetic.

**It corrected one prediction.** The surface was supposed to cure last, being
the cooler part, and that would have mattered because surface tension acts on
the surface. It does not happen: heat needs 48 s to cross the strand and the
flight is 333 ms, so there is no time for a radial gradient to form at all.
Core and surface converge together, 0.993 against 0.992.

**And it found a constraint no scaling argument could see.** Cure starts where
the two streams meet, not at the muzzle, and a 6 x 60 mm mixer at 137 mL/s
holds the resin for 12 ms before it reaches the orifice. A fast accelerator
wins the flight and loses the nozzle:

| gel time | conversion at exit | viscosity there | pressure | |
|---|---|---|---|---|
| 50 ms | 0.158 | 46 Pa.s | 14.0 bar | stalls the flow |
| 100 ms | 0.082 | 21 Pa.s | 6.6 bar | stalls the flow |
| **150 ms** | 0.056 | 16 Pa.s | 5.2 bar | works |
| **300 ms** | 0.028 | 13 Pa.s | 4.2 bar | works |
| **500 ms** | 0.017 | 11 Pa.s | 3.9 bar | works |
| 1000 ms | 0.009 | 11 Pa.s | 3.6 bar | lands liquid |

**Working window: a gel time of 150-500 ms**, bounded below by the mixer and
above by the flight. The two limits belong to different hardware, so both are
tunable, and the cheapest move is a shorter mixer rather than a new resin.

What the simulation does not do is check its own inputs. The cure kinetics, the
gel exponent and the 30 J/g all went in as assumptions and all come back out
inside the answer. It tests whether the physics is consistent, not whether the
resin exists.

### The cheap experiments that would test it

Three, none of which need the exotic oligomer, because each isolates one
mechanism at matched dimensionless groups:

- **Jet coherence.** Fire glycerol or corn syrup at 10 Pa.s through the 4.4 mm
  orifice at 9 m/s and photograph it. Same Ohnesorge and Weber numbers as the
  real resin, no chemistry at all. If it does not fly as a rope, nothing
  downstream matters.
- **Exotherm.** Cure a known mass of plain cyanoacrylate in a vacuum flask with
  a thermocouple. It should give 480 J/g and a ~320 K rise. This is the number
  the whole oligomer argument rests on, and it is a one-afternoon measurement.
- **Gel clock.** Mix cyanoacrylate and accelerator at several ratios and time
  it to immobility with a stopwatch. Directly locates you inside or outside the
  150-500 ms window.

The first two are decisive and cost nothing. They should happen before anyone
tries to source a trifunctional oligomer.

Honest caveats beyond that: cured cyanoacrylate is brittle unless it is the
rubber-toughened grade, which is again a two-phase material the engine cannot
represent; and CA bonds skin instantly, which is the real safety issue.

### A data bug found on the way

Asked about the monomer, the panel disagreed with itself by 166 K on the
boiling point - `measured` said 328.1 K and `joback` said 493.7 K. The measured
figure is right as a number and wrong as a property: ethyl cyanoacrylate boils
at 55 degC *under vacuum*, and at one atmosphere it polymerises rather than
boiling. A reduced-pressure boiling point has been compiled as a normal one.

That it is a bad record rather than ordinary model scatter is checkable. Across
the 46 reference compounds carrying both a measured and a Joback boiling point,
the median disagreement is 7.6 K and the worst is 117 K:

| compound | measured | joback | diff |
|---|---|---|---|
| gamma-butyrolactone | 477.8 K | 360.8 K | -117 K |
| n-methylpyrrolidone | 477.4 K | 409.9 K | -67 K |
| ethylene glycol | 470.3 K | 429.7 K | -41 K |
| **ethyl cyanoacrylate** | **328.1 K** | **493.7 K** | **+166 K** |

The engine has no cross-check for this. `measured` carries a +/- 0.5 K
uncertainty and wins dispatch on that basis, so 328.1 K would be taken as fact
- and a spec asking for a carrier boiling between 50 and 105 degC would admit
superglue as a solvent. Two experts disagreeing by twenty times the median is a
signal the evaluation layer currently throws away.

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

**A bundle**, which is also what a spinneret and a spider's dragline both are.
The reason is drying: solvent escapes a filament on a `R^2/D` clock, so thin
beats thick. It costs pressure to do it - `dP` goes as `1/R^2` per hole, so
small holes are *worse*, not better - and the bundle is worth that cost only
while drying is what sets the strand. Under a reactive cure it is not, and the
section below drops the bundle entirely.

| filaments | dry filament | solids | viscosity | pressure | flow | holds at 100 MPa |
|---|---|---|---|---|---|---|
| 1 | 3000 um | 30% | 10 Pa.s | 2 bar | 132 mL/s | 159 lb |
| 40 | 300 um | 30% | 10 Pa.s | 191 bar | 53 mL/s | 64 lb |
| **40** | **300 um** | **20%** | **1 Pa.s** | **13 bar** | **80 mL/s** | **64 lb** |
| 200 | 150 um | 20% | 1 Pa.s | 50 bar | 100 mL/s | 79 lb |

The 13 bar design point is the cheap one, and it is cheap because of the
viscosity, not the hardware: 20% solids in acetone is a thin dope, and 13 bar
is a bicycle pump into a PET bottle.

That was the solvent-dope build. The reactive version needs far less, because
the pressure collapsed once the bundle went away:

| | solvent dope, 40 x 0.7 mm | cyanoacrylate, one 4.4 mm |
|---|---|---|
| pressure | 13 bar | **1.9 bar** (1.5 viscous + 0.45 kinetic) |
| flow | 80 mL/s | 137 mL/s |
| a 3 m shot | 203 g | **46 mL, 51 g** |
| shot duration | 600 ms | 333 ms |

1.9 bar is a hand-pumped garden sprayer, not a pressure vessel. Better still,
a mechanical plunger does it and guarantees the mix ratio regardless of
pressure: a 30 mm bore barrel needs 137 N at 19 cm/s, which is a spring or a
lever, not a compressor.

**Parts list:**

- **Dual cartridge and gun**, 10:1 or 20:1, the standard two-part adhesive
  kind - this is the pressure source *and* the metering in one part, ~$25
- **Static mixer tips**, one per shot, ~$0.30 each
- **Orifice**: drill a mixer tip out to 4.4 mm with a 2 mm land. Free
- **Spring or lever drive** for a repeatable 137 N, ~$10
- **Sealed cap with a desiccant pellet** - this is what stops the orifice
  clogging, ~$2
- **Rubber-toughened cyanoacrylate and accelerator**, ~$15

Under $60. One 60 mL cartridge is one 3 m shot.

The hard part is not on this list. It is that the rope arrives liquid.

## The answer, once the engine could see the whole library

Three things changed between the sections above and this one, and together
they turned "no candidate, and here is a coverage gap" into "a candidate, and
here is the part you cannot buy".

**The polymer class was unreachable.** `ReferenceDatabaseExplorer` proposes
molecules and `PolymerBlendExplorer` proposes mixtures, so a specification
saying `material_classes: [polymer]` drew an *empty pool* - which reads like
"nothing feasible" and is not. `PolymerLibraryExplorer` now sweeps the 57
bundled repeat units across six chain lengths, and both tacticities where
tacticity decides whether the polymer crystallises.

**Four lookup tables were gating everything.** A chain dimension, a flow
activation energy, a surface energy and an Owens-Wendt split each answered for
a dozen polymers and refused the rest, and each refusal propagated: no chain
dimension meant no entanglement mass meant no viscosity meant no relaxation
time meant no strain hardening. All four are now predicted from structure.
Over the whole 546-candidate pool, fourteen of the fifteen polymer properties
score for every single candidate. The one that does not is the melting
*temperature*, and the module docstring says why.

**The pressure model had an escape hatch.** It was Hagen-Poiseuille through
the die land alone, so shortening the land drove the pressure to nothing. Swept
over geometry, the search walked straight into it and proposed a 50 um orifice
plate. A real die pays a Bagley entrance drop worth about five radii of
equivalent land whatever its land is - a millimetre for a 400 um hole, twenty
times the land the search wanted. Every short-land pressure this example
reported before that fix was low by that factor.

### What the run says now

At `die 1000 um, land 0.2 mm, draw 100, 20 m/s, 200 holes`, with the honest
pressure:

```
No candidate satisfied every hard constraint (546 evaluated).

Constraints never satisfied together by any candidate in this pool:
  - extensional_strain_hardening and extrusion_pressure
  - shear_thinning_ratio        and extrusion_pressure
  - terminal_relaxation_time    and extrusion_pressure
```

Three pairs, one name on the right of all three. That is not three problems.
Strain hardening, shear thinning and a usable relaxation time are the same
requirement seen from three sides - they all need a long terminal relaxation
time, which needs a high zero-shear viscosity, which is what costs pressure.
The 15 bar cartridge is the single binding constraint, and it is hardware.

Solidification time and filament stability, which killed every earlier attempt
in this document, are no longer binding at all: 16 ms to set against a 100 ms
budget, and the worst filament in the pool survives capillary breakup by 1.2x.
The geometry solved those.

### So how much pressure does it need?

Lift the cartridge limit and change nothing else:

```
73 candidates pass every other hard constraint.

      bar    eta Pa.s      Wi    thin  polymer
      102         593     0.7     3.6  poly(alpha-methylstyrene) 20 kg/mol
      110         637     0.5     3.2  poly(1-butene) 400 kg/mol
      176    1.02e+03     0.7     3.6  poly(methyl acrylate) 400 kg/mol
      177    1.02e+03     0.7     3.6  poly(vinyl acetate) 400 kg/mol
      264    1.53e+03     1.7     5.9  poly(ethyl methacrylate) 400 kg/mol
```

**102 bar**, against the 15 the specification assumed. That is the finding.

It is worth being precise about what it means, because 102 bar is not an
impossible number - a CO2 cartridge sits near 57 bar at room temperature and a
compressed-air cylinder of the kind paintball hardware uses runs at 200 to 300.
A wrist-mounted source at 100 bar exists. A 15 bar regulator was the
assumption, and it was the wrong one.

The load is not the problem either. 200 filaments drawn to 100 um is 1.57 mm^2
of bundle, which at 300 MPa - a drawn, oriented fibre, not a cast bar - holds
**106 lb**, against the 30-50 asked for.

### The caveat that matters

Look at the Weissenberg column. The cheapest candidates sit at 0.5 to 0.8
against a hard lower bound of 0.5. They pass, but with no margin, and
`extensional_strain_hardening` here is a Weissenberg number rather than a
measured Trouton ratio - right sign, right order, not an extensional
rheometer. The first entry with real margin, poly(ethyl methacrylate) at
Wi = 1.7, costs 264 bar rather than 102.

So the honest reading is not "poly(alpha-methylstyrene) at 102 bar". It is:
**the chemistry and the load are solved, and the device is a pressure-source
problem in the 100-250 bar range** - with the top of that range buying the
margin the bottom does not have.

And three acceptance criteria for the finished fibre still cannot be stated at
all, because the registry has no name for them: tensile strength, chain
orientation after drawing, and creep. A run here says whether a dope can be
spun. It does not say whether the fibre it produces reaches 300 MPa or holds a
hanging load without cold-flowing, and the 106 lb above assumes the first of
those.

## Can the engine name the OPTIMAL fluid?

No. It can rank, and it does, and the ranking is worth reading - but "optimal"
is a stronger claim than anything here supports, and the engine's own output
says why rather than leaving it to be assumed.

`optimal.yaml` is the whole ask in one specification, which was not previously
possible. Three things had to change first: a polymer can now be asked how hard
it sticks (the Owens-Wendt split is predicted from polar surface area rather
than looked up for nine polymers), whether it can be made (the repeat unit is
reconstructed to its monomer and scored), and what it costs to push (the
entrance pressure no longer vanishes with the die land). Eleven requirements,
546 candidates, the geometry the sweep found.

```
formulate run examples/web_shooter/optimal.yaml --top 8
python examples/web_shooter/robustness.py
```

### The ranking

```
Ranked 546 candidates (12 feasible, 534 infeasible)

#1  poly(vinyl methyl ether) 400 kg/mol      [*]CC(OC)[*]
#2  poly(1-butene) 400 kg/mol                [*]CC(CC)[*]
#3  poly(alpha-methylstyrene) 20 kg/mol      [*]CC(C)(c1ccccc1)[*]
#5  poly(methacrylonitrile) 2 kg/mol         [*]CC(C)(C#N)[*]
```

### Why that is not an optimum

```
12 feasible of 546 evaluated
0 clear every hard bound robustly, i.e. one sigma the wrong way too

requirements that hold only inside their own error bar:
   extensional_strain_hardening      12 / 12
   shear_thinning_ratio              12 / 12
   terminal_relaxation_time          12 / 12
   shear_viscosity                   12 / 12
   work_of_separation                12 / 12
   extrusion_pressure                 8 / 12
```

Not one of the twelve survives a single standard deviation in the unfavourable
direction, and five requirements are marginal for **every** one of them. A
ranking whose whole feasible set sits inside its own error bars is ordering
noise, and the first entry beats the second by an amount smaller than either
one's uncertainty.

The relative one-sigma on the leader says where it comes from:

```
   shear_viscosity                      2515%   via polymer_melt
   terminal_relaxation_time              100%   via spinline
   extensional_strain_hardening           60%   via spinline
   shear_thinning_ratio                   50%   via spinline
   extrusion_pressure                     50%   via spinline
   ...
   glass_transition_temperature           14%   via polymer_tg
   amorphous_density                       4%   via polymer_density
```

One number, and four of the five universally-marginal requirements are its
children. The relaxation time is `eta0 / G_N`; strain hardening is that time
against the spinline strain rate; the shear-thinning ratio is the zero-shear
viscosity over the apparent one; the extrusion pressure is linear in it. So
this is not five uncertain properties. It is one uncertain property, counted
five times.

That 2515% is a factor of ten in each direction and it is not a defect to be
tuned away. It is what the universal WLF constants cost: checked against
polystyrene at 200 C the form lands about an order of magnitude low, and
`melt.py` says so rather than claiming better. Getting a real optimum out of
this specification means cutting that bar, and there are only two ways to do
it - measure the melt viscosity of the shortlist, or replace universal WLF
constants with polymer-specific ones, which is the same thing done once per
polymer instead of once per candidate.

**So the engine's answer is a shortlist of four chemistries and an instruction:
measure their melt viscosity at 200 C.** That is a smaller and more useful
answer than a name, and it is the one the numbers support.

### Three further reasons the word "optimal" does not apply

**The search space is 57 repeat units.** Ranked over 57 chemistries, the best
of 57 is what you get. The property predictors no longer need a lookup for
anything on the gating path, so this limit is now the explorer's rather than
the panel's - which is the one place left where "everything" could actually
mean everything.

**The leader would cold-flow.** Poly(vinyl methyl ether) has a glass transition
near -31 C, so at room temperature under a hanging load it creeps. It is sold
as an adhesive tackifier, which is exactly why it scores well on the
requirement that is in the registry. Creep is not in the registry, so nothing
penalised it. The engine is not wrong here; it answered the question it was
asked, and the question was incomplete.

**Tensile strength and chain orientation are also absent.** The 106 lb figure
assumes a drawn fibre reaches 300 MPa. Nothing here establishes that.

## And as a blend? The same answer, reached a different way

`optimal.yaml` asked which single polymer. `optimal_blend.yaml` asks which
BLEND, which is the question a real dope is an answer to: a single polymer
couples its melt viscosity to its strength through one variable, chain length,
and a blend of the same polymer at two chain lengths breaks that coupling. The
long fraction carries load, the short one thins the melt. That is bimodal
polyethylene, and it is the obvious lever on the one constraint `optimal.yaml`
found binding.

Getting the question asked at all took two fixes, and both were the same class
of bug: something that looked like an answer and was not.

**A mixture specification drew an empty pool.** The deterministic coordinator's
default explorers were the reference database and the polymer library, so
`material_classes: [mixture]` proposed nothing and the run reported "0 of 0
candidates satisfy every hard constraint" - which reads as a search that found
nothing feasible rather than a search that never ran. Exactly the hole the
polymer class had before `PolymerLibraryExplorer`, still open for mixtures.

**Nothing could tell a polymer blend whether it can be made.** `mixture_thermal`
covers `synthetic_accessibility` for a formulation, but it delegates to the
MOLECULAR panel, which refuses a polymer component with the correct reason - "a
polymer has no critical point and no boiling point". So one hard requirement
eliminated all 2000 candidates for a value nobody could compute.
`PolymerFeasibilityExpert` now answers for a blend whose components are all
polymers, by the rule `mixture_thermal` already uses: the hardest component
gates the formulation. A blend of a polymer with itself scores exactly what
that polymer scores, because blending is a processing step and the monomer is
the same monomer. A polymer dissolved in a SOLVENT is still refused - what has
to be made is the polymer, and charging the formulation for a solvent nobody
synthesises would be the wrong question.

### The answer

```
proposed 2000   fully scored 1329   feasible 21   robust 0

#1  poly(acrylic acid), bimodal 50/0.8 kg/mol at 15% short   [*]CC(C(=O)O)[*]
#2  poly(acrylic acid), bimodal 50/2 kg/mol at 15% short
#3  poly(4-methylstyrene), bimodal 120/2 kg/mol at 30% short
#5  poly(alpha-methylstyrene), bimodal 50/0.8 kg/mol at 30% short
```

Twenty-one feasible blends, and **zero robust** - the same result the single
polymers gave, for the same reason. The leader's numbers are almost
indistinguishable from `optimal.yaml`'s: Weissenberg 0.616 against 0.622,
thinning ratio 3.39 against 3.41, relaxation 2.18 ms against 2.19 ms. The blend
did not escape the viscosity uncertainty, because a blend's viscosity is a
log-additive mix of its components' and each component carries the same decade.

### A defect this run found, which mattered to the answer

The blend expert reported its viscosity bar as `value * 1.5`, behind a comment
claiming it carried "the components' own decade". It carried nothing. It never
looked at the components, and a decade in log space is a factor of about 4.5 in
linear units rather than 1.5. Measured:

```
PE 50 kg/mol                  5800 Pa.s  +/- 2.844e+04  =  490%
PE 2 kg/mol                 0.1024 Pa.s  +/-     0.5023  =  490%
bimodal 50/2 at 15% short     1123 Pa.s  +/-       1685  =  150%   <-- wrong
```

A blend claiming to be three times better known than either thing it is made
of. That is the worst kind of error in this repository, because section 12 has
the ranker rank on uncertainty: an under-claimed bar does not merely mislead a
reader, it reorders the answer. The blend viscosity spread is now combined in
log space, where the mixing rule is linear, and the components are treated as
FULLY correlated rather than independent - deliberately, because the bimodal
case is one repeat unit at two chain lengths scored through the same universal
WLF constants, so the two errors move together almost exactly. The blend now
inherits 490%, and a blend of identical components reproduces that component's
own bar, which is the behaviour any such rule has to show to be believable.

Correcting it did not change which blends are feasible. It changed how much the
ranking deserves to be believed, which was the question being asked.

## The solution, which is what a web shooter actually holds

Everything above spins a MELT. A web shooter holds a fluid in a cartridge, and
the fluid that becomes a solid in flight is a polymer dissolved in a solvent -
dry spinning, the route cellulose acetate and acrylic fibres have been made by
since the 1920s.

The engine could not score one. Asked for a 20% polystyrene in acetone it
refused eight of nine properties:

```
shear_viscosity          REFUSED: every component must be a polymer for these rules
terminal_relaxation_time REFUSED: the melt viscosity this rests on was not available
...
```

`polymer_blend_melt` was right to refuse - its mixing rules are over polymer
components - but nothing else answered, so the whole spinline chain collapsed
behind a viscosity nobody supplied. `PolymerSolutionExpert` closes it, and
needs no new lookup: the chain dimension already comes from side-group bulk,
the solvent's viscosity from `liquid_transport`, both densities from the
existing experts.

**Intrinsic viscosity by Flory-Fox**, `[eta] = Phi <R^2>^1.5 / M`. This is the
one place a solution viscosity is normally a table - Mark-Houwink `K` and `a`
are quoted per polymer, per solvent and per temperature - and going through the
chain dimension means any repeat unit can be asked. Against intrinsic
viscosities computed from published Mark-Houwink constants at theta: 0.88x to
1.61x, worst polypropylene.

**Two concentration branches joined at coil overlap**, `c[eta] = 1`: Huggins
below, a power law above, continuous by construction - the same join `melt.py`
uses between Rouse and reptation, for the same reason.

The exponent above overlap is **4.3, not the melt's 3.4**, and the difference
is not cosmetic. The melt's 3.4 is the exponent in CHAIN LENGTH at fixed
concentration; here the variable is concentration, which adds entanglements and
shrinks the screening length at once. Using 3.4 put a 20% solution of 200
kg/mol polystyrene at 0.22 Pa s - thinner than a dope can be, since being
drawable is the whole point of one.

### What it says, and it is the most useful thing in this document

```
  wt%   kg/mol    eta Pa.s        Wi       bar  Me kg/mol  spins?
   20      200       1.014    0.0127     0.175       91.1  no
   20     1000       32.27     0.405      5.58       91.1  no
   20     4000       635.7      7.98       110       91.1  YES
   35      200       13.57    0.0892      2.35       49.8  no
   35     1000       431.9      2.84      74.6       49.8  YES
   50      200       76.55     0.321      13.2       33.3  no
   65      200       290.5     0.853      50.2       24.4  YES
```

A conventional dope - 20% polymer at 200 kg/mol - has a Weissenberg number of
**0.013**, three orders below the 0.5 a thread needs to thin instead of bead.
It will not spin, and the reason is in the last column: dilution pushed the
entanglement molar mass from about 18 kg/mol in the melt to 91. Solvent does
not join the network, it dilutes it, and a network that sparse has no
elasticity to stretch.

That is the trade the solution route makes. Dilution buys pumpability and costs
elasticity, and the two have to be bought back with chain length.

**35% polystyrene at 1000 kg/mol in acetone: 432 Pa s, Wi = 2.8, 75 bar.**

That is the first candidate anywhere in this document with real MARGIN on the
binding requirement. Every feasible melt candidate sat at a Weissenberg number
of 0.5 to 0.8 against a hard bound of 0.5 - passing inside its own error bar,
which is why none of them survived a robustness check. This one clears it by
more than five times, and at a pressure well inside the 100-250 bar a
compressed-air cylinder gives.

The caveats are real and are not hidden. 1000 kg/mol polystyrene is a specialty
grade rather than a commodity. Acetone is a marginal solvent for polystyrene -
`polymer_dissolution` is the expert to ask, and this one assumes dissolution
and says so on every prediction it makes. And the solvent still has to leave
in flight, which is an evaporation calculation nothing here has done.

## The critique that was right, and what it cost to answer

A reviewer looked at the blend ranking - poly(acrylic acid), bimodal 50/0.8
kg/mol at 15% short - and rejected it on five grounds. Every one of them was
correct, and none of them was a number this engine got wrong:

* 50 kg/mol is not a high molar mass for a structural fibre
* poly(acrylic acid) is amorphous, not the crystalline oriented structure
  behind multi-GPa fibres
* it DEHYDRATES to an anhydride from about 150 C rather than melting - so a
  200 C melt process is chemistry, not processing
* it is water-soluble, and a web that changes properties between a dry room
  and a wet one is not a structural material
* the 15% short fraction plasticises, which is the wrong direction for final
  strength and creep

The reviewer also said the "zero robust" result was itself a rejection
criterion. It is, and this document had already said so.

### The engine already knew most of it

Asked directly, before anything was added:

```
crystallinity, atactic  : crystalline=False - the chain has a stereocentre
hansen_hydrogen_bonding : 13.26 MPa^0.5     - among the highest of any polymer
theoretical_strength    : 286 MPa           - an order below the 2-3 GPa wanted
```

So the failure was the SPECIFICATION, not the panel. A ranking answers the
question it was asked, and the blend specification never asked whether the
candidate survives its own processing temperature, whether it can crystallise,
or whether it takes up water. Three questions, four hard constraints, and
poly(acrylic acid) fails all four.

### The one thing genuinely missing: the top of the processing window

The registry had a melting point and a glass transition - the BOTTOM of a
processing window - and nothing at all for the top. A window with one end is
not a window, and a search handed one will propose a material whose two ends
are in the wrong order. That is precisely what happened.

`decomposition_temperature` and `crystallisability` are new properties, served
by `ThermalStabilityExpert`. The decomposition onset is group contributions
over the repeat unit, FITTED to eleven measured onsets rather than asserted -
the first version was written from chemical intuition and came out 0.59x to
0.82x against every reference, uniformly and badly low. Intuition had the
ordering right and the magnitudes wrong, which is the usual way for a table
nobody checked. Fitted: 1.10x in sample, 1.37x held out.

Over the group sum sit SMARTS gates for the chemistries that have a named
reaction waiting at a particular temperature - poly(acrylic acid)'s anhydride
at 150 C, PVC's dehydrochlorination at 200, polyacrylonitrile's cyclisation at
250, poly(vinyl alcohol)'s dehydration at 200. These are stated, not predicted,
because a group sum answers "how hot before bonds in general break" and cannot
see that two carboxylic acids four atoms apart on one chain will find each
other.

Writing those gates produced one more bug of the kind this repository keeps
finding. The SMARTS ran on the repeat unit with its attachment points capped as
`[H]`, which is what the parachor and the group schemes do - they sum over
atoms and can subtract the caps again. A SMARTS cannot subtract anything.
`[*]OCCOC(=O)...` capped becomes `[H]OCCOC(=O)...`, whose terminal `[OX2H1]` on
a `[CX4]` is a hydroxyl, so poly(ethylene terephthalate) matched the poly(vinyl
alcohol) dehydration rule and was told it decomposes at 200 C. PET is melt-spun
at 280 every day. The gates now run on the unit with its dummies intact.

### The answer changes

`fiber.yaml` states the four of the reviewer's seven requirements that the
registry can express, and lists the other three in its assumptions rather than
pretending to have covered them.

```
feasible 24 of 546

#1  polyethylene                    [*]CC[*]
    poly(ethylene terephthalate)
    poly(butylene terephthalate)

poly(acrylic acid): ELIMINATED BY FOUR HARD CONSTRAINTS
   decomposition_temperature   423 K against a 523 K floor
   crystallisability           0
   hansen_hydrogen_bonding     13.3 against a 10 MPa^0.5 ceiling
   melting_point               it has none
```

Polyethylene, and that is the right answer: UHMWPE is the multi-GPa fibre the
reviewer's requirement list describes - fully crystalline, zero hydrogen
bonding, the widest processing window of any commodity polymer, and gel-spun
rather than melt-spun for exactly the chain-length reasons this document has
been circling since the beginning. PET and PBT behind it are the other two
melt-spun structural fibres in commercial production.

### Three of the seven still cannot be stated

Chain orientation after drawing, fibre tensile strength and creep have no name
in this registry. `theoretical_strength` exists and is explicitly a flaw-free
ISOTROPIC bound - it gives poly(acrylic acid) 286 MPa - and it is not a fibre
strength. A candidate that passes `fiber.yaml` has cleared four of seven
criteria, and the three missing ones are the ones that decide whether a web
holds a person.

## The final material

Two independent routes through this engine converge on one answer.

`fiber.yaml`, which states the four statable requirements of a load-bearing
web, ranks **polyethylene** first out of 546 candidates. The dope map, which
asks which fluid can actually be fired, finds exactly one cell that clears both
spinnability and a wrist-mountable pressure:

```
polyethylene in xylene, at the geometry the sweep found

  wt%   kg/mol    eta Pa.s        Wi        bar  verdict
    2     6000       78.64     0.446      13.59  will not spin
    5     1000       86.26     0.195      14.91  will not spin
    5     3000       915.5      2.07      158.2  *** BOTH ***
    5     6000        4063      9.19      702.1  spins, too stiff
   10     1000        1713      1.93        296  spins, too stiff
   20     3000   3.638e+05       203  6.287e+04  spins, too stiff
```

**5 wt% polyethylene at 3000 kg/mol in a hydrocarbon solvent. 916 Pa s,
Weissenberg 2.07, 158 bar.**

That is gel-spun ultra-high-molar-mass polyethylene. It is Dyneema, and it is
the strongest fibre in commercial production by specific strength. The engine
was not told about it; it arrived from a fibre-requirements ranking and a
process map that were run separately and did not share a candidate list.

### And the engine says the device cannot make it

Polyethylene's predicted Hansen triple is (17.5, 0.0, 0.0) against xylene's
(17.8, 1.0, 3.1) - a distance of 3.3 MPa^0.5, comfortably compatible. The
engine was about to report that as a room-temperature solvent, and it is not
one. Hansen parameters describe cohesion in an AMORPHOUS phase; they say
nothing about a crystal lattice, and a crystalline polymer below its melting
point is insoluble however close the parameters sit. The solvent has to pay
the heat of fusion first.

`polymer_dissolution` now refuses it:

```
PE in xylene at  25 C: REFUSED
  this chain crystallises and melts at 135 C, so at 25 C it is a solid crystal
  rather than a coil a solvent can reach [...] it dissolves near 130 C, which
  is why ultra-high-molar-mass polyethylene is gel-spun hot

PE in xylene at 140 C: proceeds
```

The gate fires only on a polymer with a MEASURED melting point, below it. A
configurationally regular backbone is not enough - `melt.py` records
polyisobutylene as its known miss and bisphenol-A polycarbonate is the same
class, regular and amorphous in practice - so for everything else the weaker
warning that was already there is the right strength of claim.

### So the answer is

**The material is ultra-high-molar-mass polyethylene, about 3000 kg/mol,
spun as a 5% gel in a hydrocarbon solvent.** It meets every requirement this
engine can state, with margin rather than inside its error bars - the only
candidate in this document that does.

**The device cannot hold it.** The dope only exists above about 130 C, because
that is where the crystals let go. A wrist cartridge of 130 C xylene is not a
web shooter; it is a hazard. The chemistry problem is solved and the
engineering problem is worse than it looked, and that is a more useful place to
end than a ranked list would have been.

What would change it: a polymer that crystallises fast enough to lock
orientation but dissolves cold, which is a genuine contradiction rather than a
gap in this engine - crystallinity is what makes both of those true. The
routes out are a soluble precursor that crystallises after spinning, the way
poly(vinyl alcohol) fibre and carbon fibre are made, or an entirely different
solidification mechanism. Neither is a search this engine can run, because
neither is a property in its registry.

## The wrist device: three questions, asked together

The gel-spun polyethylene answer was the right material and the wrong device -
its dope needs 130 C. Constraining the search to a wearable temperature turned
up a second failure of method, which is worth more than the first answer.

Three questions have to be answered at once, and every earlier run in this
document answered one or two and reported a winner:

1. does it SPIN - strain hardening at or above about a half
2. can it be FIRED - pressure inside what a cartridge on a person supplies
3. does it DISSOLVE - at a temperature a person can wear

The third kept getting skipped. `PolymerSolutionExpert` prints
"THIS ASSUMES THE POLYMER DISSOLVES" on every prediction it makes, for exactly
this reason, and the assumption still went unchecked through several rounds.
A viscosity computed for a polymer that will not dissolve is a number about a
suspension, not a dope.

Two bugs came out of asking properly, both mine:

**The whole pool was in water.** `PolymerSolutionExplorer` nested its four
axes with fifty solvents second and five hundred repeat units fourth, so a
draw of 2500 never reached a second solvent. Water has no carbon, the
Orrick-Erbar viscosity correlation refuses it on exactly those grounds, and
the run reported 2500 candidates whose viscosity could not be predicted -
which reads as a finding about polymers and is a finding about loop order. The
product is now walked diagonally: 34 distinct solvents in the first 60 draws
against 1 before.

**The winner could not be checked.** Poly(4-methylstyrene) outscored
everything on spinnability and pressure and was recommended twice. Asked
whether it dissolves, the engine answers "no fitted interaction radius" -
because this repository established that the Hansen radius is NOT derivable
from structure, over seven tabulated spheres whose R0/delta ratios span 3.45x
at r^2 = 0.11, and built that refusal into `polymer_dissolution`. A solubility
that cannot be computed is a refusal, not a pass. The recommendation walked
straight past a limit this repository had measured and written down itself.

### What survives all three

```bash
python examples/web_shooter/wrist_candidates.py
```

```
    bar     Wi    RED  polymer / solvent (flash) / wt% / kg per mol
   24.7   1.46   0.65  polystyrene in toluene (fp 5 C), 20%, 1500
   30.6   2.00   0.88  PMMA in 2-heptanone (fp 37 C), 20%, 1500
   40.4   2.64   0.80  polystyrene in 2-heptanone (fp 37 C), 20%, 1500
   56.6   4.55   0.65  polystyrene in toluene (fp 5 C), 15%, 4000
   91.5   8.17   0.80  polystyrene in 2-heptanone (fp 37 C), 15%, 4000

8 combinations could not be answered at all: poly(4-methylstyrene)
```

**Polystyrene at 4000 kg/mol, 15% in 2-heptanone. 92 bar, Weissenberg 8.17,
RED 0.80.**

Sixteen times the strain-hardening threshold, against melt candidates that
scraped past at 0.5 to 0.8 and did not survive a robustness check. A solvent
whose flash point is 37 C, so there is no flammable vapour at room temperature
- which acetone at -24 C and 2-butanone at -10 cannot say. And 92 bar is a
small compressed-air cylinder.

For the simplest hardware, the same polymer at 1500 kg/mol and 20% gives
Weissenberg 2.64 at 40 bar, which is a CO2 cartridge.

Polystyrene wins partly for an unglamorous reason: it is one of the seven
polymers with a MEASURED Hansen sphere, so the solubility question can be
asked of it at all. That is a statement about this repository's data rather
than about polystyrene's superiority. Poly(4-methylstyrene) may well be better
and is simply not checkable.

### What it is worth

Amorphous, so it dissolves cold. Glassy at room temperature - polystyrene's
glass transition is 100 C - so it does not cold-flow while a web is loaded,
which is a different regime from the poly(vinyl methyl ether) that led an
earlier ranking with a glass transition of -31 C. Two hundred filaments drawn
to 100 um is 1.57 mm^2, which at 100 to 300 MPa is 35 to 106 lb.

That range is an assumption about draw quality, not a prediction. Chain
orientation, fibre tensile strength and creep still have no name in this
registry, and they are the three that decide whether a web holds a person.

## Verdict

The engine handled the parts it covers and refused the rest legibly rather than
guessing - the infeasibility diagnosis found the solvency/adhesion gap on its
own, and the uncertainty reporting made clear that neither surviving polymer
clears its bounds robustly.

The adhesion gap turned out not to be a data gap at all: the question had been
put to the wrong phase, and asking the polymer instead needed only an existing
table read from the other side. The unreachable polymer class remains a
plumbing gap worth closing. What no amount of data fixes is that reversible
work is not peel strength, so the engine still cannot say whether the web
holds - only which polymer holds better.
