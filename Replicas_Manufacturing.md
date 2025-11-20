
# Replicas_Manufacturing.md — Recipes to create high-fidelity Indus seal replicas

## Goals
Produce three classes of replicas that capture mechanical, optical, and surface characteristics:
A) SLA resin optical/geometry replicas
B) Fired steatite-like ceramic replicas
C) Metal-backed composite replicas

## 1) SLA resin replica (geometry-first)
Materials: SLA resin, isopropanol, UV cure chamber.
Steps:
1. Import 3D scan (.ply/.stl). Scale to exact dimensions if needed.
2. Orient to minimize supports across fine details.
3. Print at highest resolution (25–50 μm).
4. Post-process: wash in IPA, remove supports carefully, cure in UV chamber.
5. Smooth micro-scratches using micro-abrasive polishing with alumina paste to match surface finish.

## 2) Fired steatite-like ceramic replica (acoustic fidelity)
Materials: Talc-rich soapstone or steatite powder, bentonite/clay binder, small kiln.
Steps (adapted from glyptic literature):
1. Prepare a talc-rich paste: 70% finely milled talc/steatite, 20% refractory clay (bentonite), 10% water (adjust to workable consistency).
2. Press-cast or hand-model the SLA master into a silicone mold made from the SLA print.
3. Dry slowly for 48–72 hours to avoid cracking.
4. Fire at 900–1100°C depending on recipe; ramp rates slow (100°C/hour) up to soak for 1–2 hours, cool slowly.
5. Optional light glazing: apply thin alkaline glaze if testing faience-like surface.

Notes: exact composition influences density and elasticity—document material properties (mass, Young’s modulus estimate) for each batch.

## 3) Metal-backed composite (EM/acoustic boundary conditions)
Materials: Thin brass/aluminum sheet, epoxy resin.
Steps:
1. Bond SLA or fired replica to 0.5–1 mm metal backing using epoxy.
2. Ensure even contact and no air gaps (use vacuum bagging if available).
3. Cure fully before testing.

## Quality control
- Microscope inspection for fidelity to carved lines.
- Measure mass and dimensions; compare to original.
- Record kiln run logs, batch numbers, and material lot numbers for reproducibility.
