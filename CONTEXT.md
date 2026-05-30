# tinypt

A Monte Carlo path tracer. This file pins the vocabulary the renderer's modules share, so that names stay consistent across the integrator, materials, geometry, and sampling code.

## Language

### Shading

**BSDF**:
The scattering behaviour at a surface point — how an incoming direction relates to an outgoing one. In this codebase every `Material` variant *is* a BSDF: it can `sample` a scattered direction, `eval` its value and pdf for a given pair of directions, and report whether it is `is_delta`.
_Avoid_: BRDF (too narrow — we include transmission), shader, surface model.

**BsdfSample**:
The result of sampling a BSDF: the scattered `Ray` (origin included, so transmissive and subsurface offsets stay inside the BSDF), the throughput `weight` (`f·cos/pdf`), the `pdf`, and the `is_delta` flag. The `pdf` it reports is the same value `eval` would return for that direction pair.
_Avoid_: ScatterResult, BounceResult.

**Delta BSDF**:
A BSDF whose scattering is a Dirac distribution — perfect mirror (Metal) or refraction (Dielectric). Has no finite pdf, so it is excluded from Next Event Estimation. Reported by `is_delta()`.
_Avoid_: specular (ambiguous — GGX is "specular" but not delta), singular.

**Emitter / emitted radiance**:
A surface that contributes light. Queried via `Material::emitted() -> Option<Color>`; only `DiffuseLight` returns `Some`.
_Avoid_: light material, glow.

### Estimation

**NEE** (Next Event Estimation):
Directly sampling a light (environment map or area light) at each non-delta bounce to estimate direct illumination.

**MIS** (Multiple Importance Sampling):
Combining BSDF sampling and light sampling with the power heuristic (β=2). Needs the BSDF's pdf for a given direction pair — supplied by `eval` and by `BsdfSample.pdf`.

**Throughput** (`weight`):
The accumulated attenuation along a path, `f·cos/pdf` folded together. Carried in `BsdfSample.weight` and multiplied into the path's running throughput.

## Conventions

- `eval` returns the BSDF value `f` **without** the cosine term; the cosine is folded into `BsdfSample.weight` and applied explicitly by the integrator in NEE.
- Normal orientation (the entering/exiting decision) is handled **inside** the BSDF, not by the integrator.
- `wo` is the outgoing direction `(-ray.d).norm()`, pointing back toward where the ray came from.

## Example dialogue

> **Dev:** The integrator was computing `cos/π` by hand for Lambert and calling `ggx_pdf` for GGX — two copies of each pdf.
> **Expert:** That's because `scatter` returned throughput, not a pdf. Now the BSDF owns it: `sample` hands back a `BsdfSample` whose `pdf` matches what `eval` reports. One source of truth.
> **Dev:** And the delta materials?
> **Expert:** `is_delta()` is true for Metal and Dielectric, so the integrator skips NEE for them and sets `last_bsdf_pdf` to zero — no special-casing in the bounce loop.
