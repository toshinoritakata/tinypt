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
A surface that contributes light, from the *material* side. Queried via `Material::emitted() -> Option<Color>`; only `DiffuseLight` returns `Some`. Distinct from a **Light**, which is the *geometry* side.
_Avoid_: light material, glow.

**Light**:
A reference to an emitting primitive for sampling, from the *geometry* side: `Light::Sphere { idx }` or `Light::Triangle { mesh_id, tri_id, inst_id }` indexing into the `World`. Owns the per-shape geometry of emission — `area` and `sample_surface` — shared by CDF construction (`build_lights`) and light sampling (`sample_light`). A new emitting shape is one new arm here.
_Avoid_: emitter (reserved for the material side), light source.

### Estimation

**NEE** (Next Event Estimation):
Directly sampling a light (environment map or area light) at each non-delta bounce to estimate direct illumination.

**MIS** (Multiple Importance Sampling):
Combining BSDF sampling and light sampling with the power heuristic (β=2). Needs the BSDF's pdf for a given direction pair — supplied by `eval` and by `BsdfSample.pdf`.

**Throughput** (`weight`):
The accumulated attenuation along a path, `f·cos/pdf` folded together. Carried in `BsdfSample.weight` and multiplied into the path's running throughput.

### Scene description

**Scene file**:
An external description of a `Scene` (shapes, BSDFs, emitters, sensor), loaded as a subset of the Mitsuba renderer's XML format (see [ADR-0002](docs/adr/0002-mitsuba-xml-scene-format.md)). Distinct from the **default scene** built in code by `build_scene`.
_Avoid_: scene graph, scene format.

**Sensor**:
The Mitsuba term for the camera, mapped to `Camera`. A `perspective` sensor carries `fov`, a `to_world` transform (via `lookat`), and optional `aperture_radius`/`focus_distance` for depth of field.
_Avoid_: viewpoint, eye.

**Shape**:
The Mitsuba term for a renderable primitive, mapped onto our geometry. `sphere` becomes a `Sphere`; `obj` and the parametric shapes `rectangle` / `cube` / `disk` become a `Mesh` + `Instance` placed by a `to_world` `Transform`. The parametric shapes are generated in Mitsuba's canonical form (rectangle: XY `[-1,1]²`; cube: `[-1,1]³`; disk: unit disk at z=0). A shape carries a child `bsdf` and optionally a child `area` `emitter`.

**Transform**:
An object→world affine transform (`Transform`): a general linear part (`Mat3`) plus a translation, with the inverse and inverse-transpose precomputed for ray and normal transforms. Composed from Mitsuba `translate` / `rotate` (any axis) / `scale` (non-uniform) / `matrix` operations; handles arbitrary rotation, non-uniform scale, and shear.

## Conventions

- `eval` returns the BSDF value `f` **without** the cosine term; the cosine is folded into `BsdfSample.weight` and applied explicitly by the integrator in NEE.
- Normal orientation (the entering/exiting decision) is handled **inside** the BSDF, not by the integrator.
- `wo` is the outgoing direction `(-ray.d).norm()`, pointing back toward where the ray came from.
- In scene files, `<rgb>` colour values are **linear** (read straight into `Color`); `<srgb>` values are **sRGB** (gamma-decoded via `from_srgb`). A scene file uses `<srgb>` to reproduce a `from_srgb` albedo and `<rgb>` for scene-referred radiance.
- **Colour encoding is symmetric**: input decodes with the exact piecewise sRGB curve (`srgb_to_linear`) and PPM output encodes with its exact inverse (`linear_to_srgb`) — not a `1/2.2` approximation. HDR/EXR stay linear.
- **Background**: a scene file with no environment emitter defaults to a **black** background (Mitsuba semantics). The built-in default scene (via `build_scene`) instead falls back to the procedural `sky()` gradient.

## Example dialogue

> **Dev:** The integrator was computing `cos/π` by hand for Lambert and calling `ggx_pdf` for GGX — two copies of each pdf.
> **Expert:** That's because `scatter` returned throughput, not a pdf. Now the BSDF owns it: `sample` hands back a `BsdfSample` whose `pdf` matches what `eval` reports. One source of truth.
> **Dev:** And the delta materials?
> **Expert:** `is_delta()` is true for Metal and Dielectric, so the integrator skips NEE for them and sets `last_bsdf_pdf` to zero — no special-casing in the bounce loop.
