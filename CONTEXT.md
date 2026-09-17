# tinypt

A Monte Carlo path tracer. This file pins the vocabulary the renderer's modules share, so that names stay consistent across the integrator, materials, geometry, and sampling code.

## Language

### Shading

**BSDF**:
The scattering behaviour at a surface point — how an incoming direction relates to an outgoing one. In this codebase every `Material` variant *is* a BSDF: it can `sample` a scattered direction, `eval` its value and pdf for a given pair of directions, and report whether it is `is_delta`.
_Avoid_: BRDF (too narrow — we include transmission), shader, surface model.

**BsdfSample**:
The result of sampling a BSDF: the scattered `Ray` (origin included, so transmissive offsets stay inside the BSDF), the throughput `weight` (`f·cos/pdf`), the `pdf`, the `is_delta` flag, and `eta` — the relative index of refraction η_t/η_i of a transmission (1 for reflection), which the integrator multiplies along the path so Russian roulette can use `max(throughput)·η²` as in Mitsuba 3. The `pdf` it reports is the same value `eval` would return for that direction pair.
_Avoid_: ScatterResult, BounceResult.

**Delta BSDF**:
A BSDF whose scattering is a Dirac distribution — perfect mirror (Metal) or refraction (Dielectric). Has no finite pdf, so it is excluded from Next Event Estimation. Reported by `is_delta()`.
_Avoid_: specular (ambiguous — GGX is "specular" but not delta), singular.

**Emitter / emitted radiance**:
A surface that contributes light, from the *material* side. Queried via `Material::emitted() -> Option<Color>`; only `DiffuseLight` returns `Some`. Distinct from a **Light**, which is the *geometry* side.
_Avoid_: light material, glow.

**Light**:
A reference to an emitting primitive for sampling, from the *geometry* side: `Light::Sphere { idx }` or `Light::Triangle { mesh_id, tri_id, inst_id }` indexing into the `World`. Owns the per-shape geometry of emission — `area` (used for the CDF weights in `build_lights`), `sample` (a point on the light as seen from a reference point) and `pdf_omega` (the solid-angle pdf of that sampling) — so `sample_light` and `light_pdf` share one pdf and MIS weights always agree. Spheres are sampled uniformly within the cone they subtend from an outside reference point (exact in f64, no small-cone approximation), falling back to uniform area sampling from inside or from within rounding distance of the surface (sin²θmax > 1 − 1e-12); triangles use uniform area sampling. A new emitting shape is one new arm here.
_Avoid_: emitter (reserved for the material side), light source.

### Estimation

**NEE** (Next Event Estimation):
Directly sampling a light (environment map or area light) at each non-delta bounce to estimate direct illumination.

**MIS** (Multiple Importance Sampling):
Combining BSDF sampling and light sampling with the power heuristic (β=2). Needs the BSDF's pdf for a given direction pair — supplied by `eval` and by `BsdfSample.pdf`.

**Path depth** (`max_depth`, `rr_depth`):
The length of a path counted in vertices from the camera, with Mitsuba's meaning: depth 1 is an emitter or the background seen directly, depth 2 is direct illumination (one scattering vertex), and so on. `RenderConfig.max_depth` limits it (`usize::MAX` = unlimited, Mitsuba `max_depth = -1`); an intersection at loop index `bounce` is depth `bounce + 1`, and NEE / BSDF sampling from it (depth `bounce + 2`) only happen while that is within the limit, so the last depth always has both MIS strategies. `rr_depth` is the depth from which Russian roulette decides whether to extend a path. Scene files set both through `<integrator>`; the built-in scene and CLI use the defaults `MAX_DEPTH = 9` (at most 8 scattering events) and `RR_DEPTH = 4`, and there is no CLI flag for them.
_Avoid_: bounces (ambiguous about whether the camera ray or the light hit counts).

**Throughput** (`weight`):
The accumulated attenuation along a path, `f·cos/pdf` folded together. Carried in `BsdfSample.weight` and multiplied into the path's running throughput.

**Russian roulette**:
Probabilistic path termination from depth `rr_depth` on; a surviving path's throughput is divided by the survival probability, so the estimate stays unbiased. The survival probability is `max(throughput)·η²` clamped to [0.05, 0.95], where η is the product of `BsdfSample.eta` along the path — the same η² compensation as Mitsuba 3, so paths inside glass are not killed just because transmission scaled their radiance by 1/η². Unlike Mitsuba 3, which decides after multiplying in the current vertex's BSDF weight, tinypt decides **before** BSDF sampling (after emission and NEE at the vertex), using the throughput without that weight: deciding after the weight makes the survival probability ≈ albedo right after a diffuse bounce, which with a shallow `rr_depth` costs far more variance than the time it saves (measured: 2.7× variance×time in diffuse regions of `sample/default.xml` at `rr_depth = 1`), while the glass-region benefit of the η² compensation is the same either way. Russian roulette is unbiased on its own, but it interacts with the firefly clamp: a surviving path's contributions are scaled by 1/p, so they reach the clamp more often, and the clamp's (darkening) bias grows the more aggressively roulette terminates paths (shallow `rr_depth`, low throughput); the η² compensation reduces this inside glass.

**Ray origin offset** (`Hit.p_error`, `offset_ray_origin`):
How a ray leaving a surface avoids hitting that surface again, independent of scene scale and of distance from the origin (PBRT v4's `OffsetRayOrigin`). Every intersection returns a conservative per-component floating-point error bound `p_error` for its point: triangles from the barycentric reconstruction, spheres after re-projecting onto the sphere, and instance hits by propagating the object-space bound through the transform (including the matrix magnitudes and the residual of the numerical inverse). A new ray starts at `p` pushed out of that error box along the geometric normal, on the side of the ray direction, plus one ulp per component; its `tmin` is 0. Primitives additionally accept only hits whose computed `t` exceeds its own error bound, and instance intersection advances the object-space origin by its transform error (PBRT's ray transform). Shadow rays offset both ends (shading point and light sample point with `LightSample.p_error`) and do not count a hit on the sampled light itself as an occluder. That exclusion is required, not a safety net: for sphere lights the ray–sphere t error can exceed the endpoint's `p_error`, so the light surface can still be hit inside the segment; without the exclusion `sample/default.xml` loses about 6% of its light. Bounding boxes are padded relative to their coordinates and the slab test widens the far distance by 1 + 2γ(3), so flat boxes far from the origin are not missed.
Known limit: the Möller–Trumbore triangle test is not watertight — even at unit scale about 2% of rays aimed exactly at an edge shared by two triangles miss both; a watertight test (PBRT v4) would fix this.
_Avoid_: a fixed ray epsilon (absolute or scene-relative).

**Firefly clamp**:
Every contribution added to a path's radiance — any emitter/background hit (MIS-weighted or not, e.g. seen directly from the camera or after a delta bounce) **and** every NEE contribution — is luminance-scaled to at most `FIREFLY_CLAMP` (50) before being accumulated. Biased by design; applied per contribution (not per path), so both MIS strategies share the same limit.

### Scene description

**Scene file**:
An external description of a `Scene` (shapes, BSDFs, emitters, sensor), loaded as a subset of the Mitsuba renderer's XML format (see [ADR-0002](docs/adr/0002-mitsuba-xml-scene-format.md)). Distinct from the **default scene** built in code by `build_default_scene`.
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
- Normal orientation (the entering/exiting decision): `sample` orients the normal internally from `hit.n` and the incoming ray; `eval` expects an **already-oriented** `n`, which the integrator supplies via `oriented_normal` (also used for NEE).
- `wo` is the outgoing direction `(-ray.d).norm()`, pointing back toward where the ray came from.
- In scene files, `<rgb>` colour values are **linear** (read straight into `Color`); `<srgb>` values are **sRGB** (gamma-decoded via `from_srgb`). A scene file uses `<srgb>` to reproduce a `from_srgb` albedo and `<rgb>` for scene-referred radiance.
- **Colour encoding is symmetric**: input decodes with the exact piecewise sRGB curve (`srgb_to_linear`) and PPM output encodes with its exact inverse (`linear_to_srgb`) — not a `1/2.2` approximation. HDR stays linear sRGB; EXR is linear ACEScg.
- **Background**: a scene file with no environment emitter defaults to a **black** background (Mitsuba semantics). The built-in default scene (via `build_default_scene`) instead falls back to the procedural `sky()` gradient.

## Example dialogue

> **Dev:** The integrator was computing `cos/π` by hand for Lambert and calling `ggx_pdf` for GGX — two copies of each pdf.
> **Expert:** That's because `scatter` returned throughput, not a pdf. Now the BSDF owns it: `sample` hands back a `BsdfSample` whose `pdf` matches what `eval` reports. One source of truth.
> **Dev:** And the delta materials?
> **Expert:** `is_delta()` is true for Metal and Dielectric, so the integrator skips NEE for them and sets `last_bsdf_pdf` to zero — no special-casing in the bounce loop.
