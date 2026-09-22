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

**Geometric normal** (`Hit.ng`):
The normal of the actual surface — a triangle's face normal (from its winding, `e1 × e2`), or the outward direction from a sphere's centre. It is what the renderer's *geometry* is: the side a ray leaves from, the plane a ray must clear to avoid self-intersection, and the orientation an emitting surface radiates into. Used by `offset_ray_origin` (the `p_error` box is escaped along `ng`), by the entering/exiting decision in `Material::sample`, and by everything on the light side — `area`, `sample`, `pdf_omega`, `light_pdf`, `is_light_itself`. Never interpolated.
_Avoid_: true normal, face normal (that is the triangle-level name; `ng` is the per-hit one).

**Shading normal** (`Hit.ns`):
The normal the BSDF pretends the surface has: the mesh's vertex normals interpolated with the hit's barycentric coordinates (`Hit.bary`), renormalised. Meshes without vertex normals — every parametric shape, and any OBJ without `vn` — set `ns = ng`, so their output is bit-identical to before smooth shading existed. `ns` is always on the same side as `ng` (`face_forward` at the point it is produced, and again after an instance transform, which matters for mirroring transforms). Used by `Material::sample` / `eval` and by the cosine term in NEE. A scene file can force `ns = ng` per shape with `<boolean name="face_normals" value="true"/>`.
_Avoid_: smooth normal, vertex normal (that is the per-vertex input, not the interpolated result).

**Shading-normal breakdown**:
The case where the interpolated `ns` and the geometric `ng` disagree enough that a direction sampled around `ns` points below the real surface (or, for transmission, above it). tinypt discards those samples — `Material::sample` returns `None` and NEE returns zero contribution — because the ray's origin is offset along `ng`, so such a ray would start outside the surface but travel into the mesh's interior. Discarding loses a little energy (it never creates any), and how the loss shows up depends on the BSDF. On a diffuse surface it is small and spread almost uniformly over the mesh — a coarse sphere (144 triangles, adjacent vertex normals 30° apart) loses 0.9% overall in a white furnace, with no concentration at the silhouette — so there is no visible dark rim. On a **transmissive** BSDF it is larger and does concentrate near the silhouette: on the same sphere, refraction discards 2.8% of samples against 1.0% for Lambert, and a thin dark band is visible just inside the outline of a coarsely tessellated glass ball. Rejection rates measured on rough GGX (~15%) are mostly *not* from interpolation: ~12.5% of them are the pre-existing VNDF samples below the horizon that `sample` already discarded before shading normals existed. The alternative of bending the direction back above the surface is not used: it breaks the `BsdfSample` contract (`weight == f·cos/pdf`, `pdf == eval`'s pdf) that MIS depends on.
_Avoid_: normal flipping, shading normal fix-up.

**Texture coordinates** (`Hit.uv`):
The 2D coordinate a texture is looked up with. A mesh interpolates its per-vertex UVs (`vt` in the OBJ) with the hit's barycentric coordinates, sharing one `uv` array per mesh and spending 12 bytes of indices per triangle — the same arrangement as vertex normals, so a mesh without UVs costs nothing. The parametric shapes carry Mitsuba's own parameterisation: `rectangle` is `((x+1)/2, (y+1)/2)`, `cube` gives each face `[0,1]²`, `disk` is `(r, φ/2π)`, and `sphere` is `(φ/2π, θ/π)` with the poles on ±z. A hit with no UV reports `(0, 0)`. UVs are an object-space attribute, so an instance transform does not change them.
_Avoid_: texture space, st coordinates.

**Texture** (`texture::Texture`, `Material::resolve_textures`):
A bitmap sampled bilinearly at `Hit.uv`, held in the scene's `textures` arena and referenced from a material by a `TexId` index — the index keeps `Material` `Copy`, which matters because it is copied at every intersection. Colour textures are decoded from **sRGB** on load, the exact inverse of the PPM encode, while `raw` textures stay linear; out-of-range UVs follow the texture's `Wrap` (`repeat` by default, as in Mitsuba). The integrator evaluates textures exactly once per intersection, folding the result into a texture-free copy of the material, so `sample` and `eval` never see a UV or a texture.

**usemtl group / MTL material** (`obj_loader::load_obj_groups`, `mitsuba::parse_obj_with_mtl`):
An OBJ stays one mesh with one BVH; each triangle carries its own `mat_id`, taken from the `usemtl` in force when its face was read (faces before any `usemtl`, or naming a material the MTL lacks, get a grey diffuse). Only names actually used by a face become materials. `Instance.mat_override` keeps its meaning: a `<bsdf>` child overrides everything and the MTL is never read; without one (and without `use_mtl=false`) the MTL decides. MTL maps to `Lambert` (with the `map_Kd` texture) when the material has `map_Kd`; only when it has none, a bright `Ks` with `Ns` > 1 maps to `Ggx` instead. `map_Kd` wins on purpose — a material with both used to fall into the `Ggx` branch before its diffuse texture was ever read, silently dropping it (this bit Sponza's floor/arch/chain/vase_hanging). A blended diffuse+glossy BSDF would need a new `Material` variant, so this stays a priority rule rather than a combined shading model. Alpha (`d`/`map_d`), `map_Ka` and emission are not yet supported and warn once per material (alpha: once per scene).
_Avoid_: material group as a separate mesh.

**Normal map / height map** (`normal_map::NormalMap`, `HeightMap`, wired in `integrator::perturb_shading_normal`):
A perturbation of the *shading* normal `ns` only — the geometric normal `ng` (ray-origin offsets, front/back tests, light area and pdf) is never touched, and the caller applies `face_forward(ns', ng)`. A tangent-space map decodes `2·rgb − 1` (read linear, never sRGB) in the frame `(t, b, ns)`. A height map is turned into slopes with a central difference exactly one texel wide, expressed **per texel** (the UV-unit gradient divided by the map size — per UV unit a 512px map's edges reach slopes of ~25, so `strength = 1` would always clamp; per texel, `strength` 4–16 gives a clear relief on Sponza's brick map and 1 is subtle), `su = strength·hu·g / |∂p/∂u|`, `sv = strength·hv·g / |∂p/∂v|` with `g = √(|∂p/∂u|·|∂p/∂v|)`, clamped to ±tan 85°. **`strength` is dimensionless on purpose**: scaling a model uniformly by k scales `|∂p/∂u|`, `|∂p/∂v|` and `g` all by k, so the slope — and the look — does not change. PBRT's `dpdu + bm·hu·ns` form gives `bm` a world-length unit, so Sponza (in cm, placed with `to_world` scale 0.01) would look a hundred times bumpier; this renderer guarantees scale-invariant results (3b/3c), so it uses the dimensionless form. **MTL** `map_bump` (with `-bm`, default 1) becomes a height map with `strength = bm · MTL_BUMP_K` and `norm` (an extension) a tangent-space normal map; `norm` wins when both are given. `MTL_BUMP_K = 8` was picked by rendering Sponza at K = 1 / 4 / 8 / 16: 8 gives clear mortar grooves and stone relief, 16 over-shades; Sponza's nine bump maps have per-texel slopes of 0.05–0.09 at p99 (max 0.17), so even K = 16 reaches `SLOPE_MAX` on 0 % of texels. The material → map link is a side table (`Scene::mat_maps`, empty or exactly as long as `mats`; `Material` stays `Copy`), filled only through `mitsuba::push_material`. The perturbation happens once per intersection, right after texture resolution, so NEE and `Material::sample` see the same `ns`. The tangent vectors come from `World::surface_tangents`, which maps them with the forward linear part `A` (a tangent is not a normal: the inverse transpose is wrong under shear).
_Avoid_: bump height in world units.
_Avoid_: material map, shader parameter.

**Ray origin offset** (`Hit.p_error`, `offset_ray_origin`):
How a ray leaving a surface avoids hitting that surface again, independent of scene scale and of distance from the origin (PBRT v4's `OffsetRayOrigin`). Every intersection returns a conservative per-component floating-point error bound `p_error` for its point: triangles from the barycentric reconstruction, spheres after re-projecting onto the sphere, and instance hits by propagating the object-space bound through the transform (including the matrix magnitudes and the residual of the numerical inverse). A new ray starts at `p` pushed out of that error box along the geometric normal, on the side of the ray direction, plus one ulp per component; its `tmin` is 0. Primitives additionally accept only hits whose computed `t` exceeds its own error bound, and instance intersection advances the object-space origin by its transform error (PBRT's ray transform). Shadow rays offset both ends (shading point and light sample point with `LightSample.p_error`) and do not count a hit on the sampled light itself as an occluder. That exclusion is required, not a safety net: for sphere lights the ray–sphere t error can exceed the endpoint's `p_error`, so the light surface can still be hit inside the segment; without the exclusion `sample/default.xml` loses about 6% of its light. Bounding boxes are padded relative to their coordinates and the slab test widens the far distance by 1 + 2γ(3), so flat boxes far from the origin are not missed.
Triangles use PBRT v4's watertight intersection (translate to the ray origin, permute so the dominant ray axis is z, shear, 2D edge functions where exactly-zero edges count as inside), so rays through an edge or vertex shared by adjacent triangles never slip through (the earlier Möller–Trumbore test missed a few percent of rays aimed exactly at shared edges, depending on the mesh). `World::hit` rejects an instance with a conservative world-space bounding box (transformed mesh bounds padded both by the transform's error bound and by γ(3) of the coordinates; either padding alone suffices, and a test fails if both are removed) before transforming the ray into object space.
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
- Normal orientation (the entering/exiting decision): `sample` decides entering/exiting from `hit.ng` and the incoming ray, then orients the **shading** normal to that same side; `eval` expects an **already-oriented** shading normal, which the integrator supplies by orienting `hit.ng` with `oriented_normal` and face-forwarding `hit.ns` onto it (also used for NEE).
- `wo` is the outgoing direction `(-ray.d).norm()`, pointing back toward where the ray came from.
- In scene files, `<rgb>` colour values are **linear** (read straight into `Color`); `<srgb>` values are **sRGB** (gamma-decoded via `from_srgb`). A scene file uses `<srgb>` to reproduce a `from_srgb` albedo and `<rgb>` for scene-referred radiance.
- **Colour encoding is symmetric**: input decodes with the exact piecewise sRGB curve (`srgb_to_linear`) and PPM output (binary P6) encodes with its exact inverse (`linear_to_srgb`) — not a `1/2.2` approximation. HDR stays linear sRGB; EXR is linear ACEScg.
- **Background**: a scene file with no environment emitter defaults to a **black** background (Mitsuba semantics). The built-in default scene (via `build_default_scene`) instead falls back to the procedural `sky()` gradient.

## Example dialogue

> **Dev:** The integrator was computing `cos/π` by hand for Lambert and calling `ggx_pdf` for GGX — two copies of each pdf.
> **Expert:** That's because `scatter` returned throughput, not a pdf. Now the BSDF owns it: `sample` hands back a `BsdfSample` whose `pdf` matches what `eval` reports. One source of truth.
> **Dev:** And the delta materials?
> **Expert:** `is_delta()` is true for Metal and Dielectric, so the integrator skips NEE for them and sets `last_bsdf_pdf` to zero — no special-casing in the bounce loop.
