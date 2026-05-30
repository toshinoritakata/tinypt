# Use a Mitsuba XML subset as the scene file format

We will support loading scenes from a subset of the Mitsuba renderer's XML
format, rather than OpenUSD, glTF, PBRT, or a custom serde format. The
priority was semantic alignment with a physically-based Monte Carlo path
tracer: Mitsuba's `roughconductor` (distribution `ggx`, parameter `alpha`)
maps almost 1:1 to our `Ggx` material, and the format natively has the
sphere primitive, DOF camera (`apertureRadius`/`focusDistance`), area
emitters, and environment maps that this renderer already supports.
Crucially, Mitsuba XML is a declarative tree that parses trivially in Rust
(`quick-xml`) and maps directly onto our `Scene`, with none of the stateful
graphics-state stack a PBRT parser would require.

## Considered Options

- **OpenUSD**: industry standard for film pipelines, but overkill here —
  the reference implementation is a large C++ library with immature Rust
  support, and its value (composition, scale, DCC interop) is wasted on a
  tiny renderer. Rejected unless film-pipeline interop becomes a goal.
- **glTF 2.0**: best Rust crate support and DCC export, but loses the sphere
  primitive and DOF camera, and maps our perfect-mirror `Metal` and
  absorbing `Dielectric` only via KHR extensions. Better for asset interop
  than for path-tracer fidelity.
- **PBRT**: equally aligned semantically, but the format is a token stream
  with a stateful graphics-state/CTM stack — a heavier parser to hand-write
  with no mature standalone Rust crate.
- **Custom serde (RON/JSON/TOML)**: minimal effort, zero impedance mismatch,
  but no standard interchange and no path to reading externally authored
  scenes.

## Consequences

We will implement a *subset reader*, not full Mitsuba compatibility. Known
mapping frictions to resolve as a subset (or via custom attributes): our
`Dielectric` bakes in Beer-Lambert absorption where Mitsuba uses a
participating medium; our conductors use `albedo` as Schlick F0 where
Mitsuba defaults to complex IOR (read `reflectance`/`specular_reflectance`
instead); our `Subsurface` is a simplified hack, not Mitsuba's BSSRDF;
spectral inputs are read as RGB triples.

A loaded scene with no environment emitter defaults to a **black**
background (Mitsuba semantics), deliberately *not* the procedural `sky()`
gradient the built-in `build_scene` falls back to — otherwise the sky would
leak in as ambient light through any open/interior scene (e.g. the Cornell
box). The parametric Mitsuba shapes (`rectangle`, `cube`, `disk`) are
generated as canonical-form triangle meshes placed by a `to_world`
transform, reusing the mesh+instance path.
