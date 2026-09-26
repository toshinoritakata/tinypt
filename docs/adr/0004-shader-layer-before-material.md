# Compute material parameters in a shader layer in front of `Material`

We will keep `Material` (the BSDF enum) exactly as it is — a `Copy` value
with concrete numbers, the only thing `sample` / `eval` / NEE / MIS know —
and put a small **shader layer** in front of it: a `Shader` is a base
`Material`, an optional albedo *expression* and an optional normal map,
evaluated once per intersection into a concrete `Material`. Expressions
live in a flat arena (`Const`, `Texture`, `Noise`, `Mul`; children by
index) on the scene next to the textures, noises and normal maps.

## Why

"One map per material" had been hit three times: normal and bump maps
cannot be stacked, albedo and roughness cannot both be modulated, noise
and an image cannot be combined. The cause was structural — a texture
lived inside a BSDF variant (`Lambert::albedo_tex`), a normal map in a
side table indexed by `mat_id`, and procedural noise in the top bit of a
`TexId` (a stopgap we called "provisional"). Every new kind of input
added a new place. The layer gives them one place and one evaluation
step, and the integrator below it does not change (13 reference scenes
stayed byte-identical; the multiply is the same `hadamard` as before).

## Considered and rejected

- **More `Material` variants** (a textured Lambert, a noisy Lambert, a
  textured-and-noisy one, …). The number of variants multiplies with
  every combination, `sample` / `eval` / `emitted` / `is_delta` and all
  their tests grow with it, and `Material` would still have to carry
  indices into arenas. It is the shape we were escaping.
- **Make the BSDF itself dynamic** (a trait object or a graph that
  evaluates inside `sample` / `eval`). Then the expression is evaluated
  every time the BSDF is sampled or evaluated — NEE, BSDF sampling and
  MIS each look at the material — instead of once, `Material` stops being
  `Copy`, and the delta / emission checks the integrator relies on become
  virtual. The white-furnace and contract tests that pin the BSDFs would
  have to be rebuilt around it.
- **Keep the side tables and add one more** for each new input. This is
  what produced the top-bit flag.

## Consequences

- Constant-only materials must not get slower: a shader without an
  expression returns its base `Material` unchanged (measured equal on
  sponza, spiral, cornell and rungholt).
- The first step only unifies placement. Still one normal map per
  material, expressions only for albedo, and no `Mix` / `Add` nodes; they
  are added when something needs them, not ahead of time.
- The scene shrinks from five material-related fields to one
  (`Scene::shaders`).
