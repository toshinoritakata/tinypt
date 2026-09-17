# Keep RenderConfig a flat struct

`RenderConfig` is a flat 17-field struct threaded into `render`, `build_scene` (since renamed `build_default_scene`),
and the output path. Splitting it into cohesive bundles (Frame / Sampling /
Checkpoint / Output) was considered and rejected: `render` reads 12 of the 17
fields, so its interface barely narrows, and the bundles would be pure data
groupings with no behaviour — by the deletion test they concentrate no
complexity. The one genuinely separable group (output post-processing) is
already captured by `OutputSettings { exposure, tonemap }` from the
`OutputFormat` work, so the flat bag stays as-is.

## Considered Options

- **Nested composition** (`RenderConfig { frame, sampling, checkpoint, output }`):
  improves cohesion but `render` still needs three of the four bundles; churn
  without leverage.
- **Targeted `OutputConfig` extraction only**: the sole real narrowing, but the
  output knobs are already grouped in `OutputSettings`.
- **Keep flat (chosen)**: lowest churn; no testability or navigability loss.

If `render` is later decomposed so a sub-function reads only a coherent subset
of fields, revisit this — the bundle would then earn its keep.

**Update:** `scene_path`, `max_bounces`, and `rr_start` have since been added
(now 20 fields). They were added flat, consistent with this decision; the
struct grew but was not split.

**Update:** `scene_hash` is no longer a meaningful constant default. When
checkpointing is enabled, `main` overwrites it with a hash derived from the
final config (after scene-file settings and CLI overrides) and the scene
contents; the value in `RenderConfig::default()` is only a placeholder. The
field count is unchanged (20).

**Update:** `max_bounces` and `rr_start` were renamed `max_depth` and
`rr_depth` and now carry Mitsuba's path-length meaning (depth 1 = directly
visible emitters, 2 = direct illumination, `usize::MAX` = unlimited). This is a
rename with a semantic fix, not a new field; the struct stays flat (20 fields).
