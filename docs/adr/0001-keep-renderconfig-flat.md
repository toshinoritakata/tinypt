# Keep RenderConfig a flat struct

`RenderConfig` is a flat 17-field struct threaded into `render`, `build_scene`,
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
