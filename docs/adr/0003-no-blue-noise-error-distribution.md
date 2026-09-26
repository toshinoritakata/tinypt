# Do not distribute the sampling error as blue noise

We will keep the per-pixel sampler seeded by a hash of the pixel
coordinate, rather than by a blue-noise mask, and accept that the
residual error is white in screen space.

Blue-noise error distribution does not reduce variance — it moves the
same error into the high frequencies, where the eye is less sensitive
and, more usefully here, where a denoiser removes it more cleanly. With
Intel OIDN downstream and 256spp already close to converged on our
scenes, that seemed worth having.

Four ways of applying a 64×64 void-and-cluster mask were measured on
`cornell`, `default`, `sponza` and the spiral scene, against a converged
reference, with the denoised error as the primary metric and the radial
power spectrum of the raw error as the evidence:

- seeding the pixel's Owen scramble with the mask rank: no change, since
  hashing the rank leaves neighbouring pixels uncorrelated again;
- a Cranley–Patterson rotation by the mask: better at 1spp, but 40%
  worse raw MSE from 4spp, the error becoming a sawtooth of period 1/N;
- a shared stratum assignment with the mask placing the sample inside
  its stratum: the low frequencies grew three- to sevenfold, because
  every pixel then shares the same QMC structure;
- a per-pixel stratum assignment with the mask inside the stratum: 1spp
  improves, 4 to 256spp are unchanged within noise.

The reason all four fail above one sample is the same. Once a pixel
takes several samples, its error is dominated by *which* strata the
scrambling assigns it, and that assignment is white. The mask can only
move the position *within* a stratum, whose share of the error shrinks
as the count grows. Reading a mask is therefore not enough: the seeds
themselves have to be chosen so that the resulting error is blue, which
is an optimisation over the image (Heitz & Belcour 2019), not a lookup.

Only 1spp benefited — denoised MSE fell to 0.77 on `default` and 0.91 on
the spiral scene, with the low-frequency bands down to 0.73 of their
former energy, so the mechanism does work there. Taking that would mean
one sample per pixel behaving differently from two, a special case in
the sampler in exchange for better previews alone. We would rather keep
the sampler uniform.

This is recorded so the cheap version is not attempted again. Revisit
only with seed optimisation, and only if preview quality at low sample
counts becomes a goal in itself.
