# Methods: gaze geometry {#sec:geometry}

The geometry layer turns raw image measurements into calibrated quantities in
degrees of visual angle (dva). Pixel offsets from screen centre are converted by
the exact relation

$$ \theta = \arctan\!\left(\frac{s_{\mathrm{cm}}}{d_{\mathrm{cm}}}\right) $$ {#eq:pix2deg}

where $s_{\mathrm{cm}}$ is the offset in centimetres (pixels times the screen's
cm-per-pixel) and $d_{\mathrm{cm}}$ the viewing distance. The inverse of
[@eq:pix2deg], `deg2pix`, is exact, so `pix2deg`∘`deg2pix` round-trips to
floating-point tolerance.

For appearance-based capture, normalised iris displacement within the eye
aperture is mapped to eyeball rotation through a sphere model,

$$ \theta_{\mathrm{gaze}} = \arcsin\!\big(o \cdot \sin\theta_{\max}\big) $$ {#eq:iris}

with $o \in [-1, 1]$ the normalised iris offset and $\theta_{\max}$ the angle at
full deflection; the map ([@eq:iris]) is monotone and odd. Head-distance dependence is removed
by dividing offsets by the inter-ocular distance in pixels and rescaling to a
fixed reference, so the same physical movement yields the same value regardless of
how far the subject sits from the camera.

Two conventions are fixed package-wide to prevent sign errors: gaze direction
uses $0^\circ$ = right and $+90^\circ$ = up (image-$y$, which grows downward, is
negated on the way in); all non-finite inputs raise rather than silently
coercing to zero, so a NaN can never masquerade as a valid centred gaze.
