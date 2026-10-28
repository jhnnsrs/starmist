"""Starmist: StarDist nucleus segmentation for Arkitekt

Two 2D actions over the pretrained StarDist models, one for fluorescence and one
for H&E brightfield. Each takes a spec-typed image lens and returns a label image
that derives from it, so the result lands in the same coordinate space as the
input.

The app also brings a 3D view of itself: ``models/star_mist.glb`` is registered as
the asset ``star`` and shown by the scene blok ``starmist-3d``. The model carries one
clip, ``spiral``, in which a star flies around a colony of cells and each cell turns
into a coloured instance as the star passes it. The blok plays the clip as far as the
latest task of either action has come (``self.<action>.latest``), so the app keeps no
state for it.

Its logo shows the same star over a colony of nuclei, each nucleus it has passed drawn
the way StarDist predicts it: a polygon through the ends of rays from its centre. Each
action's logo draws one such nucleus in the viewer's palette.

Run:  arkitekt run dev
"""

import functools
import math
from typing import TYPE_CHECKING, Annotated
from xml.sax.saxutils import quoteattr

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from arkitekt import (
    App,
    DescriptorOperator,
    Description,
    Provides,
    Requires,
    Task,
    bsx,
    run,
)
from mikro import Mikro, mikro_service
from mikro.api.schema import (
    ArrayLens,
    AxisInput,
    AxisType,
    DerivedFromInput,
    ValueRelation,
)
from mikro.arkitekt.specs import N_CHANNEL_AXES, N_CHANNELS, N_SPACE_AXES, N_TIMEPOINTS, VALUE_KIND

if TYPE_CHECKING:
    from stardist.models import StarDist2D


app = App(
    "starmist",
    "0.1.0",
    description="StarDist nucleus segmentation with the pretrained 2D models",
    author="jhnnsrs",
)
app.service(mikro_service)
app.register_asset("star", "models/star_mist.glb")

COLLECTIONS = ["segmentation", "prediction", "nuclei"]


# --- logos ------------------------------------------------------------------------


def _ray_points(cx: float, cy: float, radii: list[float]) -> list[tuple[float, float]]:
    """Where rays at equal angles from (cx, cy), the first pointing up, end."""
    step = 2 * math.pi / len(radii)
    return [(cx + r * math.sin(k * step), cy - r * math.cos(k * step)) for k, r in enumerate(radii)]


def _star_convex(cx: float, cy: float, radii: list[float]) -> tuple[str, str]:
    """A StarDist object, as path data: the polygon through the ends of its rays, and the rays."""
    points = _ray_points(cx, cy, radii)
    outline = "M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in points) + "Z"
    rays = "".join(f"M{cx} {cy}L{x:.1f} {y:.1f}" for x, y in points)
    return outline, rays


def _star(cx: float, cy: float, outer: float, inner: float) -> str:
    """A five-pointed star, point up, as path data."""
    return _star_convex(cx, cy, [outer, inner] * 5)[0]


def _segmented(cx: float, cy: float, radii: list[float], colour: str, edge: str) -> str:
    """A nucleus StarDist has found: its instance colour, the polygon and its rays."""
    outline, rays = _star_convex(cx, cy, radii)
    return (
        f'<Path d="{outline}" fill="{colour}" fillOpacity="0.85" stroke="{edge}" strokeWidth="1.4" strokeLinejoin="round" />'
        f'<Path d="{rays}" stroke="surface" strokeOpacity="0.45" strokeWidth="0.7" />'
        f'<Circle cx="{cx}" cy="{cy}" r="1.1" fill="surface" fillOpacity="0.8" />'
    )


def _sparkle(cx: float, cy: float, size: float, fill: str, extra: str = "") -> str:
    """A four-pointed glint, as the star's trail leaves them."""
    s, w = size, size * 0.3
    return (
        f'<Path d="M{cx} {cy - s}L{cx + w} {cy - w}L{cx + s} {cy}L{cx + w} {cy + w}L{cx} {cy + s}'
        f'L{cx - w} {cy + w}L{cx - s} {cy}L{cx - w} {cy - w}Z" fill="{fill}" {extra} />'
    )


# The app's logo, in colours of its own: the star from the 3D view flies over a colony
# of nuclei, and every nucleus it has passed is segmented (an instance colour, and the
# star-convex polygon StarDist predicts, rays and all). The last one is still raw.
app.blok(
    "starmist_logo",
    f"""
    <AgentLogo viewBox="0 0 64 64" title="Starmist">
      <Defs>
        <LinearGradient id="ground" x1="0" y1="0" x2="1" y2="1">
          <Stop offset="0" stopColor="ink" />
          <Stop offset="1" stopColor="spectrum-violet" />
        </LinearGradient>
        <RadialGradient id="glow" cx="0.5" cy="0.5" r="0.5">
          <Stop offset="0" stopColor="spectrum-yellow" stopOpacity="0.45" />
          <Stop offset="1" stopColor="spectrum-yellow" stopOpacity="0" />
        </RadialGradient>
        <ClipPath id="tile"><Rect x="0" y="0" width="64" height="64" rx="14" /></ClipPath>
      </Defs>
      <G clipPath="url(#tile)">
        <Rect x="0" y="0" width="64" height="64" fill="url(#ground)" />
        {_segmented(14, 46, [8, 7.2, 8.4, 7.6, 8.8, 7.4, 8, 7], "spectrum-red", "surface")}
        {_segmented(31, 52, [6.8, 7.6, 7, 6.4, 7.2, 7.8, 6.6, 7.2], "spectrum-green", "surface")}
        {_segmented(21, 29, [6.2, 5.6, 6.6, 6, 5.8, 6.4, 5.6, 6.2], "spectrum-cyan", "surface")}
        <Ellipse cx="49" cy="47" rx="8.5" ry="7.2" fill="muted" fillOpacity="0.3" stroke="muted" strokeOpacity="0.5" strokeWidth="1" strokeDasharray="2 2" />
        <Circle cx="48" cy="46" r="2.6" fill="muted" fillOpacity="0.35" />
        {_sparkle(30.5, 20.5, 2.2, "spectrum-yellow", 'animate="blink"')}
        {_sparkle(25, 13, 1.6, "spectrum-yellow", 'animate="blink" animateDelay="0.5"')}
        {_sparkle(36, 30, 1.4, "spectrum-yellow", 'animate="blink" animateDelay="1"')}
        <G animate="float">
          <Circle cx="45" cy="19" r="15" fill="url(#glow)" />
          <Path d="{_star(45, 19, 12, 5.6)}" fill="spectrum-yellow" stroke="spectrum-orange" strokeWidth="1.6" strokeLinejoin="round" />
          <Circle cx="42" cy="19.5" r="1.3" fill="ink" />
          <Path d="M46.6 19.6q1.4-1.4 2.8 0" fill="none" stroke="ink" strokeWidth="1.2" strokeLinecap="round" />
          <Path d="M42.6 22.6q2.4 2 4.8 0" fill="none" stroke="ink" strokeWidth="1.2" strokeLinecap="round" />
          <Circle cx="40.2" cy="22" r="1.1" fill="spectrum-red" fillOpacity="0.6" />
          <Circle cx="49.8" cy="22" r="1.1" fill="spectrum-red" fillOpacity="0.6" />
        </G>
      </G>
      <Rect x="0.5" y="0.5" width="63" height="63" rx="13.5" fill="none" stroke="spectrum-blue" strokeOpacity="0.3" strokeWidth="1" />
    </AgentLogo>
    """,
    description="A star flies over a colony of nuclei and segments each one it passes.",
    default=True,
)

# The actions' logos, in the viewer's palette. Each draws a nucleus the way StarDist
# sees it, a polygon through the ends of rays from its centre (the rays draw in while
# the task runs), and the app's star beside it. Strokes are ink and round unless a
# shape says otherwise; fills are off, since SVG's own default fill is black.
_PEN = 'fill="none" stroke="ink" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"'
_MASCOT = f'<Path d="{_star(19.5, 4.5, 3.6, 1.6)}" fill="brand" stroke="none" animate="pulse" />'


def _nucleus(cx: float, cy: float, radii: list[float]) -> str:
    """The found nucleus: its rays drawing in, and the polygon they end on."""
    outline, rays = _star_convex(cx, cy, radii)
    return (
        f'<Path d="{rays}" stroke="brand" strokeWidth="1" animate="draw" />'
        f'<Path d="{outline}" stroke="brand-strong" />'
        f'<Circle cx="{cx}" cy="{cy}" r="1" fill="brand-strong" stroke="none" />'
    )


def _action_logo(name: str, title: str, body: str) -> str:
    """Declare the action logo ``name`` that draws ``body`` on a 24 px grid; returns the name."""
    app.blok(
        name,
        f'<ActionLogo viewBox="0 0 24 24" title={quoteattr(title)}><G {_PEN}>{body}{_MASCOT}</G></ActionLogo>',
        description=f"{title}.",
    )
    return name


# Fluorescence: one nucleus glowing in the dark, inside a marching halo of emission.
FLUO_LOGO = _action_logo(
    "starmist_predict_flou2_logo",
    "Segment fluorescent nuclei",
    '<Circle cx="11" cy="13.5" r="9" stroke="muted" strokeWidth="1.2" strokeDasharray="2 2" animate="march" />'
    '<Ellipse cx="11" cy="13.5" rx="6.8" ry="6.2" fill="brand-soft" stroke="none" />'
    + _nucleus(11, 13.5, [6.4, 5.8, 6.8, 6.2, 6.6, 5.6, 6.6, 6]),
)
# H&E: a patch of stained tissue scattered with nuclei, one of them found.
HE_LOGO = _action_logo(
    "starmist_predict_stardist_he_logo",
    "Segment H&E nuclei",
    '<Path d="M3.5 9.5c2-3.5 6-5 10-4.5s6.5 3 6.5 7-1.5 7.5-5.5 8.5-9 .5-11-2.5-2-5 0-8.5z" fill="brand-soft" />'
    '<Circle cx="6.5" cy="16.5" r="1.4" fill="muted" stroke="none" />'
    '<Circle cx="16.5" cy="17" r="1.6" fill="muted" stroke="none" />'
    '<Circle cx="7" cy="10.5" r="1.1" fill="muted" stroke="none" />'
    + _nucleus(12.5, 12.5, [4, 3.6, 4.2, 3.8, 4, 3.4, 4.2, 3.7]),
)

# Written out as plain assignments of one Annotated each: a spec alias nested inside
# another Annotated is not flattened, and a PEP 695 ``type`` statement drops the
# markers. Each argument check is a Requires (the UI offers only lenses that fit, and
# the run refuses one that does not), each return check a Provides.
StillFluorescenceImage = Annotated[
    ArrayLens,
    Requires(key=N_SPACE_AXES, operator=DescriptorOperator.EQUALS, value=2),
    Requires(key=N_CHANNELS, operator=DescriptorOperator.LTE, value=1),
    Requires(key=N_TIMEPOINTS, operator=DescriptorOperator.LTE, value=1),
]
"""A single-channel plane with at most one timepoint."""

StillRGBImage = Annotated[
    ArrayLens,
    Requires(key=N_SPACE_AXES, operator=DescriptorOperator.EQUALS, value=2),
    Requires(key=N_CHANNEL_AXES, operator=DescriptorOperator.EQUALS, value=1),
    Requires(key=N_CHANNELS, operator=DescriptorOperator.EQUALS, value=3),
    Requires(key=N_TIMEPOINTS, operator=DescriptorOperator.LTE, value=1),
]
"""An RGB plane with at most one timepoint."""

LabelImage = Annotated[
    ArrayLens,
    Provides(key=N_SPACE_AXES, operator=DescriptorOperator.EQUALS, value=2),
    Provides(key=VALUE_KIND, operator=DescriptorOperator.EQUALS, value="categorical"),
]
"""A plane of object ids: what both actions produce. The value kind is provenance,
so the run tests only the rank; the derivation edge says the values are labels."""


@functools.lru_cache(maxsize=None)
def _model(name: str) -> "StarDist2D":
    """Load a pretrained StarDist2D model once per process.

    stardist (and with it TensorFlow) is imported here rather than at module
    level, so declaring or inspecting the app does not need either.
    """
    from stardist.models import StarDist2D

    return StarDist2D.from_pretrained(name)


def _segment(
    model_name: str,
    x: NDArray[np.generic],
    prob_thresh: float | None = None,
    nms_thresh: float | None = None,
) -> NDArray[np.int32]:
    """Normalize a (y, x) or (y, x, c) array and predict its instance labels."""
    from csbdeep.utils import normalize

    axis_norm = (0, 1)  # per channel for YXC, the whole plane for YX
    labels, _ = _model(model_name).predict_instances(
        normalize(x, 1, 99.8, axis=axis_norm),
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
    )
    return labels.astype(np.int32)


def _upload_labels(
    mikro: Mikro,
    labels: xr.DataArray,
    name: str,
    axes: list[AxisInput],
    derived_from: list[DerivedFromInput],
) -> ArrayLens:
    """Upload a label map, stage a scene over it, and hand back a lens on all of it.

    No histogram anchor: a histogram over object ids would offer contrast
    limits for values that are not intensities.
    """
    dataset = mikro.create_array_dataset(
        data=labels,
        scales=[],
        name=name,
        axes=axes,
        anchors=[],
        derived_from=derived_from,
    )
    if dataset.intrinsic_system is None:
        raise RuntimeError(f"Uploaded dataset {dataset.id} has no intrinsic system")
    dataset.intrinsic_system.stage(name="Default")
    return dataset.lens()


def _publish(
    mikro: Mikro,
    image: ArrayLens,
    labels: NDArray[np.int32],
    space_axes: tuple[str, ...],
    source_dims: tuple[str, ...],
) -> ArrayLens:
    """Upload the labels as a categorical derivation of `image`."""
    n_objects = int(labels.max())
    array = xr.DataArray(labels, dims=space_axes)
    # The labels keep only the space axes. When the image had others (a pinned
    # channel, a single timepoint), the edge is a rank-dropping projection.
    if tuple(space_axes) == tuple(source_dims):
        edge = image.derive_identity(value_relation=ValueRelation.CATEGORIZED)
    else:
        edge = image.derive_projection(
            list(space_axes), value_relation=ValueRelation.CATEGORIZED
        )

    return _upload_labels(
        mikro,
        array,
        f"StarDist segmentation ({n_objects} objects)",
        image.carried_axes(space_axes),
        derived_from=[edge],
    )


@app.action(name="Segment Fluorescent Nuclei", collections=COLLECTIONS, logo=FLUO_LOGO)
def predict_flou2(
    image: Annotated[
        StillFluorescenceImage, Description("The fluorescence image to segment")
    ],
    prob_thresh: Annotated[
        float | None, Description("Probability threshold (model default if unset)")
    ] = None,
    nms_thresh: Annotated[
        float | None, Description("Overlap threshold (model default if unset)")
    ] = None,
    *,
    mikro: Mikro,
    task: Task,
) -> LabelImage:
    """Segments nuclei in a single-channel fluorescence image with the pretrained
    StarDist 2D_versatile_fluo model.
    """
    data = image.data
    space_axes = image.axes_of_type(AxisType.SPACE)
    # The spec allows a surviving size-1 channel or time axis: squeeze them.
    plane = data.squeeze([d for d in data.dims if d not in space_axes])

    task.progress(5, "Reading image")
    x = np.asarray(plane.transpose(*space_axes).data)

    task.progress(15, "Loading model and predicting")
    labels = _segment("2D_versatile_fluo", x, prob_thresh, nms_thresh)

    task.progress(90, "Uploading labels")
    return _publish(mikro, image, labels, space_axes, tuple(str(d) for d in data.dims))


@app.action(name="Segment H&E Nuclei", collections=COLLECTIONS, logo=HE_LOGO)
def predict_stardist_he(
    image: Annotated[StillRGBImage, Description("The H&E image to segment")],
    prob_thresh: Annotated[
        float | None, Description("Probability threshold (model default if unset)")
    ] = None,
    nms_thresh: Annotated[
        float | None, Description("Overlap threshold (model default if unset)")
    ] = None,
    *,
    mikro: Mikro,
    task: Task,
) -> LabelImage:
    """Segments nuclei in an RGB H&E brightfield image with the pretrained
    StarDist 2D_versatile_he model.
    """
    data = image.data
    space_axes = image.axes_of_type(AxisType.SPACE)
    (channel_axis,) = image.axes_of_type(AxisType.CHANNEL)
    plane = data.squeeze(
        [d for d in data.dims if d not in space_axes and d != channel_axis]
    )

    task.progress(5, "Reading image")
    # StarDist wants YXC
    x = np.asarray(plane.transpose(*space_axes, channel_axis).data)

    task.progress(15, "Loading model and predicting")
    labels = _segment("2D_versatile_he", x, prob_thresh, nms_thresh)

    task.progress(90, "Uploading labels")
    return _publish(mikro, image, labels, space_axes, tuple(str(d) for d in data.dims))


# The star is where the task is. Each action's latest task is a value a blok can read
# (``self.<action>.latest``). Progress runs from 0 to 100 and the clip from 0 to 1, and
# each report is a place to fly to rather than to jump to: the clip glides there over
# ``duration`` seconds. The clip follows whichever action is further along, since a
# blok expression has no "if" to pick the running one: after one action has finished,
# a run of the other only shows in the badge until it too is done.
FLUO = "self.tasks.predict_flou2"
HE = "self.tasks.predict_stardist_he"
PROGRESS = f"math.max({FLUO}.progress, {HE}.progress)"

app.blok(
    "starmist-3d",
    bsx(f"""
    <Scene camera="front" grid={{false}} height="100%">
      <Model id="star" src={{self.assets.star}}>
        <Animate clip="spiral" progress={{0}} animateTo={{math.div({PROGRESS}, 100)}} duration={{4}} />
        <Popout node="star" open={{{FLUO}.running}}>
          <Badge text={{{FLUO}.message}} />
        </Popout>
        <Popout node="star" open={{{HE}.running}}>
          <Badge text={{{HE}.message}} />
        </Popout>
      </Model>
    </Scene>
    """),
    description="The star flies around a colony of cells and segments them as far as the latest segmentation has come.",
    default=True,
)


if __name__ == "__main__":
    run(app)
