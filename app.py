"""Starmist: StarDist nucleus segmentation for Arkitekt

Two 2D actions over the pretrained StarDist models, one for fluorescence and one
for H&E brightfield. Each takes a spec-typed image lens and returns a label image
that derives from it, so the result lands in the same coordinate space as the
input.

Run:  arkitekt run dev
"""

import functools
from typing import TYPE_CHECKING, Annotated

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from arkitekt import App, Description, Task, run
from mikro import Mikro, mikro_service
from mikro.api.schema import AxisInput, AxisType, DerivedFromInput, Lens, ValueRelation
from mikro.arkitekt.specs import (
    N_TIMEPOINTS,
    VALUE_KIND,
    Image,
    RGBImage,
    SingleChannelImage,
    at_most,
    ensure,
    exactly,
)

if TYPE_CHECKING:
    from stardist.models import StarDist2D


app = App(
    "starmist",
    "0.1.0",
    description="StarDist nucleus segmentation with the pretrained 2D models",
    author="jhnnsrs",
)
app.service(mikro_service)

COLLECTIONS = ["segmentation", "prediction", "nuclei"]

# Spelled out rather than nesting `Still`: a spec alias used as metadata inside
# another Annotated is not flattened, so its constraint would silently vanish.
StillFluorescenceImage = Annotated[SingleChannelImage, *at_most(N_TIMEPOINTS, 1)]
"""A single-channel plane with at most one timepoint."""

StillRGBImage = Annotated[RGBImage, *at_most(N_TIMEPOINTS, 1)]
"""An RGB plane with at most one timepoint."""

LabelImage = Annotated[Image, *exactly(VALUE_KIND, "categorical")]
"""A plane of object ids: what both actions produce."""


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
) -> Lens:
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
    image: Lens,
    labels: NDArray[np.int32],
    space_axes: tuple[str, ...],
    source_dims: tuple[str, ...],
) -> Lens:
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

    lens = _upload_labels(
        mikro,
        array,
        f"StarDist segmentation ({n_objects} objects)",
        image.carried_axes(space_axes),
        derived_from=[edge],
    )
    return ensure(lens, LabelImage, declares={VALUE_KIND: "categorical"})


@app.action(name="Segment Fluorescent Nuclei", collections=COLLECTIONS)
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

    task.progress(10, "Reading image")
    x = np.asarray(plane.transpose(*space_axes).data)

    task.progress(30, "Loading model and predicting")
    labels = _segment("2D_versatile_fluo", x, prob_thresh, nms_thresh)

    task.progress(80, "Uploading labels")
    return _publish(mikro, image, labels, space_axes, tuple(str(d) for d in data.dims))


@app.action(name="Segment H&E Nuclei", collections=COLLECTIONS)
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

    task.progress(10, "Reading image")
    # StarDist wants YXC
    x = np.asarray(plane.transpose(*space_axes, channel_axis).data)

    task.progress(30, "Loading model and predicting")
    labels = _segment("2D_versatile_he", x, prob_thresh, nms_thresh)

    task.progress(80, "Uploading labels")
    return _publish(mikro, image, labels, space_axes, tuple(str(d) for d in data.dims))


if __name__ == "__main__":
    run(app)
