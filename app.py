from stardist.models import StarDist3D
from csbdeep.utils import Path, normalize
import sys
import numpy as np
import xarray as xr
from mikro_next.api.schema import (
   from_array_like, 
   Image,
   PartialDerivedViewInput,
   PartialPixelViewInput,
   PartialRGBViewInput,
   create_rgb_context,
   ColorMap,
)
import numpy as np
from pydantic import Field
from arkitekt_next.tqdm import tqdm as atdqm
from arkitekt_next import easy
from stardist import (
    fill_label_holes,
    random_label_cmap,
    calculate_extents,
    gputools_available,
)
from stardist import Rays_GoldenSpiral
from stardist.models import StarDist2D
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D, StarDistData3D
from tqdm import tqdm
import shutil
import uuid
from arkitekt_next import register
from enum import Enum
from typing import Optional
from concurrent.futures import ProcessPoolExecutor


class PreTrainedModels(str, Enum):
    STARDIST_ORGANOID_3D = "stardist3"
    STARDIST_STYLED = "stardist_styled"



@register(collections=["segmentation", "prediction", "nuclei"])
def predict_flou2(image: Image) -> Image:
    """Segment Flou2

    Segments Cells using the stardist flou2 pretrained model

    Args:
        image (Image): The Input Image.
    Returns:
        Image: An Image with the Segmented Cells.

    """
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    x = image.data.sel(c=0, t=0, z=0).transpose(*"xy").data.compute()
    x = normalize(x)

    labels, details = model.predict_instances(x)

    array = xr.DataArray(labels, dims=list("xy"))
    
    print(array.max())
    

    nana = from_array_like(
        array,
        name="Segmented " + image.name,
        derived_views=[PartialDerivedViewInput(originImage=image)],
        rgb_views=[PartialRGBViewInput(cMin=0, cMax=0, contrastLimitMin=0, contrastLimitMax=array.max(), colorMap=ColorMap.VIRIDIS, baseColor=[0, 0, 0])],
        pixel_views=[PartialPixelViewInput()],
    )
    return nana

