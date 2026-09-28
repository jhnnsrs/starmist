# Starmist

Starmist is Stardist enabled for the Arkitekt platform. You can check out the original Stardist repository [here](https://github.com/mpicbg-csbd/stardist).

## Actions

| Action | Input | Model |
| --- | --- | --- |
| Segment Fluorescent Nuclei (`predict_flou2`) | a single-channel 2D image, at most one timepoint | `2D_versatile_fluo` |
| Segment H&E Nuclei (`predict_stardist_he`) | an RGB 2D image (one channel axis, 3 channels), at most one timepoint | `2D_versatile_he` |

Both actions return a categorical label image that derives from the input, so it lands in the input's coordinate space. The probability and overlap thresholds are optional; left unset, the model defaults apply.

## Running

```bash
pip install stardist tensorflow "arkitekt[all]" mikro
arkitekt run dev
```

The GPU image is built from `.arkitekt/flavours/vanilla/` (`arkitekt plugin build`), and it bakes both pretrained models in via `init.py`.

## Comparison to Segmentor

Starmist currently only supports 2D images, while Segmentor supports 2D and 3D images. Additionally Starmist does not support training yet. As of now, Starmist is only a prediction tool.

## Roadmap

- [ ] Add support for 3D images
- [ ] Add support for training (utilizing a new model backend)
