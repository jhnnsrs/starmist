"""Download the pretrained StarDist models at image build time."""

from stardist.models import StarDist2D

for name in ("2D_versatile_fluo", "2D_versatile_he"):
    StarDist2D.from_pretrained(name)
