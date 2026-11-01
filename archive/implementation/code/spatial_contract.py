"""CROSS-style image-plane vs allocentric prompt contract. No pixel overlays."""

SPATIAL_CONTRACT = (
    "Spatial contract: Decide image-plane relations from the frame "
    "(screen left/right, top/bottom) unless the question names heading, north, "
    "or an object's own front. Do not treat camera motion as object motion. "
    "Nadir UAV views have no camera-relative ground 'in front'; use the image axes. "
    "Oblique views still keep left/right in the image plane unless a ground heading "
    "is stated. Then answer with the option's letter from the given choices directly."
)
