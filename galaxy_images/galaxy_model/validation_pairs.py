from __future__ import annotations

from typing import Mapping, Sequence

import torch


def reconstruct_hsc_legacy_pairs(
    anchor_image: torch.Tensor,
    same_galaxy: torch.Tensor,
    metadata: Sequence[Mapping[str, object]],
    survey_names: tuple[str, str] = ("hsc", "legacy"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct both survey images for every validation pair.

    Each neighbors batch stores one anchor image plus the paired same-galaxy
    image from the other survey. `anchor_survey` determines which tensor is the
    first-survey view and which is the second-survey view. Returns
    (images_of_survey_names[0], images_of_survey_names[1]); the default pair
    keeps the historical (HSC, Legacy) behaviour and the function name.
    """
    first, second = survey_names
    if anchor_image.shape != same_galaxy.shape:
        raise ValueError(
            "anchor_image and same_galaxy must have identical shapes to reconstruct pairs."
        )
    if len(metadata) != anchor_image.shape[0]:
        raise ValueError(
            f"metadata length {len(metadata)} does not match batch size {anchor_image.shape[0]}."
        )

    anchor_is_hsc = []
    for item in metadata:
        survey = item.get("anchor_survey")
        if survey == first:
            anchor_is_hsc.append(True)
        elif survey == second:
            anchor_is_hsc.append(False)
        else:
            raise ValueError(f"Unsupported anchor_survey value: {survey!r} (expected one of {survey_names})")

    anchor_is_hsc_tensor = torch.tensor(
        anchor_is_hsc,
        device=anchor_image.device,
        dtype=torch.bool,
    ).view(-1, 1, 1, 1)

    hsc_images = torch.where(anchor_is_hsc_tensor, anchor_image, same_galaxy)
    legacy_images = torch.where(anchor_is_hsc_tensor, same_galaxy, anchor_image)
    return hsc_images, legacy_images
