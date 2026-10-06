"""Behaviour of the U-Net's size handling: a Grid that is not a multiple of 4 is padded with plain wall."""

import torch
import torch.nn.functional as F

from unmaze.model import UNet


def randomised_unet() -> UNet:
    torch.manual_seed(0)
    model = UNet(base=8).eval()
    torch.nn.init.normal_(model.out[-1].weight, std=0.2)  # the real net starts at zero output; make it speak
    return model


def test_a_grid_that_needs_padding_is_treated_exactly_as_if_it_were_padded_with_wall_by_hand():
    model = randomised_unet()
    # A hand-made 13x13 Grid (3 pixels short of 16): random walls, a start and a goal.
    walls = (torch.rand(3, 1, 13, 13) > 0.5).float()
    start, goal = torch.zeros(3, 1, 13, 13), torch.zeros(3, 1, 13, 13)
    start[:, :, 1, 1] = goal[:, :, 11, 11] = 1.0
    cond = torch.cat([walls, start, goal], dim=1)
    x_t = torch.randn(3, 1, 13, 13)
    t = torch.tensor([10, 500, 999])

    as_given = model(x_t, t, cond)

    # The same Puzzle with the missing 3 rows and columns drawn in by hand: wall, no start, no goal,
    # and a noisy mask that says "no path" there.
    pad = (0, 3, 0, 3)
    by_hand_cond = torch.cat(
        [F.pad(cond[:, :1], pad, value=1.0), F.pad(cond[:, 1:], pad, value=0.0)], dim=1
    )
    by_hand = model(F.pad(x_t, pad, value=-1.0), t, by_hand_cond)[:, :, :13, :13]

    assert torch.allclose(as_given, by_hand, atol=1e-5)
