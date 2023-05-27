import torch

BOXES_TOP_LEFT_ORIGIN_UNIQUE = torch.tensor([[0.0, 0.0, 3.0, 2.0]])
BOXES_BOTTOM_LEFT_ORIGIN_UNIQUE = torch.tensor([[0.0, 2.0, 3.0, 2.0]])
BOXES_CENTER_UNIQUE = torch.tensor([[1.5, 1.0, 3.0, 2.0]])
BOXES_TWO_CORNERS_UNIQUE = torch.tensor([[0.0, 0.0, 3.0, 2.0]])
BOXES_CENTER_SQUARED_UNIQUE = torch.tensor([[1.5, 1.0, 3.0, 3.0]])

BOXES_TOP_LEFT_ORIGIN = torch.tensor([[0.0, 0.0, 3.0, 2.0], [10.0, 10.0, 10.0, 10.0]])
BOXES_BOTTOM_LEFT_ORIGIN = torch.tensor(
    [[0.0, 2.0, 3.0, 2.0], [10.0, 20.0, 10.0, 10.0]]
)
BOXES_CENTER = torch.tensor([[1.5, 1.0, 3.0, 2.0], [15.0, 15.0, 10.0, 10.0]])
BOXES_TWO_CORNERS = torch.tensor([[0.0, 0.0, 3.0, 2.0], [10.0, 10.0, 20.0, 20.0]])
BOXES_CENTER_SQUARED = torch.tensor([[1.5, 1.0, 3.0, 3.0], [15.0, 15.0, 10.0, 10.0]])

IMAGE_HEIGHT = 30
EXPECTED_BOXES_TOP_LEFT_WITH_MODIFY_ORIGIN = torch.tensor(
    [[0.0, 30.0, 3.0, 2.0], [10.0, 20.0, 10.0, 10.0]]
)

MASK_DIMENSION = (10, 10)
MASK = torch.tensor(
    [
        [
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    ]
).to(torch.int)
