import torch

from anyboxes import TorchBoxes


def apply_transformation(detections: torch.Tensor) -> torch.Tensor:
    boxes = TorchBoxes.from_top_left_corner(detections)
    boxes = boxes.to_center()
    return boxes.as_tensor


if __name__ == "__main__":
    detections = torch.randint(0, 1000, (5, 4))
    print(f"Detections:\n {detections}")
    boxes = apply_transformation(detections)
    print(f"Bboxes transformed: \n{boxes}")
