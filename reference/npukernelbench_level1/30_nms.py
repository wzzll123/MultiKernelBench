import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, boxes: torch.Tensor, scores: torch.Tensor,
                max_output_size: int, iou_threshold: float,
                scores_threshold: float, pad_to_max_output_size: bool = False):
        """
        Performs Non-Maximum Suppression (NMS) on bounding boxes.
        Pure PyTorch reference implementation.
        """
        boxes_f32 = boxes.float()
        scores_f32 = scores.float()

        score_mask = scores_f32 > scores_threshold
        filtered_boxes = boxes_f32[score_mask]
        filtered_scores = scores_f32[score_mask]
        original_indices = torch.where(score_mask)[0]

        selected_indices = torch.zeros(max_output_size, dtype=torch.int32, device=boxes.device)

        if filtered_boxes.shape[0] == 0:
            num_selected = torch.tensor(0, dtype=torch.int32, device=boxes.device)
            return selected_indices, num_selected

        sorted_indices = torch.argsort(filtered_scores, descending=True, stable=True)
        sorted_boxes = filtered_boxes[sorted_indices]
        sorted_original_indices = original_indices[sorted_indices]

        num_boxes = sorted_boxes.shape[0]
        selected_indices_list = []
        suppressed = torch.zeros(num_boxes, dtype=torch.bool, device=boxes.device)

        areas = (sorted_boxes[:, 2] - sorted_boxes[:, 0]) * (sorted_boxes[:, 3] - sorted_boxes[:, 1])

        for i in range(num_boxes):
            if suppressed[i]:
                continue

            selected_indices_list.append(sorted_original_indices[i].item())

            if len(selected_indices_list) >= max_output_size:
                break

            rest = torch.arange(i + 1, num_boxes, device=boxes.device)
            if rest.numel() == 0:
                break
            mask = ~suppressed[rest]
            if not mask.any():
                continue
            candidates = rest[mask]

            cur_box = sorted_boxes[i]
            cand_boxes = sorted_boxes[candidates]

            x1_inter = torch.maximum(cur_box[0].expand(cand_boxes.shape[0]), cand_boxes[:, 0])
            y1_inter = torch.maximum(cur_box[1].expand(cand_boxes.shape[0]), cand_boxes[:, 1])
            x2_inter = torch.minimum(cur_box[2].expand(cand_boxes.shape[0]), cand_boxes[:, 2])
            y2_inter = torch.minimum(cur_box[3].expand(cand_boxes.shape[0]), cand_boxes[:, 3])

            inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
            union_area = areas[i] + areas[candidates] - inter_area
            iou = inter_area / union_area.clamp(min=1e-6)

            suppress_mask = iou >= iou_threshold
            suppressed[candidates[suppress_mask]] = True

        num_selected = len(selected_indices_list)

        if num_selected > 0:
            selected_indices[:num_selected] = torch.tensor(
                selected_indices_list, dtype=torch.int32, device=boxes.device
            )

        num_selected_tensor = torch.tensor(num_selected, dtype=torch.int32, device=boxes.device)

        return selected_indices, num_selected_tensor


def _make_legal_boxes(shape, dtype):
    assert len(shape) == 2 and shape[1] == 4, f"boxes shape must be [N, 4], got {shape}"
    n = shape[0]

    raw = torch.randn(n, 2, 2, dtype=torch.float32)
    pt_a = raw[:, 0, :]
    pt_b = raw[:, 1, :]

    x1 = torch.minimum(pt_a[:, 0], pt_b[:, 0])
    y1 = torch.minimum(pt_a[:, 1], pt_b[:, 1])
    x2 = torch.maximum(pt_a[:, 0], pt_b[:, 0])
    y2 = torch.maximum(pt_a[:, 1], pt_b[:, 1])

    eps = 1e-3
    x2 = torch.where(x2 - x1 < eps, x1 + eps, x2)
    y2 = torch.where(y2 - y1 < eps, y1 + eps, y2)

    boxes = torch.stack([x1, y1, x2, y2], dim=-1).to(dtype)
    return boxes


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "30_nms.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        boxes_info = inputs[0]
        scores_info = inputs[1]
        max_output_size_info = inputs[2]
        iou_threshold_info = inputs[3]
        scores_threshold_info = inputs[4]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[boxes_info["dtype"]]

        boxes = _make_legal_boxes(boxes_info["shape"], dtype)
        scores = torch.randn(scores_info["shape"], dtype=dtype)

        max_output_size = max_output_size_info["value"]
        iou_threshold = iou_threshold_info["value"]
        scores_threshold = scores_threshold_info["value"]
        input_groups.append([boxes, scores, max_output_size, iou_threshold, scores_threshold])
    return input_groups


def get_init_inputs():
    return []
