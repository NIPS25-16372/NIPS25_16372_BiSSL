import copy
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, get_args

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from source.types import BackboneArchs, ConvBackbones, DetectionBackbones

from .misc import ConvBackbone


### CLASSIFICAITON ###
class DSClassifier(nn.Module):
    def __init__(
        self,
        backbone_arch: BackboneArchs,
        n_classes: int = 10,
    ):
        super().__init__()
        if backbone_arch in get_args(ConvBackbones):
            self.backbone, self.num_features = ConvBackbone(backbone_arch)  # type: ignore
            assert self.num_features is not None
            self.head = nn.Linear(self.num_features, n_classes)
        else:
            raise ValueError(f"Backbone {backbone_arch} not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


### OBJECT DETECTION  ###
class FasterRCNNR50C4BB(FasterRCNN):  # type: ignore
    def __init__(
        self,
        backbone_arch: DetectionBackbones,
        n_classes: int,
        rpn_anchor_generator: Optional[AnchorGenerator] = None,
        box_roi_pool: Optional[MultiScaleRoIAlign] = None,
        min_size: int = 196,
        max_size: int = 320,
        pt_img_size: int = 96,
        *args,
        **kwargs,
    ):
        assert backbone_arch == "resnet50", "Only resnet50 is supported for now"

        # backbone, num_features = Backbone(args_arch)
        class BackbonePlaceholder:
            out_channels: int = (
                1024  # Specifically for resnet50, should be adapted to the specific backbones if future versions support such
            )

        if rpn_anchor_generator is None:
            rpn_anchor_generator = AnchorGenerator(
                sizes=((16, 32, 64, 128, 256),),
                aspect_ratios=((0.5, 1.0, 2.0),),
            )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0"],
                output_size=7,
                sampling_ratio=2,
                canonical_scale=pt_img_size,  # Backbones pretrained on 96x96 images
            )

        super().__init__(
            BackbonePlaceholder(),
            num_classes=n_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=box_roi_pool,
            min_size=min_size,
            max_size=max_size,
            *args,
            **kwargs,
        )
        del self.backbone

        rpn = copy.deepcopy(self.rpn)
        del self.rpn
        roi_heads = copy.deepcopy(self.roi_heads)
        del self.roi_heads

        class GeneralRCNNHeadParameters(nn.Module):
            def __init__(self):
                super().__init__()
                self.rpn = rpn
                self.roi_heads = roi_heads

            def forward(self, images, features, targets, return_loss=False):
                proposals, proposal_losses = self.rpn(images, features, targets)
                detections, detector_losses = self.roi_heads(
                    features, proposals, images.image_sizes, targets
                )
                if return_loss:
                    return detections, proposal_losses, detector_losses
                else:
                    return detections

        self.backbone, self.num_features = ConvBackbone(backbone_arch)

        # Converting the ResNet50 backbone to a ResNet50-C4 backbone
        self.backbone.layer4 = nn.Identity()
        del self.backbone.global_pool
        del self.backbone.fc
        self.backbone.forward = self.backbone.forward_features

        # Defining head
        self.head = GeneralRCNNHeadParameters()

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        ext_backbone: Optional[torch.nn.Module] = None,
    ) -> (
        Tuple[Dict[str, torch.Tensor]]
        | Tuple[Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]]
    ):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(
                            False,
                            f"Expected target boxes to be of type Tensor, got {type(boxes)}.",
                        )

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = (
                    boxes[:, 2:] == boxes[:, :2]  # MODIFIED from <= to ==
                )
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        ### MODIFICATION ###
        if ext_backbone is None:
            features = self.backbone(images.tensors)  # type: ignore
        else:
            features = ext_backbone(images.tensors)  # type: ignore

        ### END MODIFICATION ###

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        ### MODIFICATION ###
        detections, proposal_losses, detector_losses = self.head(
            images=images,
            features=features,
            targets=targets,
            return_loss=True,
        )
        ### END MODIFICATION ###

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():  # type: ignore
            if not self._has_warned:  # pylint: disable=E0203
                warnings.warn(
                    "RCNN always returns a (Losses, Detections) tuple in scripting"
                )
                self._has_warned = True
            return losses, detections  # type: ignore
        else:
            return self.eager_outputs(losses, detections)  # type: ignore
