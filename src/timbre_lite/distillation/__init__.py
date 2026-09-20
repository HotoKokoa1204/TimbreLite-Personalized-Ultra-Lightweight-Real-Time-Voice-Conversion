"""Offline ContentVec teacher knowledge distillation modules."""

from timbre_lite.distillation.alignment import align_teacher_features
from timbre_lite.distillation.loss import DistillationLoss, DistillationProjectionHead
from timbre_lite.distillation.teacher import ContentVecTeacher

__all__ = [
    "ContentVecTeacher",
    "DistillationLoss",
    "DistillationProjectionHead",
    "align_teacher_features",
]
