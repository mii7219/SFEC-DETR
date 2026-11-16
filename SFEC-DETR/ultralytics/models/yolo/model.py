# Ultralytics YOLO, AGPL-3.0 license

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel


class YOLO(Model):
    @property
    def task_map(self):
        return {'detect': {
            'model': DetectionModel,
            'trainer': yolo.detect.DetectionTrainer,
            'validator': yolo.detect.DetectionValidator,
            'predictor': yolo.detect.DetectionPredictor,
        }}
