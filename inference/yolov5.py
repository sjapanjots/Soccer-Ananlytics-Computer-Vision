from typing import List

import os

import numpy as np
import pandas as pd
import torch

from inference.base_detector import BaseDetector


class YoloV5(BaseDetector):
    def __init__(
        self,
        model_path: str = None,
    ):
        """
        Initialize detector

        Parameters
        ----------
        model_path : str, optional
            Path to model, by default None. If it's None, it will download the model with COCO weights
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        if model_path and not os.path.exists(model_path):
            print(
                f"Custom model not found at {model_path}; "
                "falling back to COCO-pretrained yolov5x."
            )
            model_path = None

        if model_path:
            self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
        else:
            self.model = torch.hub.load(
                "ultralytics/yolov5", "yolov5x", pretrained=True
            )

    def predict(self, input_image: List[np.ndarray]) -> pd.DataFrame:
        """
        Predicts the bounding boxes of the objects in the image

        Parameters
        ----------
        input_image : List[np.ndarray]
            List of input images

        Returns
        -------
        pd.DataFrame
            DataFrame containing the bounding boxes
        """

        result = self.model(input_image, size=640)

        return result.pandas().xyxy[0]
