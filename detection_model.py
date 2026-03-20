import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from collections import deque

class ObjectDetector:
    """
    Object detection using YOLOv11 from Ultralytics
    """
    def __init__(self, model_size='nano', conf_thres=0.25, iou_thres=0.45, classes=None, device=None):
        """
        Initialize the object detector
        
        Args:
            model_size (str): Model size ('nano', 'small', 'medium', 'large', 'extra')
            conf_thres (float): Confidence threshold for detections
            iou_thres (float): IoU threshold for NMS
            classes (list): List of classes to detect (None for all classes)
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        
        # Set MPS fallback for operations not supported on Apple Silicon
        if self.device == 'mps':
            print("Using MPS device with CPU fallback for unsupported operations")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        print(f"Using device: {self.device} for object detection")
        
        # Map model size to model name
        model_map = {
            'nano': 'yolo11n.pt',
            'small': 'yolo11s.pt',
            'medium': 'yolo11m.pt',
            'large': 'yolo11l.pt',
            'extra': 'yolo11x.pt'
        }
        
        model_name = model_map.get(model_size.lower(), model_map['nano'])
        
        try:
            self.model = YOLO(model_name)
            print(f"Loaded YOLOv11 {model_size} model on {self.device}")
        except Exception as e:
            print(f"Error loading {model_name}: {e}, falling back to yolo11n.pt")
            self.model = YOLO('yolo11n.pt')
        
        # Set model parameters
        self.model.overrides['conf'] = conf_thres
        self.model.overrides['iou'] = iou_thres
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000
        
        if classes is not None:
            self.model.overrides['classes'] = classes
        
        # Initialize tracking trajectories
        self.tracking_trajectories = {}
    
    def detect(self, image, track=True):
        """
        Detect objects in an image
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            track (bool): Whether to track objects across frames
            
        Returns:
            tuple: (annotated_image, detections)
                - annotated_image (numpy.ndarray): Image with detections drawn
                - detections (list): List of detections [bbox, score, class_id, object_id]
        """
        detections = []
        annotated_image = image.copy()
        
        try:
            if track:
                results = self.model.track(image, verbose=False, device=self.device, persist=True)
            else:
                results = self.model.predict(image, verbose=False, device=self.device)
        except RuntimeError as e:
            err_str = str(e)
            if self.device == 'mps' and "not currently implemented for the MPS device" in err_str:
                print(f"MPS error during detection: {e}")
                print("Falling back to CPU for this frame")
                results = self.model.track(image, verbose=False, device='cpu', persist=True) if track else self.model.predict(image, verbose=False, device='cpu')
            elif self.device == 'cuda' and ("out of memory" in err_str.lower() or "CUDA" in err_str):
                print(f"CUDA OOM during detection: {e}")
                print("Falling back to CPU for this frame")
                results = self.model.track(image, verbose=False, device='cpu', persist=True) if track else self.model.predict(image, verbose=False, device='cpu')
            else:
                raise
        
        if not results or results[0] is None:
            return annotated_image, detections
        
        result = results[0]
        n_boxes = len(result.boxes) if result.boxes is not None else 0
        
        # Use Ultralytics built-in plot() for reliable bbox drawing (works on Jetson/Windows)
        try:
            annotated_image = result.plot()
        except Exception as e:
            print(f"[Detection] plot() failed: {e}")
            annotated_image = image.copy()
        
        if result.boxes is None or n_boxes == 0:
            return annotated_image, detections
        
        # Parse boxes using direct tensor access (avoids iteration bugs on Jetson)
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy()
        ids_t = result.boxes.id
        if ids_t is not None:
            ids_np = ids_t.cpu().numpy()
        else:
            ids_np = np.full(len(xyxy), np.nan)
        
        names = result.names if hasattr(result, 'names') else self.model.names
        
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            score = float(conf[i]) if conf.size > 0 else 0.0
            class_id = int(cls[i]) if cls.size > 0 else 0
            tid = int(ids_np[i]) if ids_np.size > 0 and not np.isnan(ids_np[i]) else None
            
            detections.append([[float(x1), float(y1), float(x2), float(y2)], score, class_id, tid])
            
            if track and tid is not None:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if tid not in self.tracking_trajectories:
                    self.tracking_trajectories[tid] = deque(maxlen=10)
                self.tracking_trajectories[tid].append((cx, cy))
        
        # Clean up stale trajectories
        if track and ids_np.size > 0:
            active_ids = set(int(x) for x in ids_np if not np.isnan(x))
            for tid in list(self.tracking_trajectories.keys()):
                if tid not in active_ids:
                    del self.tracking_trajectories[tid]
        
        # Draw trajectories on top
        for tid, trajectory in self.tracking_trajectories.items():
            for j in range(1, len(trajectory)):
                t = int(2 * (j / len(trajectory)) + 1)
                cv2.line(annotated_image,
                         (int(trajectory[j-1][0]), int(trajectory[j-1][1])),
                         (int(trajectory[j][0]), int(trajectory[j][1])),
                         (255, 255, 255), t)
        
        return annotated_image, detections
    
    def get_class_names(self):
        """
        Get the names of the classes that the model can detect
        
        Returns:
            list: List of class names
        """
        return self.model.names 