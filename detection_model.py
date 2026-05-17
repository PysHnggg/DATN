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
    def __init__(self, model_size='nano', conf_thres=0.25, iou_thres=0.45, classes=None, device=None,
                 weights_path=None):
        """
        Initialize the object detector
        
        Args:
            model_size (str): Model size ('nano', 'small', 'medium', 'large', 'extra')
            conf_thres (float): Confidence threshold for detections
            iou_thres (float): IoU threshold for NMS
            classes (list): List of classes to detect (None for all classes)
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
            weights_path (str): Optional path to custom .pt weights (e.g. 'best.pt').
                If provided, overrides model_size.
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
        
        if weights_path:
            try:
                self.model = YOLO(weights_path)
                print(f"Loaded custom YOLO weights from {weights_path} on {self.device}")
            except Exception as e:
                print(f"Error loading {weights_path}: {e}, falling back to yolo11n.pt")
                self.model = YOLO('yolo11n.pt')
        else:
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
        
        # Make a copy of the image for annotation
        annotated_image = image.copy()
        
        try:
            if track:
                # Run inference with tracking
                results = self.model.track(image, verbose=False, device=self.device, persist=True)
            else:
                # Run inference without tracking
                results = self.model.predict(image, verbose=False, device=self.device)
        except RuntimeError as e:
            # Handle potential MPS errors
            if self.device == 'mps' and "not currently implemented for the MPS device" in str(e):
                print(f"MPS error during detection: {e}")
                print("Falling back to CPU for this frame")
                if track:
                    results = self.model.track(image, verbose=False, device='cpu', persist=True)
                else:
                    results = self.model.predict(image, verbose=False, device='cpu')
            else:
                # Re-raise the error if not MPS or not an implementation error
                raise
        
        if track:
            # Clean up trajectories for objects that are no longer tracked
            for id_ in list(self.tracking_trajectories.keys()):
                if id_ not in [int(bbox.id) for predictions in results if predictions is not None 
                              for bbox in predictions.boxes if bbox.id is not None]:
                    del self.tracking_trajectories[id_]
            
            # Process results
            for predictions in results:
                if predictions is None:
                    continue
                
                if predictions.boxes is None:
                    continue
                
                # Process boxes
                for bbox in predictions.boxes:
                    # Extract information
                    scores = bbox.conf
                    classes = bbox.cls
                    bbox_coords = bbox.xyxy
                    
                    # Check if tracking IDs are available
                    if hasattr(bbox, 'id') and bbox.id is not None:
                        ids = bbox.id
                    else:
                        ids = [None] * len(scores)
                    
                    # Process each detection
                    for score, class_id, bbox_coord, id_ in zip(scores, classes, bbox_coords, ids):
                        xmin, ymin, xmax, ymax = bbox_coord.cpu().numpy()
                        
                        # Add to detections list
                        detections.append([
                            [xmin, ymin, xmax, ymax],  # bbox
                            float(score),              # confidence score
                            int(class_id),             # class id
                            int(id_) if id_ is not None else None  # object id
                        ])
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, 
                                     (int(xmin), int(ymin)), 
                                     (int(xmax), int(ymax)), 
                                     (0, 0, 225), 2)
                        
                        # Add label
                        label = f"ID: {int(id_) if id_ is not None else 'N/A'} {predictions.names[int(class_id)]} {float(score):.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(annotated_image, 
                                     (int(xmin), int(ymin)), 
                                     (int(xmin) + dim[0], int(ymin) - dim[1] - baseline), 
                                     (30, 30, 30), cv2.FILLED)
                        cv2.putText(annotated_image, label, 
                                   (int(xmin), int(ymin) - 7), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Update tracking trajectories
                        if id_ is not None:
                            centroid_x = (xmin + xmax) / 2
                            centroid_y = (ymin + ymax) / 2
                            
                            if int(id_) not in self.tracking_trajectories:
                                self.tracking_trajectories[int(id_)] = deque(maxlen=10)
                            
                            self.tracking_trajectories[int(id_)].append((centroid_x, centroid_y))
            
            # Draw trajectories
            for id_, trajectory in self.tracking_trajectories.items():
                for i in range(1, len(trajectory)):
                    thickness = int(2 * (i / len(trajectory)) + 1)
                    cv2.line(annotated_image, 
                            (int(trajectory[i-1][0]), int(trajectory[i-1][1])), 
                            (int(trajectory[i][0]), int(trajectory[i][1])), 
                            (255, 255, 255), thickness)
        
        else:
            # Process results for non-tracking mode
            for predictions in results:
                if predictions is None:
                    continue
                
                if predictions.boxes is None:
                    continue
                
                # Process boxes
                for bbox in predictions.boxes:
                    # Extract information
                    scores = bbox.conf
                    classes = bbox.cls
                    bbox_coords = bbox.xyxy
                    
                    # Process each detection
                    for score, class_id, bbox_coord in zip(scores, classes, bbox_coords):
                        xmin, ymin, xmax, ymax = bbox_coord.cpu().numpy()
                        
                        # Add to detections list
                        detections.append([
                            [xmin, ymin, xmax, ymax],  # bbox
                            float(score),              # confidence score
                            int(class_id),             # class id
                            None                       # object id (None for no tracking)
                        ])
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, 
                                     (int(xmin), int(ymin)), 
                                     (int(xmax), int(ymax)), 
                                     (0, 0, 225), 2)
                        
                        # Add label
                        label = f"{predictions.names[int(class_id)]} {float(score):.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(annotated_image, 
                                     (int(xmin), int(ymin)), 
                                     (int(xmin) + dim[0], int(ymin) - dim[1] - baseline), 
                                     (30, 30, 30), cv2.FILLED)
                        cv2.putText(annotated_image, label, 
                                   (int(xmin), int(ymin) - 7), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_image, detections
    
    def get_class_names(self):
        """
        Get the names of the classes that the model can detect
        
        Returns:
            list: List of class names
        """
        return self.model.names


class MultiObjectDetector:
    """
    Runs multiple YOLO models on the same frame and merges their detections
    into a single unified list. Remaps class_id and track_id so that outputs
    from different models never collide.

    Example:
        det = MultiObjectDetector(
            weights_list=["best.pt", "yolo11n.pt"],
            conf_thres=0.25, iou_thres=0.45, device="cuda",
        )
        annotated, dets = det.detect(frame, track=True)
        names = det.get_class_names()  # unified {global_id: name}
    """

    # Each detector's class_id / track_id is offset by model_index * OFFSET.
    # 10_000 is more than enough: COCO has 80 classes, custom rarely >hundreds.
    _ID_OFFSET = 10_000

    def __init__(self, weights_list, conf_thres=0.25, iou_thres=0.45, classes=None, device=None,
                 exclude_classes_per_model=None):
        """
        Args:
            weights_list (list[str]): paths to .pt weights, in priority order.
            exclude_classes_per_model (dict[int, set[int]]):
                {model_index: {local_class_ids_to_drop}}. Useful to prevent a
                generic model from re-detecting what a specialized model already
                covers (e.g. drop any class the custom model handles).
        """
        self.detectors = []
        self.merged_names = {}
        self.exclude_classes_per_model = exclude_classes_per_model or {}

        for idx, w in enumerate(weights_list):
            det = ObjectDetector(
                weights_path=w, conf_thres=conf_thres, iou_thres=iou_thres,
                classes=classes, device=device,
            )
            self.detectors.append(det)

            raw_names = det.get_class_names()
            local = dict(raw_names) if isinstance(raw_names, dict) \
                else {i: n for i, n in enumerate(raw_names)}
            excl = self.exclude_classes_per_model.get(idx, set())
            for k, v in local.items():
                if int(k) in excl:
                    continue
                self.merged_names[idx * self._ID_OFFSET + int(k)] = str(v)

    def _remap(self, model_idx, local_class_id, local_obj_id):
        new_cls = model_idx * self._ID_OFFSET + int(local_class_id)
        new_obj = None if local_obj_id is None else model_idx * self._ID_OFFSET + int(local_obj_id)
        return new_cls, new_obj

    def unmap_class_id(self, global_class_id):
        """Return (model_idx, local_class_id) for a global class id."""
        g = int(global_class_id)
        return g // self._ID_OFFSET, g % self._ID_OFFSET

    def detect(self, image, track=True):
        annotated = image.copy()
        merged = []
        for idx, det in enumerate(self.detectors):
            excl = self.exclude_classes_per_model.get(idx, set())
            try:
                _, dets = det.detect(image.copy(), track=track)
            except Exception as e:
                print(f"[MultiObjectDetector] model {idx} detect error: {e}")
                continue
            for bbox, score, class_id, obj_id in dets:
                if int(class_id) in excl:
                    continue
                new_cls, new_obj = self._remap(idx, class_id, obj_id)
                merged.append([bbox, float(score), new_cls, new_obj])
        return annotated, merged

    def get_class_names(self):
        """Unified {global_id: name} across all models."""
        return dict(self.merged_names)
