"""
Refactored GazeFollow Dataset Processor

This module processes GazeFollow dataset annotations with LLaVA gaze prediction results,
matching ground truth gaze points with predicted gaze points and calculating error metrics.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from functools import partial
import queue
from threading import Thread

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image

# Add parent directory to path to import docs module
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from docs.research_utils import fix_wsl_paths

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def iter_gaze_point_files(base_dir: Path):
    """Generator that yields gaze point files one at a time."""
    for root, _, files in os.walk(base_dir):
        for file in files:
            if 'gaze' in file and file.endswith('attn_map_smooth_centers.pt'):
                yield Path(root) / file


def iter_person_files(base_dir: Path):
    """Generator that yields person segmentation files one at a time."""
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('all_segmentation_results.json'):
                yield Path(root) / file


def iter_gaze_and_person_files(base_dir: Path):
    """Yield each layer directory together with its gaze and person files."""

    base_dir = Path(base_dir)
    dir_data: Dict[Path, Dict[str, List[Path]]] = defaultdict(lambda: {'gaze': [], 'person': []})

    for root, dirs, files in os.walk(base_dir, topdown=True):
        root_path = Path(root)
        data = dir_data[root_path]

        # Skip if not a layer directory
        if not root_path.name.startswith('layer_'):
            continue
            
        # Extract layer number and image name from path
        layer_name = root_path.name
        parent_path = root_path.parent
        if not parent_path.name.endswith('_attn'):
            continue
            
        image_name = parent_path.name.replace('_attn', '')
        
        # Look for gaze files in the expected structure
        each_person_dir = root_path / 'each_person_attn_maps'
        if not each_person_dir.exists():
            continue
            
        gaze_files = list(each_person_dir.glob('**/gaze*_smooth_centers.pt'))

        # Look for person segmentation file
        person_files = []
        person_files = list(root_path.glob("segmentation_results/*_all_segmentation_results.json"))
        
        # Yield if we have both types of files
        # if gaze_files and person_files:
        yield root_path, gaze_files, person_files


def process_gaze_file(gaze_points_path: Path) -> Optional[Tuple[str, str, torch.Tensor]]:
    """Process a single gaze point file.
    
    Args:
        gaze_points_path: Path to the gaze point file
        
    Returns:
        Tuple of (image_name, person_id, gaze_point) or None if processing failed
    """
    try:
        gaze_filename = gaze_points_path.stem
        
        # Load gaze points safely
        gaze_data = torch.load(gaze_points_path, weights_only=False)
        if 'centers' not in gaze_data:
            return None
            
        gaze_point = gaze_data['centers']
        
        # Extract image name and person ID
        image_name_parts = [val.split('_')[0] for val in gaze_points_path.parts if val.endswith('_attn')]
        if not image_name_parts:
            return None
            
        image_name = image_name_parts[0]
        person_id = gaze_filename.split('_')[2]
        
        return (image_name, person_id, gaze_point)
    except Exception as e:
        logger.warning(f"Error processing gaze file {gaze_points_path}: {e}")
        return None


def process_person_file(person_file_info: Tuple[Path, pd.DataFrame, Path]) -> Optional[Tuple[str, Dict]]:
    """Process a single person segmentation file.
    
    Args:
        person_file_info: Tuple of (person_file, compact_df, base_data_dir_path)
        
    Returns:
        Tuple of (image_id, person_data) or None if processing failed
    """
    person_file, compact_df, base_data_dir_path = person_file_info
    
    try:
        image_id = person_file.stem.split('_')[0]
        
        # Load person data
        with open(person_file, 'r') as f:
            person_data = json.load(f)
        
        # Rename person keys (remove 'person_' prefix)
        # person_keys = list(person_data.keys())
        new_person_data = {}
        for key in person_data.keys():
            if key.startswith('person_'):
                new_key = key.replace('person_', '')
                new_person_data[new_key] = person_data[key]
        person_data = new_person_data

        # Get image path from compact_df
        matching_rows = compact_df[compact_df['image_key'] == image_id]
        if matching_rows.empty:
            return None
            
        image_path = Path(base_data_dir_path) / Path(matching_rows['image_path'].values[0])
        
        if not image_path.exists():
            return None
            
        with Image.open(str(image_path)) as img:
            w, h = img.size
            person_data['image_shape'] = (h, w)
        
        return (image_id, person_data)
    except Exception as e:
        logger.warning(f"Error processing person file {person_file}: {e}")
        return None


class GazePointProcessor:
    """Handles gaze point filtering and processing operations."""
    
    @staticmethod
    def filter_gaze_points(points: Union[torch.Tensor, np.ndarray], 
                          threshold_factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter outliers from a 2D array of coordinate points using median-based approach.

        Args:
            points: Tensor or array of shape (n, 2) containing x,y coordinates
            threshold_factor: Factor to multiply MAD for determining outlier threshold

        Returns:
            tuple: (filtered_mean, filtered_points)
        """
        if isinstance(points, torch.Tensor):
            points_np = points.cpu().numpy()
        else:
            points_np = np.array(points)

        if len(points_np) == 0:
            return np.array([0, 0]), np.array([])

        # Calculate the median point
        median_point = np.median(points_np, axis=0)

        # Calculate distance of each point from the median point
        distances = np.sqrt(np.sum((points_np - median_point) ** 2, axis=1))

        # Calculate the median absolute deviation (MAD)
        mad = np.median(np.abs(distances - np.median(distances)))

        # Define the threshold (adjusted with the scaling factor)
        threshold = np.median(distances) + threshold_factor * mad

        # Find indices of inliers
        inliers_mask = distances <= threshold
        filtered_points = points_np[inliers_mask]

        # Calculate mean of filtered points
        if len(filtered_points) > 0:
            filtered_mean = np.mean(filtered_points, axis=0)
        else:
            filtered_mean = median_point  # Fallback to median if no inliers

        return filtered_mean, filtered_points

    @staticmethod
    def calculate_angular_distance(pred_gaze: np.ndarray, gt_gaze: np.ndarray, 
                                 eye_pos: np.ndarray) -> float:
        """
        Calculate the angular distance between predicted and ground truth gaze points.
        
        Args:
            pred_gaze: predicted gaze point in normalized coordinates [0, 1]
            gt_gaze: ground truth gaze point in normalized coordinates [0, 1]  
            eye_pos: eye position in normalized coordinates [0, 1]
        
        Returns:
            angular distance in degrees
        """
        pred_gaze = np.array(pred_gaze)
        gt_gaze = np.array(gt_gaze)
        eye_pos = np.array(eye_pos)
        
        # Calculate vectors from eye to gaze points
        pred_vector = pred_gaze - eye_pos
        gt_vector = gt_gaze - eye_pos
        
        # Calculate magnitudes
        pred_magnitude = np.linalg.norm(pred_vector)
        gt_magnitude = np.linalg.norm(gt_vector)
        
        # Handle edge case where eye position equals gaze position
        if pred_magnitude == 0 or gt_magnitude == 0:
            return 0.0
        
        # Normalize vectors
        pred_unit = pred_vector / pred_magnitude
        gt_unit = gt_vector / gt_magnitude
        
        # Calculate dot product and clip to avoid numerical errors
        dot_product = np.clip(np.dot(pred_unit, gt_unit), -1.0, 1.0)
        
        # Calculate angular distance in radians, then convert to degrees
        angular_distance_rad = np.arccos(dot_product)
        angular_distance_deg = np.degrees(angular_distance_rad)
        
        return angular_distance_deg


class PersonMatcher:
    """Handles person matching between ground truth and predictions using IoU."""
    
    @staticmethod
    def calculate_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Each bbox is [x1, y1, x2, y2].
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
        area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
        union_area = area1 + area2 - intersection_area

        if union_area == 0:
            return 0.0
        return intersection_area / union_area

    def find_best_matching_person(self, llava_person_result: Dict, 
                                gt_person_bbox: List[float]) -> Tuple[Optional[str], float]:
        """
        Find the best matching person based on IoU overlap.
        
        Args:
            llava_person_result: Dictionary containing person detection results
            gt_person_bbox: Ground truth person bounding box [x1, y1, x2, y2]
            
        Returns:
            Tuple of (matched_person_id, best_iou)
        """
        all_intersects = []
        
        for person_id, person_data in llava_person_result.items():
            if 'image_shape' in person_id or 'person' not in person_data:
                continue
                
            # Validate person data structure
            if not isinstance(person_data.get('person'), dict):
                continue
            if 'boxes' not in person_data['person']:
                continue
            if not person_data['person']['boxes']:
                continue
                
            person_bbox = person_data['person']['boxes'][0]
            iou = self.calculate_bbox_iou(gt_person_bbox, person_bbox)
            
            if iou > 0:
                all_intersects.append([iou, person_id])
        
        if not all_intersects:
            return None, 0.0
            
        all_intersects = np.array(all_intersects)
        best_match_idx = np.argmax(all_intersects[:, 0])
        matched_person_id = all_intersects[best_match_idx][1]
        matched_iou = float(all_intersects[best_match_idx][0])
        
        return matched_person_id, matched_iou


class DataValidator:
    """Handles data validation and error message generation."""
    
    @staticmethod
    def validate_annotation_row(row: pd.Series) -> Optional[str]:
        """Validate a single annotation row and return error message if invalid."""
        required_numeric_cols = ['eye_x', 'eye_y', 'gaze_x', 'gaze_y', 
                               'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height']
        
        for col in required_numeric_cols:
            if pd.isna(row.get(col)):
                return f"Missing or invalid {col} value"
                
        return None
    
    @staticmethod
    def validate_image_path(image_path: str, base_data_dir: Path) -> Optional[str]:
        """Validate that image path exists and return error message if not."""
        if not image_path:
            return "Empty image path"
            
        # Convert path format
        normalized_path = image_path.replace("\\", os.sep).replace("D:", str(base_data_dir))
        full_path = Path(normalized_path)
        
        if not full_path.exists():
            return f"Image file not found: {full_path}"
            
        return None
    
    @staticmethod
    def validate_gaze_data(gaze_data: Dict, person_id: str) -> Optional[str]:
        """Validate gaze data structure and return error message if invalid."""
        if not gaze_data:
            return "No gaze data found"
            
        person_gaze = gaze_data.get(person_id)
        if not person_gaze:
            return f"No gaze data for person {person_id}"
            
        if not isinstance(person_gaze, dict):
            return f"Invalid gaze data structure for person {person_id}"
            
        if 'mean_point' not in person_gaze:
            return f"No mean_point in gaze data for person {person_id}"
            
        return None


class GazeFollowDatasetProcessor:
    """Main processor for GazeFollow dataset with LLaVA gaze prediction results."""
    
    def __init__(self, annot_path: str, base_data_dir_path: str, llava_results_dir: str, 
                 resume_from_csv: Optional[str] = None, num_workers: Optional[int] = None, 
                 save_interval: int = 1000):
        """Initialize the processor with required paths.
        
        Args:
            annot_path: Path to annotation file
            base_data_dir_path: Base directory for image data
            llava_results_dir: Directory containing LLaVA results
            resume_from_csv: Optional path to previous CSV file to resume from
            num_workers: Number of parallel workers (None for auto-detect)
            save_interval: Number of processed rows between temporary saves (default: 1000)
        """
        self.annot_path = fix_wsl_paths(annot_path)
        self.base_data_dir_path = Path(fix_wsl_paths(base_data_dir_path))
        self.llava_results_dir = Path(fix_wsl_paths(llava_results_dir))
        self.results_path = self.llava_results_dir.parent / f"{self.llava_results_dir.name}_results.csv"
        self.temp_results_path = self.llava_results_dir.parent / f"{self.llava_results_dir.name}_results_temp.csv"
        self.resume_from_csv = fix_wsl_paths(resume_from_csv) if resume_from_csv else None
        self.num_workers = num_workers if num_workers else max(1, cpu_count() - 1)
        self.batch_size = self.num_workers * 500  # Larger batches for better throughput
        self.save_interval = save_interval
        
        self.gaze_processor = GazePointProcessor()
        self.person_matcher = PersonMatcher()
        self.validator = DataValidator()
        
        logger.info(f"Initialized processor with paths:")
        logger.info(f"  Annotations: {self.annot_path}")
        logger.info(f"  Base data dir: {self.base_data_dir_path}")
        logger.info(f"  LLaVA results dir: {self.llava_results_dir}")
        logger.info(f"  Number of workers: {self.num_workers}")
        logger.info(f"  Save interval: {self.save_interval} rows")
        if self.resume_from_csv:
            logger.info(f"  Resume from CSV: {self.resume_from_csv}")
    
    def load_annotations(self) -> pd.DataFrame:
        """Load and process annotation file."""
        logger.info("Loading annotations...")
        
        df = pd.read_csv(self.annot_path, sep="\t", header=None)
        df = df[0].str.split(",", expand=True)
        
        # Define column names based on number of columns
        if len(df.columns) == 17:
            df.columns = ['image_path', 'id', 'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 
                         'body_bbox_height', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 
                         'head_bbox_x_min', 'head_bbox_y_min', 'head_bbox_x_max', 'head_bbox_y_max', 
                         'in_or_out', 'meta', 'original_path']
        else:
            df.columns = ['image_path', 'id', 'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 
                         'body_bbox_height', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 
                         'head_bbox_x_min', 'head_bbox_y_min', 'head_bbox_x_max', 'head_bbox_y_max', 
                         'meta', 'original_path']
        
        # Convert numeric columns
        numeric_columns = ['id', 'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height',
                          'eye_x', 'eye_y', 'gaze_x', 'gaze_y',
                          'head_bbox_x_min', 'head_bbox_y_min', 'head_bbox_x_max', 'head_bbox_y_max']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for conversion errors
        nan_counts = df[numeric_columns].isna().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"NaN counts after conversion: {nan_counts[nan_counts > 0].to_dict()}")
        
        logger.info(f"Loaded {len(df)} annotations")
        return df
    
    def aggregate_annotations_by_image(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group annotations by image and average the gaze points."""
        logger.info("Aggregating annotations by image...")
        
        compact_df = df.groupby('image_path').agg({
            'eye_x': 'mean',
            'eye_y': 'mean',
            'gaze_x': 'mean',
            'gaze_y': 'mean',
            'body_bbox_x': 'mean',
            'body_bbox_y': 'mean',
            'body_bbox_width': 'mean',
            'body_bbox_height': 'mean',
        }).reset_index()
        
        compact_df['image_key'] = compact_df['image_path'].apply(lambda x: Path(x).stem)
        
        logger.info(f"Aggregated to {len(compact_df)} unique images")
        return compact_df
    
    def load_previous_results(self) -> Optional[pd.DataFrame]:
        """Load previous results CSV file for resuming processing."""
        if not self.resume_from_csv:
            return None
            
        resume_path = Path(self.resume_from_csv)
        if not resume_path.exists():
            logger.warning(f"Resume CSV file not found: {resume_path}")
            return None
            
        logger.info(f"Loading previous results from: {resume_path}")
        previous_df = pd.read_csv(resume_path)
        
        # Validate required columns exist
        required_columns = ['image_path', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 
                           'body_bbox_x', 'body_bbox_y', 'body_bbox_width', 'body_bbox_height']
        missing_columns = [col for col in required_columns if col not in previous_df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in resume CSV: {missing_columns}")
            return None
            
        # Add image_key if not present
        if 'image_key' not in previous_df.columns:
            previous_df['image_key'] = previous_df['image_path'].apply(lambda x: Path(x).stem)
            
        logger.info(f"Loaded {len(previous_df)} rows from previous results")
        
        # Count already processed rows (rows without error_reason and with gaze_error)
        if 'error_reason' in previous_df.columns and 'gaze_error' in previous_df.columns:
            processed_rows = previous_df[
                (previous_df['error_reason'].isna()) & 
                (previous_df['gaze_error'].notna())
            ]
            logger.info(f"Found {len(processed_rows)} already processed rows")

        return previous_df
    
    def _filter_gaze_points(self, gaze_points_dict: Dict[str, Dict[str, Union[torch.Tensor, np.ndarray]]]) -> Dict[str, Dict[str, Dict]]:
        """Apply outlier filtering to raw gaze points and compute statistics."""
        logger.info("Applying gaze point filtering...")
        processed_gaze_dict: Dict[str, Dict[str, Dict]] = {}

        for image_name, gaze_points in tqdm(gaze_points_dict.items(), desc="Processing gaze points"):
            processed_gaze_dict[image_name] = {}

            for person_id, gaze_point in gaze_points.items():
                if len(gaze_point) == 0:
                    processed_gaze_dict[image_name][person_id] = {
                        'mean_point': np.array([0, 0]),
                        'filtered_points': np.array([]),
                        'original_points': gaze_point
                    }
                    continue

                mean_point, filtered_points = self.gaze_processor.filter_gaze_points(gaze_point)
                processed_gaze_dict[image_name][person_id] = {
                    'mean_point': mean_point,
                    'filtered_points': filtered_points,
                    'original_points': gaze_point
                }

        logger.info(f"Processed gaze points for {len(processed_gaze_dict)} images")
        return processed_gaze_dict

    def load_gaze_points(self) -> Dict[str, Dict[str, Dict]]:
        """Load and process gaze point files using producer-consumer pattern with prefetch queue."""
        logger.info(f"Loading gaze points using {self.num_workers} workers with prefetch queue...")
        
        gaze_points_dict = {}
        prefetch_size = 3000  # Prefetch 3000 files ahead
        processed_count = 0
        start_time = time.time()
        
        # Create a queue for file paths
        file_queue = queue.Queue(maxsize=prefetch_size)
        
        def file_producer():
            """Producer thread that discovers files and adds them to queue."""
            try:
                for gaze_points_path in iter_gaze_point_files(self.llava_results_dir):
                    file_queue.put(gaze_points_path)
                # Signal end of files
                file_queue.put(None)
            except Exception as e:
                logger.error(f"Error in file producer: {e}")
                file_queue.put(None)
        
        # Start producer thread
        producer_thread = Thread(target=file_producer, daemon=True)
        producer_thread.start()
        
        # Consumer loop - process files from queue
        batch = []
        while True:
            # Get next file from queue
            try:
                gaze_points_path = file_queue.get(timeout=30)  # 30 second timeout
            except queue.Empty:
                logger.warning("File queue timeout - no more files found")
                break
                
            # Check for end signal
            if gaze_points_path is None:
                break
                
            batch.append(gaze_points_path)
            
            # Process batch when it's full
            if len(batch) >= self.batch_size:
                batch_start_time = time.time()
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_path = {executor.submit(process_gaze_file, path): path for path in batch}
                    
                    for future in as_completed(future_to_path):
                        result = future.result()
                        if result is not None:
                            image_name, person_id, gaze_point = result
                            if image_name not in gaze_points_dict:
                                gaze_points_dict[image_name] = {}
                            gaze_points_dict[image_name][person_id] = gaze_point
                        processed_count += 1
                
                batch_time = time.time() - batch_start_time
                batch_speed = len(batch) / batch_time if batch_time > 0 else 0
                total_time = time.time() - start_time
                avg_speed = processed_count / total_time if total_time > 0 else 0
                queue_size = file_queue.qsize()
                logger.info(f"Processed {processed_count} gaze point files... (Batch: {batch_speed:.1f} files/sec, Avg: {avg_speed:.1f} files/sec, Queue: {queue_size} files)")
                batch = []  # Reset batch
        
        # Process remaining files in the last batch
        if batch:
            batch_start_time = time.time()
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_path = {executor.submit(process_gaze_file, path): path for path in batch}
                
                for future in as_completed(future_to_path):
                    result = future.result()
                    if result is not None:
                        image_name, person_id, gaze_point = result
                        if image_name not in gaze_points_dict:
                            gaze_points_dict[image_name] = {}
                        gaze_points_dict[image_name][person_id] = gaze_point
                    processed_count += 1
        
        # Wait for producer thread to complete
        producer_thread.join(timeout=5)
        
        total_time = time.time() - start_time
        avg_speed = processed_count / total_time if total_time > 0 else 0
        logger.info(f"Loaded gaze points from {processed_count} files for {len(gaze_points_dict)} images (Overall: {avg_speed:.1f} files/sec)")
        
        # Process gaze points with filtering (can also be parallelized if needed)
        processed_gaze_dict = self._filter_gaze_points(gaze_points_dict)
        return processed_gaze_dict
    
    def load_person_bboxes(self, compact_df: pd.DataFrame) -> Dict[str, Dict]:
        """Load person bounding box data using producer-consumer pattern with prefetch queue."""
        logger.info(f"Loading person bounding boxes using {self.num_workers} workers with prefetch queue...")
        
        person_bboxes = {}
        prefetch_size = 1500  # Prefetch 1500 files ahead (larger for person files)
        processed_count = 0
        start_time = time.time()
        
        # Create a queue for file paths
        file_queue = queue.Queue(maxsize=prefetch_size)
        
        def file_producer():
            """Producer thread that discovers files and adds them to queue."""
            try:
                for person_file in iter_person_files(self.llava_results_dir):
                    file_queue.put((person_file, compact_df, self.base_data_dir_path))
                # Signal end of files
                file_queue.put(None)
            except Exception as e:
                logger.error(f"Error in person file producer: {e}")
                file_queue.put(None)
        
        # Start producer thread
        producer_thread = Thread(target=file_producer, daemon=True)
        producer_thread.start()
        
        # Consumer loop - process files from queue
        batch = []
        while True:
            # Get next file from queue
            try:
                file_info = file_queue.get(timeout=30)  # 30 second timeout
            except queue.Empty:
                logger.warning("Person file queue timeout - no more files found")
                break
                
            # Check for end signal
            if file_info is None:
                break
                
            batch.append(file_info)
            
            # Process batch when it's full
            if len(batch) >= self.batch_size:
                batch_start_time = time.time()
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_info = {executor.submit(process_person_file, info): info for info in batch}
                    
                    for future in as_completed(future_to_info):
                        result = future.result()
                        if result is not None:
                            image_id, person_data = result
                            person_bboxes[image_id] = person_data
                        processed_count += 1
                
                batch_time = time.time() - batch_start_time
                batch_speed = len(batch) / batch_time if batch_time > 0 else 0
                total_time = time.time() - start_time
                avg_speed = processed_count / total_time if total_time > 0 else 0
                queue_size = file_queue.qsize()
                logger.info(f"Processed {processed_count} person files... (Batch: {batch_speed:.1f} files/sec, Avg: {avg_speed:.1f} files/sec, Queue: {queue_size} files)")
                batch = []  # Reset batch
        
        # Process remaining files in the last batch
        if batch:
            batch_start_time = time.time()
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_info = {executor.submit(process_person_file, info): info for info in batch}
                
                for future in as_completed(future_to_info):
                    result = future.result()
                    if result is not None:
                        image_id, person_data = result
                        person_bboxes[image_id] = person_data
                    processed_count += 1
        
        # Wait for producer thread to complete
        producer_thread.join(timeout=5)
        
        total_time = time.time() - start_time
        avg_speed = processed_count / total_time if total_time > 0 else 0
        logger.info(f"Processed {processed_count} person segmentation files (Overall: {avg_speed:.1f} files/sec)")
        logger.info(f"Loaded person data for {len(person_bboxes)} images")
        return person_bboxes

    def load_gaze_and_person_data(self, compact_df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, Dict]], Dict[str, Dict]]:
        """Load gaze points and person segmentation data in a single pass over result directories."""
        logger.info("Loading gaze points and person segmentation data together...")

        raw_gaze_points: Dict[str, Dict[str, Union[torch.Tensor, np.ndarray]]] = {}
        person_bboxes: Dict[str, Dict] = {}
        total_dirs = 0
        total_gaze_files = 0
        total_person_files = 0
        start_time = time.time()
        last_save_time = start_time
        
        # Save paths for intermediate data
        temp_gaze_path = self.llava_results_dir.parent / f"{self.llava_results_dir.name}_gaze_temp.pkl"
        temp_person_path = self.llava_results_dir.parent / f"{self.llava_results_dir.name}_person_temp.pkl"

        def _process_gaze(files):
            for result in map(process_gaze_file, files):
                if result is None:
                    continue
                image_name, person_id, gaze_point = result
                raw_gaze_points.setdefault(image_name, {})[person_id] = gaze_point

        def _process_person(files):
            info_iter = ((person_file, compact_df, self.base_data_dir_path) for person_file in files)
            for result in map(process_person_file, info_iter):
                if result is None:
                    continue
                image_id, person_data = result
                person_bboxes[image_id] = person_data

        use_parallel = self.num_workers > 1
        gaze_executor = None
        person_executor = None

        if use_parallel:
            gaze_executor = ProcessPoolExecutor(max_workers=self.num_workers)
            person_executor = ThreadPoolExecutor(max_workers=self.num_workers)

        for _directory, gaze_files, person_files in iter_gaze_and_person_files(self.llava_results_dir):
            total_dirs += 1

            if total_dirs % 1000 == 0:
                logger.info(
                    "Scanned %d directories (gaze files: %d, person files: %d)",
                    total_dirs,
                    total_gaze_files,
                    total_person_files,
                )
                
                # Periodic save every 250 directories
                current_time = time.time()
                save_duration = current_time - last_save_time
                self._save_intermediate_data(raw_gaze_points, person_bboxes, temp_gaze_path, temp_person_path, total_dirs)
                save_time = time.time() - current_time
                logger.info(f"Periodic data save completed in {save_time:.2f}s (processing time since last save: {save_duration:.2f}s)")
                last_save_time = current_time

            if gaze_files:
                total_gaze_files += len(gaze_files)
                if use_parallel and gaze_executor is not None:
                    for result in gaze_executor.map(process_gaze_file, gaze_files):
                        if result is None:
                            continue
                        image_name, person_id, gaze_point = result
                        raw_gaze_points.setdefault(image_name, {})[person_id] = gaze_point
                else:
                    _process_gaze(gaze_files)

            if person_files:
                total_person_files += len(person_files)
                if use_parallel and person_executor is not None:
                    info_iter = ((person_file, compact_df, self.base_data_dir_path) for person_file in person_files)
                    for result in person_executor.map(process_person_file, info_iter):
                        if result is None:
                            continue
                        image_id, person_data = result
                        person_bboxes[image_id] = person_data
                else:
                    _process_person(person_files)

        if gaze_executor is not None:
            gaze_executor.shutdown(wait=True, cancel_futures=False)
        if person_executor is not None:
            person_executor.shutdown(wait=True, cancel_futures=False)

        elapsed = time.time() - start_time
        logger.info(
            f"Loaded {total_gaze_files} gaze files and {total_person_files} person files "
            f"across {total_dirs} directories in {elapsed:.2f}s"
        )
        
        # Final save before filtering
        if total_dirs > 0 and total_dirs % 250 != 0:
            self._save_intermediate_data(raw_gaze_points, person_bboxes, temp_gaze_path, temp_person_path, total_dirs)
            logger.info("Final intermediate data save before processing")

        processed_gaze_dict = self._filter_gaze_points(raw_gaze_points)
        
        # Clean up temporary files after successful processing
        self._cleanup_temp_files([temp_gaze_path, temp_person_path])
        
        return processed_gaze_dict, person_bboxes
    
    def _save_intermediate_data(self, raw_gaze_points: Dict, person_bboxes: Dict, 
                              gaze_path: Path, person_path: Path, total_dirs: int) -> None:
        """Save intermediate gaze and person data to temporary files."""
        import pickle
        
        try:
            with open(gaze_path, 'wb') as f:
                pickle.dump(raw_gaze_points, f)
            with open(person_path, 'wb') as f:
                pickle.dump(person_bboxes, f)
            logger.info(f"Saved intermediate data (gaze: {len(raw_gaze_points)} images, person: {len(person_bboxes)} images, {total_dirs} dirs processed)")
        except Exception as e:
            logger.warning(f"Failed to save intermediate data: {e}")
    
    def _cleanup_temp_files(self, temp_paths: List[Path]) -> None:
        """Clean up temporary files after successful processing."""
        for temp_path in temp_paths:
            try:
                if temp_path.exists():
                    temp_path.unlink()
                    logger.info(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_path}: {e}")
    
    def process_annotations(self, compact_df: pd.DataFrame, gaze_points_dict: Dict, 
                          person_bboxes: Dict, previous_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Process annotations and add LLaVA results with error tracking.
        
        Args:
            compact_df: DataFrame with aggregated annotations
            gaze_points_dict: Dictionary of gaze point data
            person_bboxes: Dictionary of person bbox data
            previous_df: Optional previous results to resume from
        """
        logger.info("Processing annotations with LLaVA results...")
        
        # If resuming, merge with previous results
        if previous_df is not None:
            logger.info("Merging with previous results for resume functionality...")
            # Merge on image_path to get previous results
            result_df = compact_df.merge(previous_df, on='image_path', how='left', suffixes=('', '_prev'))
            
            # Keep previous results where they exist and are valid
            result_columns = ['llava_matched_person_id', 'llava_person_bbox', 'llava_gaze_points',
                            'gaze_error', 'angular_gaze_error', 'llava_person_bb_iou', 
                            'num_people', 'error_reason']
            
            for col in result_columns:
                if col not in result_df.columns:
                    result_df[col] = None
                    
            # Update columns with previous values where they exist
            for col in result_columns:
                prev_col = f'{col}_prev'
                if prev_col in result_df.columns:
                    # Use previous values where they exist and current is null
                    mask = (result_df[col].isna()) & (result_df[prev_col].notna())
                    result_df.loc[mask, col] = result_df.loc[mask, prev_col]
                    # Drop the temporary previous column
                    result_df.drop(columns=[prev_col], inplace=True)
                    
            # Update coordinate columns from previous results where main columns don't exist
            coord_columns = ['eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'body_bbox_x', 'body_bbox_y', 
                           'body_bbox_width', 'body_bbox_height']
            for col in coord_columns:
                prev_col = f'{col}_prev'
                if prev_col in result_df.columns:
                    # Use previous values where current is null
                    mask = (result_df[col].isna()) & (result_df[prev_col].notna())
                    result_df.loc[mask, col] = result_df.loc[mask, prev_col]
                    result_df.drop(columns=[prev_col], inplace=True)
                    
            # Clean up any remaining _prev columns
            prev_cols = [col for col in result_df.columns if col.endswith('_prev')]
            result_df.drop(columns=prev_cols, inplace=True)
            
        else:
            result_df = compact_df.copy()
            # Add result columns
            result_columns = {
                'llava_matched_person_id': None,
                'llava_person_bbox': None,
                'llava_gaze_points': None,
                'gaze_error': None,
                'angular_gaze_error': None,
                'llava_person_bb_iou': None,
                'num_people': None,
                'error_reason': None
            }
            
            for col, default_val in result_columns.items():
                if col not in result_df.columns:
                    result_df[col] = default_val
        
        # Reset index for iterating
        result_df = result_df.reset_index(drop=True)
        
        processed_count = 0
        skipped_count = 0
        error_counts = {}
        last_save_time = time.time()
        
        for ind, row in tqdm(result_df.iterrows(), total=len(result_df), desc="Processing rows"):
            # Skip if already processed (has gaze_error and no error_reason)
            if (pd.notna(row.get('gaze_error')) and pd.isna(row.get('error_reason'))) or \
               (pd.notna(row.get('error_reason'))):
                skipped_count += 1
                if pd.notna(row.get('gaze_error')):
                    processed_count += 1
                continue
            # Validate annotation row
            validation_error = self.validator.validate_annotation_row(row)
            if validation_error:
                result_df.at[ind, 'error_reason'] = validation_error
                error_counts[validation_error] = error_counts.get(validation_error, 0) + 1
                continue
            
            image_path = row['image_path']
            image_id = Path(image_path).stem
            
            # Validate image exists
            image_validation_error = self.validator.validate_image_path(image_path, self.base_data_dir_path)
            if image_validation_error:
                result_df.at[ind, 'error_reason'] = image_validation_error
                error_counts[image_validation_error] = error_counts.get(image_validation_error, 0) + 1
                continue
            
            # Get LLaVA results
            llava_gaze_result = gaze_points_dict.get(image_id)
            llava_person_result = person_bboxes.get(image_id)
            
            if llava_gaze_result is None:
                error_msg = f"No LLaVA gaze results for image {image_id}"
                result_df.at[ind, 'error_reason'] = error_msg
                error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
                continue
                
            if llava_person_result is None:
                error_msg = f"No LLaVA person results for image {image_id}"
                result_df.at[ind, 'error_reason'] = error_msg
                error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
                continue
            
            # Get image dimensions
            if 'image_shape' not in llava_person_result:
                error_msg = f"No image shape data for {image_id}"
                result_df.at[ind, 'error_reason'] = error_msg
                error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
                continue
                
            h, w = llava_person_result['image_shape']
            
            # Create ground truth person bbox
            gt_person_bbox = [
                w * row['body_bbox_x'], 
                h * row['body_bbox_y'],
                w * (row['body_bbox_x'] + row['body_bbox_width']), 
                h * (row['body_bbox_y'] + row['body_bbox_height'])
            ]
            
            # Find matching person
            matched_person_id, person_bb_iou = self.person_matcher.find_best_matching_person(
                llava_person_result, gt_person_bbox)
            
            if matched_person_id is None:
                error_msg = f"No matching person found for {image_id}"
                result_df.at[ind, 'error_reason'] = error_msg
                error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
                continue
            
            # Validate gaze data (but continue processing even if validation fails)
            gaze_validation_error = self.validator.validate_gaze_data(llava_gaze_result, matched_person_id)
            if gaze_validation_error:
                result_df.at[ind, 'error_reason'] = gaze_validation_error
                error_counts[gaze_validation_error] = error_counts.get(gaze_validation_error, 0) + 1
                # Continue processing with null values instead of skipping
                logger.debug(f"Gaze validation failed for {image_id}, person {matched_person_id}: {gaze_validation_error}")
            
            # Extract data
            llava_person_bbox = llava_person_result[matched_person_id]['person']['boxes'][0]
            
            # Handle missing or invalid gaze data
            if gaze_validation_error:
                # Set default/null values when gaze data is invalid
                llava_gaze_point_data = {'mean_point': np.array([0, 0])}
                llava_gaze_point = np.array([0, 0])
            else:
                llava_gaze_point_data = llava_gaze_result[matched_person_id]
                llava_gaze_point = llava_gaze_point_data['mean_point']
            
            # Calculate number of people
            num_people = len([key for key, val_dd in llava_person_result.items() 
                            if isinstance(val_dd, dict) and 'person' in val_dd])
            
            # Update row with results
            result_df.at[ind, 'llava_matched_person_id'] = matched_person_id
            result_df.at[ind, 'llava_person_bbox'] = llava_person_bbox
            result_df.at[ind, 'llava_gaze_points'] = llava_gaze_point
            result_df.at[ind, 'llava_person_bb_iou'] = person_bb_iou
            result_df.at[ind, 'num_people'] = num_people
            
            # Calculate errors (handle invalid gaze data)
            gt_gaze_point = np.array([row['gaze_x'], row['gaze_y']])
            
            if gaze_validation_error:
                # Set null/NaN values for errors when gaze data is invalid
                result_df.at[ind, 'gaze_error'] = np.nan
                result_df.at[ind, 'angular_gaze_error'] = np.nan
            else:
                llava_gaze_normalized = np.array(llava_gaze_point) / np.array([w, h])
                
                # Euclidean distance
                euclidean_distance = np.linalg.norm(gt_gaze_point - llava_gaze_normalized)
                result_df.at[ind, 'gaze_error'] = euclidean_distance
                
                # Angular distance
                eye_pos = np.array([row['eye_x'], row['eye_y']])
                angular_distance = self.gaze_processor.calculate_angular_distance(
                    llava_gaze_normalized, gt_gaze_point, eye_pos)
                result_df.at[ind, 'angular_gaze_error'] = angular_distance
            
            processed_count += 1
            
            # Periodic temporary save
            if processed_count % self.save_interval == 0:
                current_time = time.time()
                save_duration = current_time - last_save_time
                self.save_temp_results(result_df, processed_count)
                save_time = time.time() - current_time
                logger.info(f"Periodic save completed in {save_time:.2f}s (processing time since last save: {save_duration:.2f}s)")
                last_save_time = current_time
        
        # Final temporary save before completion
        if processed_count > 0 and processed_count % self.save_interval != 0:
            self.save_temp_results(result_df, processed_count)
            logger.info("Final temporary save before completion")
        
        logger.info(f"Successfully processed {processed_count}/{len(result_df)} rows")
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} already processed rows")
        
        if error_counts:
            logger.info("Error summary:")
            for error_msg, count in error_counts.items():
                logger.info(f"  {error_msg}: {count} rows")
        
        # Calculate and log metrics for successfully processed rows
        successful_rows = result_df[result_df['error_reason'].isna()]
        if len(successful_rows) > 0:
            avg_euclidean_error = successful_rows['gaze_error'].mean()
            avg_angular_error = successful_rows['angular_gaze_error'].mean()
            logger.info(f"Average euclidean gaze error: {avg_euclidean_error:.4f}")
            logger.info(f"Average angular gaze error: {avg_angular_error:.4f} degrees")
        
        return result_df
    
    def save_temp_results(self, df: pd.DataFrame, processed_count: int) -> None:
        """Save temporary results to CSV file during processing."""
        df.to_csv(self.temp_results_path, index=False)
        logger.info(f"Saved temporary results ({processed_count} processed rows) to {self.temp_results_path}")
    
    def save_results(self, df: pd.DataFrame) -> None:
        """Save final results to CSV file."""
        df.to_csv(self.results_path, index=False)
        logger.info(f"Saved final results to {self.results_path}")
        
        # Clean up temporary file if it exists
        if self.temp_results_path.exists():
            self.temp_results_path.unlink()
            logger.info(f"Cleaned up temporary file: {self.temp_results_path}")
    
    def process(self) -> pd.DataFrame:
        """Run the complete processing pipeline."""
        logger.info("Starting GazeFollow dataset processing...")
        
        # Load previous results if resuming
        previous_df = self.load_previous_results()
        
        # Load and process annotations
        df = self.load_annotations()
        compact_df = self.aggregate_annotations_by_image(df)
        
        # Load LLaVA results
        gaze_points_dict, person_bboxes = self.load_gaze_and_person_data(compact_df)
        
        # Process and match data
        result_df = self.process_annotations(compact_df, gaze_points_dict, person_bboxes, previous_df)
        
        # Save results
        self.save_results(result_df)
        
        logger.info("Processing completed successfully!")
        return result_df


def main():
    """Main entry point for the script."""
    # Configuration
    annot_path = r"D:\Projects\data\gazefollow\train_annotations_release.txt"
    base_data_dir_path = r"D:\Projects\data\gazefollow"
    llava_results_dir = r"D:\Projects\data\gazefollow\results\valid_runs\combined_ppl_desc"
    
    # Optional: Resume from previous CSV file (convert to WSL format)
    resume_from_csv = r"D:\Projects\data\gazefollow\results\valid_runs\combined_ppl_desc_results.csv"
    # Set to None to start fresh processing
    # resume_from_csv = None
    
    # Parallel processing configuration
    num_workers = 12  # None for auto-detect, or specify number like 4, 8, etc.
    
    # Temporary save configuration
    save_interval = 1000  # Save every 1000 processed rows
    
    # Create processor and run
    processor = GazeFollowDatasetProcessor(
        annot_path=annot_path,
        base_data_dir_path=base_data_dir_path,
        llava_results_dir=llava_results_dir,
        resume_from_csv=resume_from_csv,
        num_workers=num_workers,
        save_interval=save_interval
    )
    
    results = processor.process()
    return results


if __name__ == "__main__":
    main()
