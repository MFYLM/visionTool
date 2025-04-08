import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from typing import Dict, List, Tuple, Optional, Union
import re


class OwlV2SAM:
    def __init__(
        self,
        sam_checkpoint: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the detector with OWLv2 and SAM models.

        Args:
            sam_checkpoint: Path to SAM model checkpoint
            device: Device to run models on ("cuda" or "cpu")
        """
        self.device = device

        # Initialize SAM
        print(f"Loading SAM model from {sam_checkpoint}...")
        
        model_type =re.findall(r"vit_h|vit_l|vit_b", sam_checkpoint)
        print(f"Model type: {model_type}")
        if model_type is None or len(model_type) == 0:
            raise ValueError("Model type not recognized")
        else:
            model_type = model_type[0]
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)

        # Initialize OWL-ViT
        print("Loading OWLv2 model...")
        self.owlv2_processor = Owlv2Processor.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )
        self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )
        self.owlv2_model.to(self.device)
        print("Models loaded successfully!")

    def detect_and_segment(
        self,
        image: Union[np.ndarray, str],
        text_prompts: List[str],
        detection_threshold: float = 0.1,
        return_all_detections: bool = False,
    ) -> Dict:
        """Detect objects with OWLv2 and segment with SAM.

        Args:
            image: Input image (numpy array in RGB format) or path to image
            text_prompts: List of text prompts for detection
            detection_threshold: Confidence threshold for detection
            return_all_detections: If True, return all detections; otherwise, return best detection

        Returns:
            Dictionary containing detection and segmentation results

            "detected": True if objects are detected, False otherwise
            "detections": List of detections if return_all_detections is True, otherwise None
            "box": Bounding box of the detected object if return_all_detections is False, otherwise None, [x1, y1, x2, y2]
            "score": Score of the detected object
            "label": Label of the detected object
            "text_prompt": Text prompt of the detected object
            "mask": Mask of the detected object, shape: (H, W)
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(image)

        # Detect with OWLv2
        inputs = self.owlv2_processor(
            text=text_prompts, images=pil_image, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.owlv2_model(**inputs)

        target_sizes = torch.Tensor([pil_image.size[::-1]]).to(self.device)
        results = self.owlv2_processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=detection_threshold
        )[0]

        if len(results["boxes"]) == 0:
            return {
                "detected": False,
                "message": "No objects detected",
                "image": image,
                "text_prompts": text_prompts,
            }

        # Prepare SAM for segmentation
        self.sam_predictor.set_image(image)

        if return_all_detections:
            # Process all detections
            detections = []
            for i, (box, score, label) in enumerate(
                zip(results["boxes"], results["scores"], results["labels"])
            ):
                box_np = box.detach().cpu().numpy()
                score_np = score.item()
                label_idx = label.item()
                text_prompt = text_prompts[label_idx % len(text_prompts)]

                # Generate mask with SAM
                masks, _, _ = self.sam_predictor.predict(
                    box=box_np, multimask_output=False
                )

                detections.append({
                    "box": box_np,
                    "score": score_np,
                    "label": label_idx,
                    "text_prompt": text_prompt,
                    "mask": masks[0],
                })

            return {
                "detected": True,
                "detections": detections,
                "image": image,
                "text_prompts": text_prompts,
            }
        else:
            # Get best detection
            best_idx = torch.argmax(results["scores"])
            best_box = results["boxes"][best_idx].detach().cpu().numpy()
            best_score = results["scores"][best_idx].item()
            best_label = results["labels"][best_idx].item()
            text_prompt = text_prompts[best_label % len(text_prompts)]

            # Generate mask with SAM
            masks, _, _ = self.sam_predictor.predict(
                box=best_box, multimask_output=False
            )

            return {
                "detected": True,
                "box": best_box,
                "score": best_score,
                "label": best_label,
                "text_prompt": text_prompt,
                "mask": masks[0],
                "image": image,
                "text_prompts": text_prompts,
            }

    def segment_with_points(
        self,
        image: Union[np.ndarray, str],
        points: List[List[int]],  # List of [x, y] coordinates
        point_labels: List[int],  # 1 for foreground, 0 for background
        multimask_output: bool = True,
    ) -> Dict:
        """Segment objects with SAM using point prompts.

        Args:
            image: Input image (numpy array in RGB format) or path to image
            points: List of point coordinates [[x1, y1], [x2, y2], ...]
            point_labels: List of point labels (1 for foreground, 0 for background)
            multimask_output: If True, return multiple masks; otherwise, return single best mask

        Returns:
            Dictionary containing segmentation results

            "masks": Array of predicted masks, shape (N, H, W)
            "scores": Array of mask confidence scores
            "points": Input points
            "point_labels": Input point labels
            "image": Original image
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert points and labels to numpy arrays if they aren't already
        points_np = np.array(points)
        point_labels_np = np.array(point_labels)

        # Prepare SAM for segmentation
        self.sam_predictor.set_image(image)

        # Generate masks with SAM
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=points_np,
            point_labels=point_labels_np,
            multimask_output=multimask_output,
        )

        return {
            "masks": masks,
            "scores": scores,
            "logits": logits,
            "points": points_np,
            "point_labels": point_labels_np,
            "image": image,
        }

    def visualize(
        self,
        results: Dict,
        show_box: bool = True,
        show_mask: bool = True,
        show_points: bool = True,
        mask_idx: int = None,  # Index of mask to show when multiple masks are available
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Visualize detection and segmentation results.

        Args:
            results: Results from detect_and_segment or segment_with_points
            show_box: Whether to visualize bounding boxes (for detection results)
            show_mask: Whether to visualize masks
            show_points: Whether to visualize points (for point-based segmentation)
            mask_idx: Index of mask to show when multiple masks are available
            save_path: Path to save visualization image

        Returns:
            Visualization image
        """
        # Check if it's detection results or point-based segmentation results
        is_detection = "detected" in results
        is_point_based = "points" in results and "point_labels" in results

        if is_detection and not results["detected"]:
            print(results["message"])
            return results["image"]

        # Create a copy of the image for visualization
        vis_image = results["image"].copy()

        # Color for masks and boxes
        mask_color = np.array([0, 255, 0], dtype=np.uint8)  # Green
        box_color = (0, 255, 0)  # Green in BGR
        font = cv2.FONT_HERSHEY_SIMPLEX

        if is_detection:
            # Handle detection-based results
            if "detections" in results:  # Multiple detections
                for i, det in enumerate(results["detections"]):
                    if show_mask and "mask" in det:
                        # Apply mask
                        mask = det["mask"]
                        vis_image = self._apply_mask(
                            vis_image, mask, mask_color, alpha=0.5
                        )

                    if show_box and "box" in det:
                        # Draw box
                        box = det["box"].astype(int)
                        cv2.rectangle(
                            vis_image, (box[0], box[1]), (box[2], box[3]), box_color, 2
                        )

                        # Add label
                        label = f"{det['text_prompt']}: {det['score']:.2f}"
                        cv2.putText(
                            vis_image,
                            label,
                            (box[0], box[1] - 10),
                            font,
                            0.5,
                            box_color,
                            1,
                        )
            else:  # Single detection
                if show_mask and "mask" in results:
                    # Apply mask
                    mask = results["mask"]
                    vis_image = self._apply_mask(vis_image, mask, mask_color, alpha=0.5)

                if show_box and "box" in results:
                    # Draw box
                    box = results["box"].astype(int)
                    cv2.rectangle(
                        vis_image, (box[0], box[1]), (box[2], box[3]), box_color, 2
                    )

                    # Add label
                    label = f"{results['text_prompt']}: {results['score']:.2f}"
                    cv2.putText(
                        vis_image, label, (box[0], box[1] - 10), font, 0.5, box_color, 1
                    )

        elif is_point_based:
            # Handle point-based segmentation results
            if show_mask and "masks" in results:
                masks = results["masks"]
                scores = results["scores"]

                # Determine which mask to show
                if mask_idx is None:
                    # If not specified, show the highest-scoring mask
                    mask_idx = np.argmax(scores) if len(scores) > 0 else 0

                if 0 <= mask_idx < len(masks):
                    mask = masks[mask_idx]
                    vis_image = self._apply_mask(vis_image, mask, mask_color, alpha=0.5)

                    # Add score if available
                    if len(scores) > mask_idx:
                        score = scores[mask_idx]
                        cv2.putText(
                            vis_image,
                            f"Score: {score:.3f}",
                            (10, 30),
                            font,
                            1,
                            (255, 255, 255),
                            2,
                        )

            if show_points and "points" in results and "point_labels" in results:
                points = results["points"]
                point_labels = results["point_labels"]
                vis_image = self._show_points(vis_image, points, point_labels)

        # Save if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_path = os.path.join(save_path, "det_seg.png")
            cv2.imwrite(save_path, vis_image)
            print(f"Visualization saved to {save_path}")

        return vis_image

    def _show_points(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        marker_size: int = 20,
    ) -> np.ndarray:
        """Draw points on image.

        Args:
            image: Input image
            points: Point coordinates [[x1, y1], [x2, y2], ...]
            labels: Point labels (1 for foreground, 0 for background)
            marker_size: Size of the point markers

        Returns:
            Image with points drawn
        """
        vis_image = image.copy()

        # Colors for foreground (green) and background (red) points
        fg_color = (0, 255, 0)  # Green in BGR
        bg_color = (0, 0, 255)  # Red in BGR

        for point, label in zip(points, labels):
            # Get x, y coordinates (make sure they're integers)
            x, y = int(point[0]), int(point[1])

            # Choose color based on label
            color = fg_color if label == 1 else bg_color

            # Draw a filled circle
            cv2.circle(vis_image, (x, y), marker_size // 2, color, -1)

            # Draw a star-like pattern for better visibility
            cv2.drawMarker(
                vis_image,
                (x, y),
                color,
                markerType=cv2.MARKER_STAR,
                markerSize=marker_size,
                thickness=2,
            )

        return vis_image

    def _apply_mask(
        self, image: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """Apply colored mask to image.

        Args:
            image: Input image
            mask: Binary mask
            color: Color for mask
            alpha: Transparency of mask

        Returns:
            Image with mask applied
        """
        mask = mask.astype(bool)
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color

        # Blend the mask with the image
        return cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    def display_image(self, image: np.ndarray, window_name: str = "Image") -> None:
        """Display image in a window.

        Args:
            image: Image to display
            window_name: Name of the window
        """
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    import argparse
    import os
    
    checkpoint_path = "/root/visionTool/detect_segmentation/sam_vit_b_01ec64.pth"

    parser = argparse.ArgumentParser(
        description="Detect and segment objects in an image"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/example/image.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/checkpoints/sam_vit_b_01ec64.pth",
        help="Path to SAM model checkpoint",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detect", "point"],
        default="detect",
        help="Segmentation mode: detection-based or point-based",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["bowl on the table"],
        help="Text prompts for detection mode",
    )
    parser.add_argument(
        "--points",
        type=int,
        nargs="+",
        default=[],
        help="Point coordinates for point-based mode [x1 y1 x2 y2 ...]",
    )
    parser.add_argument(
        "--point_labels",
        type=int,
        nargs="+",
        default=[],
        help="Point labels for point-based mode (1=foreground, 0=background)",
    )
    parser.add_argument(
        "--multimask",
        action="store_true",
        help="Return multiple masks for point-based segmentation",
    )
    parser.add_argument(
        "--mask_idx",
        type=int,
        default=None,
        help="Index of mask to visualize when multiple masks are available",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="Detection confidence threshold"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/results",
        help="Path to save visualization",
    )
    parser.add_argument("--show_box", action="store_true", help="Show bounding boxes")
    parser.add_argument(
        "--show_mask", action="store_true", help="Show segmentation masks"
    )
    parser.add_argument("--show_points", action="store_true", help="Show input points")
    parser.add_argument(
        "--all_detections",
        action="store_true",
        help="Return all detections instead of best",
    )
    parser.add_argument(
        "--save-mask-path", 
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/mask.png",
        help="Path to save segmentation mask",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the results",
    )

    args = parser.parse_args()

    # Set defaults for visualization if none specified
    if not args.show_box and not args.show_mask and not args.show_points:
        args.show_box = True
        args.show_mask = True
        args.show_points = True

    # Initialize detector
    detector = OwlV2SAM(sam_checkpoint=args.sam_checkpoint)
    
    images = np.load("/root/visionTool/pose_estimation/sample_images.npy")

    # Process based on mode
    if args.mode == "detect":
        # Detect and segment
        results = detector.detect_and_segment(
            image=images[0],
            text_prompts=args.prompts,
            detection_threshold=args.threshold,
            return_all_detections=args.all_detections,
        )

        # Print detection results
        if results["detected"]:
            if "detections" in results:
                print(f"Found {len(results['detections'])} objects")
                for i, det in enumerate(results["detections"]):
                    print(
                        f"Detection {i + 1}: {det['text_prompt']} (score: {det['score']:.3f})"
                    )
            else:
                print(
                    f"Detected {results['text_prompt']} with confidence {results['score']:.3f}"
                )
                # save mask
                mask_image = (results["mask"] * 255).astype(np.uint8)
                gray_img = Image.fromarray(mask_image, mode="L")
                gray_img.save(args.save_mask_path)
        else:
            print("No objects detected")

    elif args.mode == "point":
        # Check if points are provided
        if not args.points:
            print("Error: No points provided for point-based segmentation.")
            exit(1)

        # Reshape points from flat list [x1, y1, x2, y2, ...] to [[x1, y1], [x2, y2], ...]
        point_coords = []
        for i in range(0, len(args.points), 2):
            if i + 1 < len(args.points):
                point_coords.append([args.points[i], args.points[i + 1]])

        # If point labels not provided, default to all foreground (1)
        if not args.point_labels:
            point_labels = [1] * len(point_coords)
        else:
            point_labels = args.point_labels

        if len(point_labels) != len(point_coords):
            print("Error: Number of point labels must match number of points.")
            exit(1)

        # Segment with points
        results = detector.segment_with_points(
            image=args.image,
            points=point_coords,
            point_labels=point_labels,
            multimask_output=args.multimask,
        )

        # Print results
        num_masks = len(results["masks"])
        print(f"Generated {num_masks} mask{'s' if num_masks != 1 else ''}")
        for i, score in enumerate(results["scores"]):
            print(f"Mask {i + 1} score: {score:.3f}")

    # Visualize results if requested
    if args.visualize:
        vis_image = detector.visualize(
            results=results,
            show_box=args.show_box,
            show_mask=args.show_mask,
            show_points=args.show_points,
            mask_idx=args.mask_idx,
            save_path=args.save_path,
        )

        # Display image
        detector.display_image(vis_image)
