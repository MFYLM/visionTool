import numpy as np
import os
from typing import Dict, List
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm
from typing import Optional, List


BRIDGE_V2_DIR = "/root"


class BridgeDatasetLoader:
    """Class to load and process images from Bridge dataset"""

    def __init__(self, data_dir: str):
        """
        Initialize the Bridge dataset loader

        Args:
            data_dir: Path to the dataset directory
        """
        self.data_dir = data_dir
        self.dataset_name = "bridge_orig"

    def load_dataset(self, num_episodes: Optional[int] = None):
        """
        Load the Bridge dataset

        Args:
            num_episodes: Number of episodes to load (None for all)

        Returns:
            The loaded dataset
        """
        split = f"train[:{num_episodes}]" if num_episodes else "train"
        return tfds.load(self.dataset_name, data_dir=self.data_dir, split=split)

    def load_images(
        self,
        episode_indices: Optional[List[int]] = None,
        step_indices: Optional[List[int]] = None,
        image_indices: Optional[List[int]] = None,
        max_episodes: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> list[tuple[np.ndarray, int, int, int]]:
        """
        Load images from Bridge dataset with flexible selection options

        Args:
            episode_indices: List of specific episode indices to load (None for all up to max_episodes)
            step_indices: List of specific step indices to load (None for all up to max_steps)
            image_indices: List of image indices to load (e.g., [0, 1, 2] for image_0, image_1, image_2)
            max_episodes: Maximum number of episodes to load if episode_indices is None
            max_steps: Maximum number of steps per episode to load if step_indices is None

        Returns:
            List of tuples (image_array, episode_idx, step_idx, image_idx)
        """
        # Default to image_0 if not specified
        if image_indices is None:
            image_indices = [0]

        # Load dataset
        raw_dataset = self.load_dataset(max_episodes)

        image_data = []
        skipped_images = []

        # Convert to list to get length for tqdm and create progress bar
        episodes = list(raw_dataset)

        for episode_idx, episode in tqdm(
            enumerate(episodes), total=len(episodes), desc="Processing episodes"
        ):
            # Skip if not in specified episode indices
            if episode_indices is not None and episode_idx not in episode_indices:
                continue

            for step_idx, step in enumerate(episode["steps"].as_numpy_iterator()):
                if max_steps is not None and max_steps > 0:
                    # Skip if not in specified step indices or exceeds max_steps
                    if (step_indices is not None and step_idx not in step_indices) or (
                        max_steps is not None and step_idx >= max_steps
                    ):
                        continue
                else:
                    # Extract images based on specified indices
                    for image_idx in image_indices:
                        image_key = f"image_{image_idx}"
                        if image_key in step["observation"]:
                            image: np.ndarray = step["observation"][image_key]
                            # Skip if image is empty (all zeros)
                            if np.all(image == 0):
                                skipped_images.append((
                                    episode_idx,
                                    step_idx,
                                    image_idx,
                                ))
                                continue

                            image_data.append((image, episode_idx, step_idx, image_idx))
                        else:
                            skipped_images.append((episode_idx, step_idx, image_idx))

        # Print summary of skipped images
        if skipped_images:
            print(
                f"\nSkipped {len(skipped_images)} images that were empty or not found:"
            )
            for ep_idx, st_idx, img_idx in skipped_images[:10]:  # Show first 10 skipped
                print(f"  Episode {ep_idx}, Step {st_idx}, Image {img_idx}")
            if len(skipped_images) > 10:
                print(f"  ... and {len(skipped_images) - 10} more")

        return image_data

    def get_image_arrays(
        self,
        episode_indices: Optional[List[int]] = None,
        step_indices: Optional[List[int]] = None,
        image_indices: Optional[List[int]] = None,
        max_episodes: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> list[np.ndarray]:
        """
        Get only the image arrays from the dataset

        Args:
            Same as load_images method

        Returns:
            List of image arrays
        """
        image_data = self.load_images(
            episode_indices, step_indices, image_indices, max_episodes, max_steps
        )
        return np.array([item[0] for item in image_data])



if __name__ == "__main__":
    bridge_dataset = BridgeDatasetLoader(data_dir=BRIDGE_V2_DIR)
    image_arrays = bridge_dataset.get_image_arrays(max_episodes=1)
    np.save("/root/visionTool/process_data/sample_images.npy", image_arrays)    