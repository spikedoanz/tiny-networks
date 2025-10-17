import json
import numpy as np

COLORS = {
    0: "\033[48;5;16m  \033[0m",  # Black
    1: "\033[48;5;21m  \033[0m",  # Blue
    2: "\033[48;5;196m  \033[0m",  # Red
    3: "\033[48;5;46m  \033[0m",  # Green
    4: "\033[48;5;226m  \033[0m",  # Yellow
    5: "\033[48;5;244m  \033[0m",  # Gray
    6: "\033[48;5;201m  \033[0m",  # Magenta
    7: "\033[48;5;208m  \033[0m",  # Orange
    8: "\033[48;5;51m  \033[0m",  # Light blue
    9: "\033[48;5;88m  \033[0m",  # Brown
}

PAD_TOKEN = 10
SEED = 1337

def display_grid(grid: np.ndarray, title: str = "", raw: bool = False) -> None:
    """Display a grid with colored blocks or as raw array."""
    print(f"\n{title}")
    if raw:
        print(grid)
    else:
        print("-" * (grid.shape[1] * 2))
        for row in grid:
            print("".join(COLORS.get(cell, "  ") for cell in row))

# Data augmentation functions
def rotate_90(grid: np.ndarray, k: int = 1) -> np.ndarray:
    """Rotate grid by 90*k degrees"""
    return np.rot90(grid, k)

def flip_horizontal(grid: np.ndarray) -> np.ndarray:
    return np.flip(grid, axis=1)

def flip_vertical(grid: np.ndarray) -> np.ndarray:
    return np.flip(grid, axis=0)

def permute_colors(grid: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Randomly permute color labels"""
    np.random.seed(seed)
    unique_colors = np.unique(grid)
    # Keep padding token unchanged
    unique_colors = unique_colors[unique_colors != PAD_TOKEN]
    
    permutation = np.random.permutation(unique_colors)
    mapping = {old: new for old, new in zip(unique_colors, permutation)}
    mapping[PAD_TOKEN] = PAD_TOKEN  # Don't change padding
    
    result = grid.copy()
    for old, new in mapping.items():
        result[grid == old] = new
    return result

def pad_to_30x30(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Pad grid to 30x30 and return grid, mask, and original shape"""
    original_shape = grid.shape
    padded = np.full((30, 30), PAD_TOKEN, dtype=np.int32)
    h, w = grid.shape
    padded[:h, :w] = grid
    
    # Create mask (1 for valid, 0 for padding)
    mask = (padded != PAD_TOKEN).astype(np.float32)
    
    return padded, mask, original_shape

def augment_sample(x: np.ndarray, y: np.ndarray, num_augmentations: int = 10):
    """Generate augmented versions of a sample"""
    augmented_x, augmented_y, masks_x, masks_y = [], [], [], []
    
    for i in range(num_augmentations):
        # Apply same transforms to both input and output
        aug_x, aug_y = x.copy(), y.copy()
        
        # Random rotation
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            aug_x = rotate_90(aug_x, k)
            aug_y = rotate_90(aug_y, k)
        
        # Random flip
        if np.random.rand() > 0.5:
            if np.random.rand() > 0.5:
                aug_x = flip_horizontal(aug_x)
                aug_y = flip_horizontal(aug_y)
            else:
                aug_x = flip_vertical(aug_x)
                aug_y = flip_vertical(aug_y)
        
        # Color permutation
        if np.random.rand() > 0.5:
            seed = np.random.randint(1000000)
            aug_x = permute_colors(aug_x, seed)
            aug_y = permute_colors(aug_y, seed)
        
        # Pad to 30x30
        padded_x, mask_x, _ = pad_to_30x30(aug_x)
        padded_y, mask_y, _ = pad_to_30x30(aug_y)
        
        augmented_x.append(padded_x.flatten())
        augmented_y.append(padded_y.flatten())
        masks_x.append(mask_x.flatten())
        masks_y.append(mask_y.flatten())
    
    return np.array(augmented_x), np.array(augmented_y), np.array(masks_x), np.array(masks_y)

def arc_agi(json_files, num_augmentations=10):
    """Load data and apply augmentation"""
    all_x, all_y, all_masks_x, all_masks_y = [], [], [], []
    
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
        
        # Process training examples
        for example in data.get("train", []):
            x = np.array(example["input"])
            y = np.array(example["output"])
            
            aug_x, aug_y, mask_x, mask_y = augment_sample(x, y, num_augmentations)
            all_x.append(aug_x)
            all_y.append(aug_y)
            all_masks_x.append(mask_x)
            all_masks_y.append(mask_y)
    
    # Stack all augmented samples
    if all_x:
        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        all_masks_x = np.concatenate(all_masks_x, axis=0)
        all_masks_y = np.concatenate(all_masks_y, axis=0)
    else:
        all_x = np.array([])
        all_y = np.array([])
        all_masks_x = np.array([])
        all_masks_y = np.array([])
    
    return all_x, all_y, all_masks_x, all_masks_y
