from PIL import Image, ImageEnhance
import random
import os
from pathlib import Path
import shutil

class SpriteDatasetAugmenter:
    def __init__(self, sprite_dir, background_dir, output_dir, class_mapping=None):
        """
        Args:
            sprite_dir: Directory containing sprite images (can have subfolders for different classes)
            background_dir: Directory containing background images
            output_dir: Output directory for augmented dataset
            class_mapping: Dict mapping sprite folder names to class IDs (optional)
        """
        self.sprite_dir = Path(sprite_dir)
        self.background_dir = Path(background_dir)
        self.output_dir = Path(output_dir)
        self.class_mapping = class_mapping or {}
        
        # Create output structure
        self.train_img_dir = self.output_dir / 'images' / 'train'
        self.train_lbl_dir = self.output_dir / 'labels' / 'train'
        self.val_img_dir = self.output_dir / 'images' / 'val'
        self.val_lbl_dir = self.output_dir / 'labels' / 'val'
        
        for dir_path in [self.train_img_dir, self.train_lbl_dir, 
                         self.val_img_dir, self.val_lbl_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_sprites(self):
        """Load all sprites and organize by class"""
        sprites = {}
        
        # Check if sprites are organized in subfolders (multi-class)
        subdirs = [d for d in self.sprite_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            # Multi-class: each subfolder is a class
            for class_idx, subdir in enumerate(sorted(subdirs)):
                class_name = subdir.name
                class_id = self.class_mapping.get(class_name, class_idx)
                sprite_files = list(subdir.glob('*.png')) + list(subdir.glob('*.jpg'))
                
                if sprite_files:
                    sprites[class_id] = {
                        'name': class_name,
                        'files': sprite_files
                    }
                    print(f"Class {class_id} ({class_name}): {len(sprite_files)} sprites")
        else:
            # Single class: all sprites in one folder
            sprite_files = list(self.sprite_dir.glob('*.png')) + list(self.sprite_dir.glob('*.jpg'))
            sprites[0] = {
                'name': 'sprite',
                'files': sprite_files
            }
            print(f"Single class: {len(sprite_files)} sprites")
        
        return sprites
    
    def load_backgrounds(self):
        """Load all background images"""
        backgrounds = (list(self.background_dir.glob('*.png')) + 
                      list(self.background_dir.glob('*.jpg')) +
                      list(self.background_dir.glob('*.jpeg')))
        print(f"Loaded {len(backgrounds)} background images")
        return backgrounds
    
    def augment_sprite(self, sprite):
        """Apply random augmentations to sprite"""
        # Random brightness
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(sprite)
            sprite = enhancer.enhance(random.uniform(0.7, 1.3))
        
        # Random contrast
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(sprite)
            sprite = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Random flip
        if random.random() > 0.5:
            sprite = sprite.transpose(Image.FLIP_LEFT_RIGHT)
        
        return sprite
    
    def create_augmented_image(self, sprite_path, background_path, class_id, 
                               img_size=640, num_sprites=None):
        """
        Create one augmented image with sprite(s) on background
        
        Args:
            sprite_path: Path to sprite image
            background_path: Path to background image
            class_id: Class ID for the sprite
            img_size: Output image size (square)
            num_sprites: Number of sprites to place (1-3 if None, random)
        """
        # Load and resize background
        background = Image.open(background_path).convert('RGBA')
        background = background.resize((img_size, img_size), Image.LANCZOS)
        
        # Load sprite
        sprite = Image.open(sprite_path).convert('RGBA')
        
        # Determine number of sprites to place
        if num_sprites is None:
            num_sprites = random.randint(1, 3)
        
        labels = []
        
        for _ in range(num_sprites):
            # Apply augmentations
            aug_sprite = self.augment_sprite(sprite.copy())
            
            # Random scale (30% to 120% of original size)
            scale = random.uniform(0.4, 0.6)
            new_w = int(aug_sprite.width * scale)
            new_h = int(aug_sprite.height * scale)
            
            # Ensure sprite isn't too large
            max_size = int(img_size * 0.8)
            if new_w > max_size or new_h > max_size:
                scale_factor = max_size / max(new_w, new_h)
                new_w = int(new_w * scale_factor)
                new_h = int(new_h * scale_factor)
            
            scaled_sprite = aug_sprite.resize((new_w, new_h), Image.LANCZOS)
            
            # Random position
            max_x = img_size - new_w
            max_y = img_size - new_h
            
            if max_x > 0 and max_y > 0:
                x_pos = random.randint(0, max_x)
                y_pos = random.randint(0, max_y)
                
                # Paste sprite onto background
                background.paste(scaled_sprite, (x_pos, y_pos), scaled_sprite)
                
                # Calculate YOLO format label (normalized)
                x_center = (x_pos + new_w / 2) / img_size
                y_center = (y_pos + new_h / 2) / img_size
                width = new_w / img_size
                height = new_h / img_size
                
                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return background.convert('RGB'), labels
    
    def generate_dataset(self, augmentations_per_sprite=50, val_split=0.2, 
                        img_size=640, sprites_per_image=None, target_images_per_class=None):
        """
        Generate the full augmented dataset
        
        Args:
            augmentations_per_sprite: Number of augmented images per sprite (ignored if target_images_per_class is set)
            val_split: Fraction of data for validation (0.0-1.0)
            img_size: Output image size
            sprites_per_image: Number of sprites per image (None for random 1-3)
            target_images_per_class: Target total images per class (adaptive mode)
        """
        sprites = self.load_sprites()
        backgrounds = self.load_backgrounds()
        
        if not backgrounds:
            print("ERROR: No background images found!")
            return
        
        print(f"\nGenerating dataset...")
        print(f"Image size: {img_size}x{img_size}")
        print(f"Validation split: {val_split*100}%\n")
        
        class_names = {}
        total_train = 0
        total_val = 0
        
        for class_id, sprite_info in sprites.items():
            class_name = sprite_info['name']
            class_names[class_id] = class_name
            sprite_files = sprite_info['files']
            
            # Adaptive augmentation: calculate augmentations based on sprite count
            if target_images_per_class:
                augs_per_sprite = max(1, target_images_per_class // len(sprite_files))
                print(f"Class {class_id} ({class_name}): {len(sprite_files)} sprites → {augs_per_sprite} augmentations each (target: {target_images_per_class} images)")
            else:
                augs_per_sprite = augmentations_per_sprite
                print(f"Class {class_id} ({class_name}): {len(sprite_files)} sprites → {augs_per_sprite} augmentations each")
            
            for sprite_idx, sprite_path in enumerate(sprite_files):
                for aug_idx in range(augs_per_sprite):
                    # Random background
                    bg_path = random.choice(backgrounds)
                    
                    # Create augmented image
                    img, labels = self.create_augmented_image(
                        sprite_path, bg_path, class_id, img_size, sprites_per_image
                    )
                    
                    # Determine train/val split
                    is_val = random.random() < val_split
                    img_dir = self.val_img_dir if is_val else self.train_img_dir
                    lbl_dir = self.val_lbl_dir if is_val else self.train_lbl_dir
                    
                    # Generate unique filename
                    filename = f"class{class_id}_sprite{sprite_idx:04d}_aug{aug_idx:04d}"
                    
                    # Save image
                    img_path = img_dir / f"{filename}.jpg"
                    img.save(img_path, 'JPEG', quality=95)
                    
                    # Save label
                    lbl_path = lbl_dir / f"{filename}.txt"
                    with open(lbl_path, 'w') as f:
                        f.write('\n'.join(labels))
                    
                    if is_val:
                        total_val += 1
                    else:
                        total_train += 1
                
                if (sprite_idx + 1) % 10 == 0:
                    print(f"  Processed {sprite_idx + 1}/{len(sprite_files)} sprites")
        
        print(f"\n✓ Dataset generation complete!")
        print(f"  Training images: {total_train}")
        print(f"  Validation images: {total_val}")
        
        # Create dataset.yaml
        self.create_yaml(class_names)
    
    def create_yaml(self, class_names):
        """Create dataset.yaml configuration file"""
        yaml_path = self.output_dir / 'dataset.yaml'
        
        with open(yaml_path, 'w') as f:
            f.write(f"# YOLOv11 Dataset Configuration\n")
            f.write(f"path: {self.output_dir.absolute()}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n\n")
            f.write(f"# Classes\n")
            f.write(f"nc: {len(class_names)}  # number of classes\n")
            f.write(f"names:\n")
            for class_id in sorted(class_names.keys()):
                f.write(f"  {class_id}: {class_names[class_id]}\n")
        
        print(f"\n✓ Created {yaml_path}")


# Example usage
if __name__ == "__main__":
    # Configuration
    SPRITE_DIR = "sprites"  # Your sprite images (can have subfolders for classes)
    BACKGROUND_DIR = "arenas"  # Your background images
    OUTPUT_DIR = "dataset"  # Output directory
    
    # Optional: manually map folder names to class IDs
    CLASS_MAPPING = {
        'mario': 0,
        'luigi': 1,
        'peach': 2,
        # Add more as needed
    }
    
    # Create augmenter
    augmenter = SpriteDatasetAugmenter(
        sprite_dir=SPRITE_DIR,
        background_dir=BACKGROUND_DIR,
        output_dir=OUTPUT_DIR,
        #class_mapping=CLASS_MAPPING  # Optional
    )
    
    # Generate dataset
    augmenter.generate_dataset(
        augmentations_per_sprite=50,  # Used if target_images_per_class is None
        val_split=0.2,  # 20% for validation
        img_size=640,  # Standard YOLO size
        sprites_per_image=None,  # Random 1-3 sprites per image, or set specific number
        target_images_per_class=1000  # Adaptive mode: aim for ~1000 images per class
    )
    
    print("\n" + "="*50)
    print("Next steps:")
    print("1. Review the generated images in yolo_dataset/images/train")
    print("2. Train your model:")
    print("   yolo task=detect mode=train model=yolo11n.pt data=yolo_dataset/dataset.yaml epochs=100")
    print("="*50)