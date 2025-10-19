from ultralytics import YOLO
import torch
import os

# ============================================
# Check GPU Configuration
# ============================================
print("="*60)
print("GPU Configuration")
print("="*60)
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: No GPU detected! Training will be very slow on CPU.")

# ============================================
# STEP 1: Configuration
# ============================================

# Model selection
MODEL_SIZE = 'yolo11s.pt'  # n, s, m, l, x

# Dataset
DATASET_YAML = 'dataset/dataset.yaml'

# Training parameters
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 64  # Increase for multi-GPU (will be split across GPUs)
PROJECT_NAME = 'sprite_detection_multigpu'

# Multi-GPU settings
DEVICE = [0, 1, 2, 3]  # List of GPU IDs to use (e.g., [0, 1] for 2 GPUs)
# Alternative: DEVICE = 'cpu' for CPU only
# Alternative: DEVICE = 0 for single GPU

# Advanced settings
WORKERS = 8  # Number of data loading workers per GPU
CACHE = True  # Cache images in RAM for faster training (if you have enough RAM)

# ============================================
# STEP 2: Calculate Effective Batch Size
# ============================================
if isinstance(DEVICE, list):
    num_gpus = len(DEVICE)
    batch_per_gpu = BATCH_SIZE // num_gpus
    effective_batch = batch_per_gpu * num_gpus
    
    print(f"\nMulti-GPU Configuration:")
    print(f"  GPUs: {DEVICE}")
    print(f"  Total Batch Size: {effective_batch}")
    print(f"  Batch per GPU: {batch_per_gpu}")
    print(f"  Workers: {WORKERS}")
else:
    print(f"\nSingle Device Configuration:")
    print(f"  Device: {DEVICE}")
    print(f"  Batch Size: {BATCH_SIZE}")

# ============================================
# STEP 3: Load Model
# ============================================
print(f"\nLoading model: {MODEL_SIZE}")
model = YOLO(MODEL_SIZE)

# ============================================
# STEP 4: Train with Multi-GPU
# ============================================
print("\n" + "="*60)
print("Starting Multi-GPU Training")
print("="*60)
print(f"Dataset: {DATASET_YAML}")
print(f"Epochs: {EPOCHS}")
print(f"Image Size: {IMAGE_SIZE}")
print(f"Device(s): {DEVICE}")
print("="*60 + "\n")

results = model.train(
    # Data
    data=DATASET_YAML,
    
    # Training duration
    epochs=EPOCHS,
    
    # Image settings
    imgsz=IMAGE_SIZE,
    
    # Multi-GPU settings
    batch=BATCH_SIZE,
    device=DEVICE,  # This enables multi-GPU automatically
    workers=WORKERS,  # Data loading workers
    
    # Performance
    cache=CACHE,  # Cache images in RAM (requires sufficient RAM)
    close_mosaic=10,  # Disable mosaic augmentation in final epochs
    
    # Saving
    project=PROJECT_NAME,
    name='train',
    save=True,
    save_period=10,  # Save checkpoint every N epochs
    
    # Early stopping
    patience=50,
    
    # Optimization
    optimizer='auto',  # 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'
    lr0=0.01,  # Initial learning rate
    lrf=0.01,  # Final learning rate (lr0 * lrf)
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,  # Warmup epochs
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # Data augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    
    # Validation
    val=True,
    
    # Logging
    verbose=True,
    plots=True,
    
    # Mixed precision training (faster on modern GPUs)
    amp=True,  # Automatic Mixed Precision
)

# ============================================
# STEP 5: Training Summary
# ============================================
print("\n" + "="*60)
print("Training Complete!")
print("="*60)

# Validate
print("\nValidating on validation set...")
metrics = model.val()

print(f"\nFinal Metrics:")
print(f"  mAP50: {metrics.box.map50:.4f}")
print(f"  mAP50-95: {metrics.box.map:.4f}")
print(f"  Precision: {metrics.box.mp:.4f}")
print(f"  Recall: {metrics.box.mr:.4f}")

print(f"\nModel saved to:")
print(f"  Best: {PROJECT_NAME}/train/weights/best.pt")
print(f"  Last: {PROJECT_NAME}/train/weights/last.pt")

# ============================================
# STEP 6: Benchmark Inference Speed
# ============================================
print("\n" + "="*60)
print("Benchmarking Inference Speed")
print("="*60)

best_model = YOLO(f'{PROJECT_NAME}/train/weights/best.pt')

# Test inference speed
print("\nTo benchmark inference speed:")
print("  results = best_model.val(split='test', batch=1, device=0)")

# ============================================
# STEP 7: Export for Deployment
# ============================================
print("\n" + "="*60)
print("Export Options for Deployment")
print("="*60)
print("\nExport to different formats:")
print("  # ONNX (cross-platform)")
print("  best_model.export(format='onnx', dynamic=True)")
print("\n  # TensorRT (NVIDIA GPUs - fastest)")
print("  best_model.export(format='engine', device=0)")
print("\n  # TorchScript")
print("  best_model.export(format='torchscript')")
print("\n  # OpenVINO (Intel)")
print("  best_model.export(format='openvino')")
print("\n  # TensorFlow Lite (mobile)")
print("  best_model.export(format='tflite')")
print("\n  # CoreML (Apple)")
print("  best_model.export(format='coreml')")

# ============================================
# STEP 8: Testing Code
# ============================================
print("\n" + "="*60)
print("Quick Test")
print("="*60)
print("\n# Single image prediction:")
print("results = best_model.predict('test.jpg', conf=0.25, device=0)")
print("results[0].show()")
print("\n# Batch prediction:")
print("results = best_model.predict('test_folder/', conf=0.25, device=0)")
print("\n# Video prediction:")
print("results = best_model.predict('video.mp4', conf=0.25, device=0, save=True)")

print("\n" + "="*60)
print("Training session complete!")
print("="*60)