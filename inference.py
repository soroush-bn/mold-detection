import torch
import argparse
from pathlib import Path
from PIL import Image

def load_model(weights_path, image_size=320, conf_thresh=0.25, iou_thresh=0.45):
    print("📦 Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
    model.conf = conf_thresh
    model.iou = iou_thresh
    model.imgsz = image_size
    return model

def run_inference(model, source, output_dir):
    print(f"🖼️ Running inference on: {source}")
    results = model(source)
    results.save(save_dir=output_dir)
    print(f"✅ Results saved in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 Inference Script for Raspberry Pi")
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLOv5 .pt model file')
    parser.add_argument('--image', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='results', help='Directory to save results')

    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights)

    run_inference(model, args.image, args.output)

if __name__ == "__main__":
    main()
