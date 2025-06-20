#!/bin/bash

echo "🛠️  Setting up YOLOv5n inference environment on Raspberry Pi..."

# Step 1: Update and install base packages
echo "📦 Updating system and installing Git + Python3 pip..."
sudo apt update && sudo apt upgrade -y
sudo apt install git python3-pip -y

# Step 2: Clone YOLOv5 repository
echo "📁 Cloning YOLOv5 repository..."
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# Step 3: Install lightweight dependencies (Torch CPU + other required libs)
echo "📦 Installing Python dependencies for Pi (lightweight versions)..."
pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install -r requirements.txt
pip3 install opencv-python-headless

# Step 4: Copy your best.pt model to YOLOv5 folder
echo "📥 Copying best.pt model (make sure it's in the same folder as this script)..."
cp ../best.pt .
cp ../detect.py .

# Step 5: Optional — Copy test image
if [ -f "../test.jpg" ]; then
    echo "🖼️  Found test.jpg — copying it in..."
    cp ../test.jpg .
else
    echo "⚠️  No test.jpg found in parent folder. Please add your own image later."
fi

# Step 6: Run inference
echo "🚀 Running YOLOv5n inference..."
python3 detect.py \
    --weights best.pt \
    --source test.jpg \
    --img 320 \
    --conf-thres 0.25 \
    --save-csv \
    --name rpi_nano_run

echo "✅ Inference complete. Output saved to: yolov5/runs/detect/rpi_nano_run/"
