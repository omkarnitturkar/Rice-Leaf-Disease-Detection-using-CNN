# Rice Leaf Disease Classification 🌾

## Overview
This project implements deep learning models to classify rice leaf diseases using both Custom CNN and VGG16 Transfer Learning approaches. The system can accurately identify three types of rice leaf diseases:
- Bacterial Leaf Blight
- Brown Spot
- Leaf Smut


## 🔧 Requirements
```
numpy
tensorflow
keras
matplotlib
Pillow
```

## 📁 Project Structure
```
project/
│
├── Dataset_split/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   ├── custom_cnn_model.h5
│   └── vgg16_transfer_learning_model.h5
│
├── src/
│   └── train.py
│
└── README.md
```

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rice-leaf-disease.git
cd rice-leaf-disease
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run training**
```bash
python src/train.py
```

## 📊 Models & Architecture

### Custom CNN Architecture
```python
Sequential([
    Input(shape=(224, 224, 3))
    Conv2D(32, (3, 3), activation='relu')
    MaxPooling2D((2, 2))
    Conv2D(64, (3, 3), activation='relu')
    MaxPooling2D((2, 2))
    Conv2D(128, (3, 3), activation='relu')
    MaxPooling2D((2, 2))
    Conv2D(256, (3, 3), activation='relu')
    MaxPooling2D((2, 2))
    Dropout(0.5)
    Flatten()
    Dense(128, activation='relu')
    Dropout(0.5)
    Dense(3, activation='softmax')
])
```

### VGG16 Transfer Learning
```python
Sequential([
    VGG16(weights='imagenet', include_top=False)
    Flatten()
    Dense(256, activation='relu')
    Dropout(0.5)
    Dense(3, activation='softmax')
])
```

## 📈 Data Processing

### Dataset Split
- Training: 70%
- Validation: 15%
- Testing: 15%

### Data Augmentation
```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

## 🎯 Results

### Model Comparison
| Model | Test Accuracy | Training Time |
|-------|---------------|---------------|
| Custom CNN | 60.53% | 203.02 s |
| VGG16 Transfer Learning | 96.05% | 612.00 s |



![](https://github.com/omkarnitturkar/Rice-Leaf-Disease-Detection-using-CNN/blob/main/Model%20accuray%20Comparision.png)
### Training Parameters
- Image Size: 224x224
- Batch Size: 32
- Epochs: 30
- Optimizer: Adam
- Loss: Categorical Crossentropy

## 🚧 Challenges & Solutions

1. **Overfitting**
   - Issue: Validation accuracy lagging behind training accuracy
   - Solution: Implemented data augmentation and dropout layers

2. **Class Imbalance**
   - Issue: Uneven distribution of samples
   - Solution: Applied targeted data augmentation

3. **Resource Constraints**
   - Issue: High computational requirements for VGG16
   - Solution: Used transfer learning with frozen base layers

## 💻 Usage

### For Training
```python
# Import required libraries
from tensorflow.keras import layers, models, applications

# Create and train model
model = create_custom_cnn()
model.compile(optimizer='adam', 
             loss='categorical_crossentropy', 
             metrics=['accuracy'])
model.fit(train_generator, 
         epochs=30, 
         validation_data=val_generator)
```

### For Prediction
```python
# Load saved model
model = load_model('models/custom_cnn_model.h5')

# Prepare image
img = load_and_preprocess_image('path/to/image.jpg')

# Make prediction
prediction = model.predict(img)
```

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

