# SatVision AI - Satellite Image Classification Website

A stunning 3D animated website for satellite image classification using Deep Learning.

## Features

### 🏠 Home Page
- Animated 3D wireframe Earth
- Glitch text effects
- Floating stat cards with model metrics
- Particle background animation

### 🖼️ Classify Page
- Drag & drop image upload
- Real-time image preview
- AI-powered classification
- Animated confidence bar
- Supports: Desert, Green Area, Water, Cloudy

### 🌍 Explore Page
- Interactive 3D globe
- Classification categories showcase
- Hover animations on class cards

## Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript, Three.js
- **Backend**: Flask, PyTorch
- **Model**: EfficientNet-B3
- **3D Graphics**: Three.js for particle systems and 3D objects

## Setup Instructions

### 1. Install Dependencies

```bash
pip install flask flask-cors torch torchvision pillow
```

### 2. Train Your Model (Optional)

If you haven't trained the model yet, run the main.py script:

```bash
python main.py
```

This will download the dataset and train the model. Save the model weights:

```python
torch.save(model.state_dict(), 'model.pth')
```

### 3. Update Backend

In `app.py`, uncomment this line to load your trained model:

```python
model.load_state_dict(torch.load('model.pth', map_location=device))
```

### 4. Run the Application

```bash
python app.py
```

### 5. Open in Browser

Navigate to: `http://localhost:5000`

## File Structure

```
├── index.html          # Main HTML structure
├── style.css           # Styling and animations
├── script.js           # Frontend logic and Three.js
├── app.py              # Flask backend
├── main.py             # Model training script
└── README.md           # This file
```

## Usage

1. **Home Page**: View project overview and statistics
2. **Classify Page**: 
   - Click or drag-drop a satellite image
   - Click "Analyze Image"
   - View classification results with confidence score
3. **Explore Page**: Browse different land classification types

## Model Classes

- 🏜️ **Desert**: Arid landscapes
- 🌲 **Green Area**: Forests and vegetation
- 🌊 **Water**: Oceans, lakes, rivers
- ☁️ **Cloudy**: Cloud-covered regions

## Demo Mode

If the Flask backend is not running, the website falls back to demo mode with simulated classifications.

## Customization

- Modify colors in `style.css` (search for `#00d4ff` and `#7b2ff7`)
- Adjust particle count in `script.js` (line 13)
- Change model architecture in `app.py`

## Browser Compatibility

- Chrome (recommended)
- Firefox
- Edge
- Safari

Requires WebGL support for 3D animations.

## Credits

Built with ❤️ using PyTorch and Three.js
