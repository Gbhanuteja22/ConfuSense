# 🧠 ConfuSense

**An AI-powered study companion that uses facial recognition to detect when you're confused and automatically provides simplified explanations.**

## ✨ Features

- **🔍 Real-time Confusion Detection**: Advanced facial landmark analysis using MediaPipe Face Mesh
- **🤖 AI-Powered Explanations**: Integration with OpenAI GPT and Google Gemini APIs
- **📝 Rich Text Editor**: Built-in ReactQuill editor for study materials
- **⚙️ Personal Calibration**: Custom baseline setup for accurate detection
- **🎯 Precision Algorithm**: Multi-factor analysis of eyebrows, eyes, mouth, and head position
- **📱 Cross-Platform**: Available as web app and desktop application

## 🚀 Quick Start

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Webcam for facial detection
- API keys (OpenAI or Google Gemini)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Gbhanuteja22/ConfuSense.git
   cd confusense
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your API keys:
   ```env
   REACT_APP_GEMINI_API_KEY=your_gemini_api_key_here
   REACT_APP_OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🎮 Usage

### Option 1: Web Browser (Recommended)
```bash
npm run start
```
Open [http://localhost:3000](http://localhost:3000) in Chrome or Firefox for best webcam support.

### Option 2: Desktop Application
```bash
# Terminal 1: Start React dev server
npm run start

```

## 📊 How It Works

1. **Calibration**: Set up personal facial baselines (neutral vs confused expressions)
2. **Detection**: Real-time analysis of 468 facial landmarks at 5fps
3. **Scoring**: Multi-factor confusion algorithm with temporal smoothing
4. **AI Response**: Automatic generation of simplified explanations when confusion detected

### Facial Analysis Features
- **Eyebrow Position**: Inner/outer tracking with asymmetry detection
- **Eye Analysis**: Upper/lower eyelid movement and width ratios
- **Mouth Tracking**: Lip position, corners, and tension analysis
- **Head Position**: Multi-point tilt and orientation detection

## 🛠️ Build & Deploy

### Development Build
```bash
npm run build
```

### Desktop Package
```bash
npm run package
```

## 🎯 Technical Stack

- **Frontend**: React 18 + ReactQuill
- **AI Detection**: TensorFlow.js + MediaPipe Face Mesh
- **Build**: Webpack 5 + Babel
- **APIs**: OpenAI GPT-3.5 / Google Gemini Pro

## 📁 Project Structure

```
confusense/
├── src/
│   ├── AppBrowser.jsx    # Main application component
│   ├── index.jsx         # Entry point
│   └── styles.css        # Application styles
├── public/
│   └── index.html        # HTML template
├── main.js               # Electron main process
├── preload.js            # Electron preload script
├── webpack.config.js     # Webpack configuration
└── package.json          # Project dependencies
```
