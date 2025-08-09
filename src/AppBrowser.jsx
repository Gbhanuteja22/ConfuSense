import React, { useEffect, useRef, useState, useCallback } from 'react';
import ReactQuill from 'react-quill';
import 'react-quill/dist/quill.snow.css';
import './styles.css';
import * as tf from '@tensorflow/tfjs';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';

function App() {
  const [content, setContent] = useState(`A closure is a function that has access to variables in its outer (enclosing) scope even after the outer function has returned. This is possible because functions in JavaScript form closures. The closure has access to variables in three scopes: variables in its own scope, variables in the enclosing function's scope, and global variables.`);
  const [faceModel, setFaceModel] = useState(null);
  const [isConfused, setIsConfused] = useState(false);
  const [confusionLevel, setConfusionLevel] = useState(0);
  const [webcamEnabled, setWebcamEnabled] = useState(false);
  const [cameraStream, setCameraStream] = useState(null);
  const [calibrationBaseline, setCalibrationBaseline] = useState(null);
  
  const videoRef = useRef(null);
  const confusionTimer = useRef(null);

  const sendEventsToAI = async () => {
    const geminiKey = process.env.REACT_APP_GEMINI_API_KEY;
    
    const prompt = `Simplify this text for a confused learner: ${content}`;
    let response = '';
    
    if (geminiKey && geminiKey !== 'your_gemini_api_key_here') {
      try {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${geminiKey}`;
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contents: [{ parts: [{ text: prompt }] }],
          }),
        });
        const data = await res.json();
        response = data.candidates?.[0]?.content?.parts?.[0]?.text || 'No response';
      } catch (err) {
        response = 'API Error';
      }
    }
    
    if (response && response !== 'API Error') {
      const timestamp = new Date().toLocaleTimeString();
      const aiSuggestion = `<br/><div style="background-color: #e3f2fd; border-left: 4px solid #2196F3; padding: 10px; margin: 10px 0; border-radius: 5px;"><strong>AI (${timestamp}):</strong><br/>${response}</div>`;
      
      setContent(prevContent => {
        const aiSuggestionRegex = /<div[^>]*background-color:\s*#e3f2fd[^>]*>.*?<\/div>/gi;
        const existingAiSuggestions = prevContent.match(aiSuggestionRegex) || [];
        const originalContent = prevContent.replace(aiSuggestionRegex, '').replace(/(<br\s*\/?>)+$/gi, '');
        const allAiSuggestions = [aiSuggestion, ...existingAiSuggestions];
        return originalContent + allAiSuggestions.join('');
      });
    }
  };

  const initializeFaceDetection = async () => {
    try {
      await tf.ready();
      const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
      const detectorConfig = {
        runtime: 'mediapipe',
        solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
        refineLandmarks: true,
      };
      const detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
      setFaceModel(detector);
    } catch (error) {
      console.error('Face detection error:', error);
    }
  };

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraStream(stream);
        setWebcamEnabled(true);
      }
    } catch (error) {
      alert('Camera access denied or unavailable');
    }
  };

  const stopWebcam = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setWebcamEnabled(false);
    setConfusionLevel(0);
    setIsConfused(false);
  };

  const analyzeFacialExpression = useCallback(async () => {
    if (!faceModel || !videoRef.current || !webcamEnabled) return;

    try {
      const faces = await faceModel.estimateFaces(videoRef.current);
      
      if (faces.length > 0) {
        const keypoints = faces[0].keypoints;
        let confusionScore = 0;
        let confidenceMultiplier = 1.0;

        const leftEyebrowInner = [70, 63, 105];
        const leftEyebrowOuter = [66, 107, 55];
        const rightEyebrowInner = [296, 334, 293];
        const rightEyebrowOuter = [300, 276, 285];
        
        const leftEyeInner = [33, 7, 163];
        const leftEyeOuter = [144, 145, 153];
        const leftEyeUpper = [159, 158, 157, 173];
        const leftEyeLower = [145, 153, 154, 155];
        
        const rightEyeInner = [362, 382, 381];
        const rightEyeOuter = [380, 374, 373];
        const rightEyeUpper = [386, 385, 384, 398];
        const rightEyeLower = [374, 373, 390, 249];

        try {
          const leftBrowInnerY = leftEyebrowInner.reduce((sum, idx) => sum + (keypoints[idx]?.y || 0), 0) / leftEyebrowInner.length;
          const leftBrowOuterY = leftEyebrowOuter.reduce((sum, idx) => sum + (keypoints[idx]?.y || 0), 0) / leftEyebrowOuter.length;
          const rightBrowInnerY = rightEyebrowInner.reduce((sum, idx) => sum + (keypoints[idx]?.y || 0), 0) / rightEyebrowInner.length;
          const rightBrowOuterY = rightEyebrowOuter.reduce((sum, idx) => sum + (keypoints[idx]?.y || 0), 0) / rightEyebrowOuter.length;
          
          const leftEyeInnerY = leftEyeInner.reduce((sum, idx) => sum + (keypoints[idx]?.y || 0), 0) / leftEyeInner.length;
          const rightEyeInnerY = rightEyeInner.reduce((sum, idx) => sum + (keypoints[idx]?.y || 0), 0) / rightEyeInner.length;
          
          const leftBrowEyeDistance = leftBrowInnerY - leftEyeInnerY;
          const rightBrowEyeDistance = rightBrowInnerY - rightEyeInnerY;
          const avgBrowDistance = (leftBrowEyeDistance + rightBrowEyeDistance) / 2;
          
          const browAsymmetry = Math.abs(leftBrowEyeDistance - rightBrowEyeDistance);
          
          if (avgBrowDistance < 20) confusionScore += 0.4;
          if (browAsymmetry > 6) confusionScore += 0.3;
          
          const leftBrowAngle = Math.abs(leftBrowInnerY - leftBrowOuterY);
          const rightBrowAngle = Math.abs(rightBrowInnerY - rightBrowOuterY);
          if (leftBrowAngle > 8 || rightBrowAngle > 8) confusionScore += 0.25;
        } catch (e) {}

        try {
          const leftEyeUpperY = leftEyeUpper.reduce((sum, idx) => sum + (keypoints[idx]?.y || 0), 0) / leftEyeUpper.length;
          const leftEyeLowerY = leftEyeLower.reduce((sum, idx) => sum + (keypoints[idx]?.y || 0), 0) / leftEyeLower.length;
          const rightEyeUpperY = rightEyeUpper.reduce((sum, idx) => sum + (keypoints[idx]?.y || 0), 0) / rightEyeUpper.length;
          const rightEyeLowerY = rightEyeLower.reduce((sum, idx) => sum + (keypoints[idx]?.y || 0), 0) / rightEyeLower.length;
          
          const leftEyeHeight = Math.abs(leftEyeUpperY - leftEyeLowerY);
          const rightEyeHeight = Math.abs(rightEyeUpperY - rightEyeLowerY);
          const avgEyeHeight = (leftEyeHeight + rightEyeHeight) / 2;
          const eyeAsymmetry = Math.abs(leftEyeHeight - rightEyeHeight);
          
          if (avgEyeHeight < 4) confusionScore += 0.35;
          if (avgEyeHeight > 20) confusionScore += 0.3;
          if (eyeAsymmetry > 4) confusionScore += 0.2;
          
          const leftEyeInnerX = leftEyeInner.reduce((sum, idx) => sum + (keypoints[idx]?.x || 0), 0) / leftEyeInner.length;
          const leftEyeOuterX = leftEyeOuter.reduce((sum, idx) => sum + (keypoints[idx]?.x || 0), 0) / leftEyeOuter.length;
          const rightEyeInnerX = rightEyeInner.reduce((sum, idx) => sum + (keypoints[idx]?.x || 0), 0) / rightEyeInner.length;
          const rightEyeOuterX = rightEyeOuter.reduce((sum, idx) => sum + (keypoints[idx]?.x || 0), 0) / rightEyeOuter.length;
          
          const leftEyeWidth = Math.abs(leftEyeOuterX - leftEyeInnerX);
          const rightEyeWidth = Math.abs(rightEyeOuterX - rightEyeInnerX);
          const eyeWidthRatio = leftEyeWidth / rightEyeWidth;
          
          if (eyeWidthRatio < 0.85 || eyeWidthRatio > 1.15) confusionScore += 0.25;
        } catch (e) {}

        try {
          const upperLip = [13, 82, 18, 17, 18, 200];
          const lowerLip = [14, 87, 178, 88, 95, 179];
          const mouthCorners = [61, 291, 39, 269];
          
          const upperLipY = upperLip.reduce((sum, idx) => sum + (keypoints[idx]?.y || 0), 0) / upperLip.length;
          const lowerLipY = lowerLip.reduce((sum, idx) => sum + (keypoints[idx]?.y || 0), 0) / lowerLip.length;
          const mouthHeight = Math.abs(upperLipY - lowerLipY);
          
          const leftCornerX = (keypoints[61]?.x || 0);
          const rightCornerX = (keypoints[291]?.x || 0);
          const mouthCenter = (leftCornerX + rightCornerX) / 2;
          const noseTipX = keypoints[1]?.x || mouthCenter;
          const mouthOffset = Math.abs(mouthCenter - noseTipX);
          
          const leftCornerY = (keypoints[61]?.y || 0);
          const rightCornerY = (keypoints[291]?.y || 0);
          const mouthTilt = Math.abs(leftCornerY - rightCornerY);
          
          if (mouthHeight > 8 && mouthHeight < 20) confusionScore += 0.3;
          if (mouthOffset > 5) confusionScore += 0.25;
          if (mouthTilt > 3) confusionScore += 0.2;
          
          const mouthWidth = Math.abs(leftCornerX - rightCornerX);
          if (mouthWidth < 25) confusionScore += 0.15;
        } catch (e) {}

        try {
          const noseTip = keypoints[1];
          const noseBridge = keypoints[168];
          const leftCheek = keypoints[234];
          const rightCheek = keypoints[454];
          const chin = keypoints[18];
          
          if (noseTip && noseBridge && leftCheek && rightCheek) {
            const headTiltX = Math.abs(leftCheek.y - rightCheek.y);
            const headTiltY = Math.abs(noseTip.x - noseBridge.x);
            
            if (headTiltX > 12) confusionScore += 0.25;
            if (headTiltY > 8) confusionScore += 0.2;
            
            const faceWidth = Math.abs(leftCheek.x - rightCheek.x);
            const faceHeight = chin ? Math.abs(noseBridge.y - chin.y) : 100;
            const faceRatio = faceWidth / faceHeight;
            
            if (faceRatio < 0.6 || faceRatio > 1.1) confusionScore += 0.15;
          }
        } catch (e) {}

        try {
          const forehead = [10, 151, 9, 10, 151];
          const jawline = [172, 136, 150, 149, 176];
          
          const foreheadY = forehead.reduce((sum, idx) => sum + (keypoints[idx]?.y || 0), 0) / forehead.length;
          const jawlineY = jawline.reduce((sum, idx) => sum + (keypoints[idx]?.y || 0), 0) / jawline.length;
          const faceLength = Math.abs(foreheadY - jawlineY);
          
          if (faceLength < 120 || faceLength > 200) {
            confidenceMultiplier *= 0.8;
          }
        } catch (e) {}

        const currentTime = Date.now();
        if (!window.confusionHistory) window.confusionHistory = [];
        
        window.confusionHistory.push({ 
          score: confusionScore * confidenceMultiplier, 
          time: currentTime,
          confidence: confidenceMultiplier 
        });
        window.confusionHistory = window.confusionHistory.filter(h => currentTime - h.time < 3500);
        
        const weights = window.confusionHistory.map((h, i) => {
          const age = (currentTime - h.time) / 3500;
          const recency = Math.exp(-age * 2);
          return recency * h.confidence;
        });
        
        const totalWeight = weights.reduce((sum, w) => sum + w, 0);
        const weightedScore = window.confusionHistory.reduce((sum, h, i) => 
          sum + (h.score * weights[i]), 0) / (totalWeight || 1);
        
        let finalScore = weightedScore;
        if (calibrationBaseline && calibrationBaseline.neutral !== undefined && calibrationBaseline.confused !== undefined) {
          const neutralScore = calibrationBaseline.neutral;
          const confusedScore = calibrationBaseline.confused;
          const range = Math.max(0.1, confusedScore - neutralScore);
          
          if (weightedScore > neutralScore) {
            finalScore = Math.min(1.0, (weightedScore - neutralScore) / range);
          } else {
            finalScore = Math.max(0, (weightedScore - neutralScore) / range);
          }
          
          finalScore = Math.pow(finalScore, 0.8);
        }
        
        setConfusionLevel(finalScore);
        const wasConfused = isConfused;
        const isCurrentlyConfused = finalScore > 0.42;
        setIsConfused(isCurrentlyConfused);
        
        if (isCurrentlyConfused && !wasConfused && window.confusionHistory.length >= 6) {
          clearTimeout(confusionTimer.current);
          confusionTimer.current = setTimeout(() => {
            sendEventsToAI();
          }, 1000);
        }
      } else {
        setConfusionLevel(0);
        setIsConfused(false);
      }
    } catch (error) {
      console.error('Analysis error:', error);
    }
  }, [faceModel, webcamEnabled, isConfused, content, calibrationBaseline]);

  const startCalibration = () => {
    setCalibrationBaseline({ neutral: 0, confused: 0, step: 1 });
    window.confusionHistory = [];
  };

  const completeCalibrationStep = () => {
    if (!calibrationBaseline || window.confusionHistory.length < 12) return;
    
    const recentSamples = window.confusionHistory.slice(-12);
    const sortedSamples = recentSamples.map(h => h.score).sort((a, b) => a - b);
    const median = sortedSamples[Math.floor(sortedSamples.length / 2)];
    const q1 = sortedSamples[Math.floor(sortedSamples.length * 0.25)];
    const q3 = sortedSamples[Math.floor(sortedSamples.length * 0.75)];
    const iqr = q3 - q1;
    
    const filteredSamples = recentSamples.filter(h => {
      const score = h.score;
      return score >= (q1 - 1.5 * iqr) && score <= (q3 + 1.5 * iqr);
    });
    
    const avg = filteredSamples.reduce((sum, h) => sum + h.score, 0) / filteredSamples.length;
    
    if (calibrationBaseline.step === 1) {
      setCalibrationBaseline(prev => ({ ...prev, neutral: avg, step: 2 }));
      window.confusionHistory = [];
    } else if (calibrationBaseline.step === 2) {
      setCalibrationBaseline(prev => ({ ...prev, confused: avg, step: 3 }));
      localStorage.setItem('confusionCalibration', JSON.stringify({ neutral: calibrationBaseline.neutral, confused: avg }));
    }
  };

  useEffect(() => {
    if (webcamEnabled && faceModel) {
      const interval = setInterval(analyzeFacialExpression, 200);
      return () => clearInterval(interval);
    }
  }, [webcamEnabled, faceModel, analyzeFacialExpression]);

  useEffect(() => {
    initializeFaceDetection();
    const saved = localStorage.getItem('confusionCalibration');
    if (saved) {
      try {
        setCalibrationBaseline(JSON.parse(saved));
      } catch (e) {}
    }
  }, []);

  return (
    <div style={{ padding: 20, maxWidth: '1000px', margin: '0 auto' }}>
      <h1 style={{ textAlign: 'center', color: '#333', marginBottom: 30 }}>
        ConfuSense
      </h1>
      
      <div style={{ display: 'flex', gap: 20, marginBottom: 30 }}>
        <div style={{ flex: 1 }}>
          <h3>Facial Detection</h3>
          <div style={{ border: '2px solid #ddd', borderRadius: 10, padding: 15 }}>
            <div style={{ 
              marginBottom: 15, 
              display: 'flex', 
              justifyContent: 'center', 
              alignItems: 'center',
              minHeight: 240
            }}>
              <video 
                ref={videoRef} 
                autoPlay 
                playsInline 
                muted 
                style={{ 
                  width: 320, 
                  height: 240,
                  borderRadius: 8,
                  backgroundColor: '#f0f0f0',
                  border: webcamEnabled ? '2px solid #4CAF50' : '2px solid #ccc'
                }}
              />
            </div>

            <div style={{ display: 'flex', gap: 10, marginBottom: 15, justifyContent: 'center' }}>
              {!webcamEnabled ? (
                <button onClick={startWebcam} style={{ 
                  padding: '12px 24px', 
                  fontSize: '16px', 
                  backgroundColor: '#4CAF50', 
                  color: 'white', 
                  border: 'none', 
                  borderRadius: 8, 
                  cursor: 'pointer'
                }}>
                  Start Camera
                </button>
              ) : (
                <button onClick={stopWebcam} style={{ 
                  padding: '12px 24px', 
                  fontSize: '16px', 
                  backgroundColor: '#f44336', 
                  color: 'white', 
                  border: 'none', 
                  borderRadius: 8, 
                  cursor: 'pointer'
                }}>
                  Stop Camera
                </button>
              )}
            </div>

            {webcamEnabled && (
              <div style={{ marginTop: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ fontSize: 14, fontWeight: 'bold' }}>Confusion:</span>
                  <div style={{
                    flex: 1,
                    height: 16,
                    backgroundColor: '#f0f0f0',
                    borderRadius: 8,
                    overflow: 'hidden',
                    border: '1px solid #ddd',
                    position: 'relative'
                  }}>
                    <div style={{
                      width: `${confusionLevel * 100}%`,
                      height: '100%',
                      backgroundColor: 
                        confusionLevel > 0.6 ? '#ff4444' : 
                        confusionLevel > 0.3 ? '#ff8800' : '#44ff44',
                      transition: 'all 0.3s ease'
                    }} />
                    <span style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      fontSize: '10px',
                      fontWeight: 'bold',
                      color: '#333'
                    }}>
                      {Math.round(confusionLevel * 100)}%
                    </span>
                  </div>
                  <span style={{ 
                    color: confusionLevel > 0.45 ? '#ff4444' : '#44ff44',
                    fontWeight: 'bold',
                    fontSize: 14,
                    minWidth: '60px'
                  }}>
                    {confusionLevel > 0.45 ? 'Confused' : 'Clear'}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
        
        <div style={{ flex: 1 }}>
          <h3>Calibration</h3>
          <div style={{ border: '2px solid #ddd', borderRadius: 10, padding: 15 }}>
            {!calibrationBaseline?.step && (
              <button 
                onClick={startCalibration}
                disabled={!webcamEnabled}
                style={{ 
                  padding: '10px 20px', 
                  fontSize: '14px', 
                  backgroundColor: webcamEnabled ? '#4CAF50' : '#ccc', 
                  color: 'white', 
                  border: 'none', 
                  borderRadius: 5, 
                  cursor: webcamEnabled ? 'pointer' : 'not-allowed',
                  width: '100%'
                }}
              >
                Start Calibration
              </button>
            )}
            
            {calibrationBaseline?.step === 1 && (
              <div>
                <div style={{ fontSize: 14, marginBottom: 10 }}>
                  Step 1: Neutral Expression
                </div>
                <div style={{ fontSize: 12, color: '#666', marginBottom: 10 }}>
                  Samples: {window.confusionHistory?.length || 0}/12
                </div>
                <button 
                  onClick={completeCalibrationStep}
                  disabled={!window.confusionHistory || window.confusionHistory.length < 12}
                  style={{ 
                    padding: '8px 16px', 
                    fontSize: '14px', 
                    backgroundColor: window.confusionHistory?.length >= 12 ? '#FF9800' : '#ccc', 
                    color: 'white', 
                    border: 'none', 
                    borderRadius: 5, 
                    cursor: window.confusionHistory?.length >= 12 ? 'pointer' : 'not-allowed',
                    width: '100%'
                  }}
                >
                  Next: Confused
                </button>
              </div>
            )}
            
            {calibrationBaseline?.step === 2 && (
              <div>
                <div style={{ fontSize: 14, marginBottom: 10 }}>
                  Step 2: Confused Expression
                </div>
                <div style={{ fontSize: 12, color: '#666', marginBottom: 10 }}>
                  Samples: {window.confusionHistory?.length || 0}/12
                </div>
                <button 
                  onClick={completeCalibrationStep}
                  disabled={!window.confusionHistory || window.confusionHistory.length < 12}
                  style={{ 
                    padding: '8px 16px', 
                    fontSize: '14px', 
                    backgroundColor: window.confusionHistory?.length >= 12 ? '#FF9800' : '#ccc', 
                    color: 'white', 
                    border: 'none', 
                    borderRadius: 5, 
                    cursor: window.confusionHistory?.length >= 12 ? 'pointer' : 'not-allowed',
                    width: '100%'
                  }}
                >
                  Complete
                </button>
              </div>
            )}
            
            {calibrationBaseline?.step === 3 && (
              <div>
                <div style={{ fontSize: 14, color: '#4CAF50', marginBottom: 10 }}>
                  Calibrated
                </div>
                <button 
                  onClick={() => {
                    setCalibrationBaseline(null);
                    localStorage.removeItem('confusionCalibration');
                  }}
                  style={{ 
                    padding: '8px 16px', 
                    fontSize: '12px', 
                    backgroundColor: '#f44336', 
                    color: 'white', 
                    border: 'none', 
                    borderRadius: 5, 
                    cursor: 'pointer',
                    width: '100%'
                  }}
                >
                  Reset
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      <div style={{ marginBottom: 30 }}>
        <h3>Study Content</h3>
        <div style={{ border: '2px solid #ddd', borderRadius: 10, overflow: 'hidden' }}>
          <ReactQuill
            value={content}
            onChange={setContent}
            modules={{
              toolbar: [
                [{ 'header': [1, 2, 3, false] }],
                ['bold', 'italic', 'underline'],
                [{ 'list': 'ordered'}, { 'list': 'bullet' }],
                ['clean']
              ],
            }}
            formats={['header', 'bold', 'italic', 'underline', 'list', 'bullet']}
            style={{ minHeight: 300 }}
          />
        </div>
        <div style={{ marginTop: 10 }}>
          <button 
            onClick={() => {
              let cleanContent = content;
              cleanContent = cleanContent.replace(/<div[^>]*background-color:\s*#e3f2fd[^>]*>.*?<\/div>/gi, '');
              cleanContent = cleanContent.replace(/(<br\s*\/?>)+$/gi, '');
              setContent(cleanContent);
            }}
            style={{ 
              padding: '8px 16px', 
              fontSize: '14px', 
              backgroundColor: '#FF9800', 
              color: 'white', 
              border: 'none', 
              borderRadius: 5, 
              cursor: 'pointer'
            }}
          >
            Clear AI Suggestions
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
