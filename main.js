const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const dotenv = require('dotenv');
const fetch = require('node-fetch');
dotenv.config();
function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false,
      allowRunningInsecureContent: true,
    },
  });
  win.webContents.session.setPermissionRequestHandler((webContents, permission, callback) => {
    if (permission === 'media') {
      callback(true);
    } else {
      callback(false);
    }
  });
  win.loadURL('http://localhost:3000');
}
app.whenReady().then(createWindow);
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
async function getAIRephrased({ eventType, content, confusionLevel = 0 }) {
  console.log(`Processing ${eventType} event with confusion level: ${confusionLevel}...`);
  const openaiKey = process.env.REACT_APP_OPENAI_API_KEY;
  const geminiKey = process.env.REACT_APP_GEMINI_API_KEY;
  console.log('OpenAI key available:', openaiKey && openaiKey !== 'your_openai_api_key_here');
  console.log('Gemini key available:', geminiKey && geminiKey !== 'your_gemini_api_key_here');
  let prompt = '';
  if (eventType === 'facial-confusion') {
    prompt = `The user appears confused based on facial expression analysis (confusion level: ${confusionLevel}). Please rephrase this text in simpler terms with analogies and examples: ${content}`;
  } else if (eventType === 'text-selection') {
    prompt = `The user selected this specific text, indicating they need clarification. Please explain this part in simpler terms: ${content}`;
  } else {
    prompt = `Rephrase this for a confused learner: ${content}`;
  }
  let response = '';
  if (openaiKey && openaiKey !== 'your_openai_api_key_here') {
    const url = 'https://api.openai.com/v1/chat/completions';
    const body = {
      model: 'gpt-3.5-turbo',
      messages: [
        { role: 'system', content: 'You are a helpful tutor.' },
        { role: 'user', content: `Rephrase this for a confused learner: ${content}` },
      ],
      max_tokens: 150,
    };
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${openaiKey}`,
        },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      console.log('Using OpenAI API...');
      response = data.choices?.[0]?.message?.content || 'No response from OpenAI.';
    } catch (err) {
      console.error('OpenAI Error:', err);
      response = 'Error contacting OpenAI.';
    }
  } else if (geminiKey && geminiKey !== 'your_gemini_api_key_here') {
    console.log('Using Gemini API...');
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${geminiKey}`;
    const body = {
      contents: [{ parts: [{ text: prompt }] }],
    };
    try {
      console.log('Gemini API request:', body);
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      console.log('Gemini response:', data);
      response = data.candidates?.[0]?.content?.parts?.[0]?.text || 'No response from Gemini.';
    } catch (err) {
      console.error('Gemini Error:', err);
      response = 'Error contacting Gemini.';
    }
  } else {
    console.log('No valid API key found');
    response = 'No API key provided.';
  }
  console.log('Final response:', response);
  return response;
}
ipcMain.handle('confusion-event', async (event, args) => {
  return await getAIRephrased(args);
});
