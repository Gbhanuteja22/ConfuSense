const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('electronAPI', {
  sendConfusionEvent: async (args) => {
    return await ipcRenderer.invoke('confusion-event', args);
  },
});
