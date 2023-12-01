// /src/renderer.tsx
import '../public/styles/global.css';

document.addEventListener('DOMContentLoaded', () => {
  window.electron
    .fetchData()
    .then((data) => {
      console.log('Data received from IPC:', data);

      const apiDataElement = document.getElementById('api-data');
      const loadingElement = document.getElementById('loading');
      const contentElement = document.getElementById('content');

      if (apiDataElement) {
        apiDataElement.innerText = JSON.stringify(data, null, 2); // Display the data
      }
      if (loadingElement) {
        loadingElement.style.display = 'none'; // Hide loading message
      }
      if (contentElement) {
        contentElement.style.display = 'block'; // Show content
      }
    })
    .catch((error) => {
      console.error('Error in fetchData:', error);
    });
});
