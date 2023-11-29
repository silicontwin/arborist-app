// /src/renderer.js
document.addEventListener('DOMContentLoaded', () => {
  const fetchData = () => {
    axios
      .get('http://localhost:8000/data')
      .then((response) => {
        document.getElementById('api-data').innerText = JSON.stringify(
          response.data,
        );
        document.getElementById('loading').style.display = 'none'; // Hide loading
        document.getElementById('content').style.display = 'block'; // Show content
      })
      .catch((error) => {
        if (error.response) {
          // The request was made and the server responded with a status code
          console.error('Error data:', error.response.data);
          console.error('Error status:', error.response.status);
          console.error('Error headers:', error.response.headers);
        } else if (error.request) {
          // The request was made but no response was received
          console.error('Error request:', error.request);
          console.log('Retrying in 1 second...');
          setTimeout(fetchData, 1000); // Retry after 1 second
        } else {
          // Something else caused the error
          console.error('Error message:', error.message);
        }
        console.error('Error config:', error.config);
      });
  };

  fetchData();
});
