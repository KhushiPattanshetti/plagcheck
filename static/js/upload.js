document.addEventListener('DOMContentLoaded', function() {
  const uploadForm = document.getElementById('upload-form');
  const statusDiv = document.getElementById('upload-status');
  const contentSection = document.getElementById('content-section');
  
  if (uploadForm) {
      uploadForm.addEventListener('submit', function(e) {
          e.preventDefault();
          
          statusDiv.className = 'status-message';
          statusDiv.textContent = 'Processing files...';
          
          const formData = new FormData();
          formData.append('train_file', document.getElementById('train-file').files[0]);
          formData.append('test_file', document.getElementById('test-file').files[0]);
          
          fetch('/upload', {
              method: 'POST',
              body: formData
          })
          .then(response => response.json())
          .then(data => {
              if (data.success) {
                  statusDiv.className = 'status-message success';
                  statusDiv.textContent = 'Files processed successfully! Loading visualizations...';
                  // Show the content section
                  contentSection.style.display = 'block';
                  // Reload the page to load all components properly
                  setTimeout(() => window.location.reload(), 1000);
              } else {
                  statusDiv.className = 'status-message error';
                  statusDiv.textContent = `Error: ${data.error}`;
              }
          })
          .catch(error => {
              statusDiv.className = 'status-message error';
              statusDiv.textContent = `Network error: ${error.message}`;
          });
      });
  }
});