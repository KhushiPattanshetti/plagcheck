// Global utility functions
function showNotification(message, isSuccess) {
  const resultDiv = document.getElementById('lookup-result');
  resultDiv.textContent = message;
  resultDiv.className = `result-message ${isSuccess ? 'success' : 'error'}`;
}

// Event listener for DOM ready
document.addEventListener('DOMContentLoaded', () => {
  // Any global initialization can go here
});