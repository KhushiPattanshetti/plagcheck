document.addEventListener('DOMContentLoaded', () => {
  const lookupForm = document.getElementById('probability-lookup');
  
  if (lookupForm) {
      lookupForm.addEventListener('submit', handleLookupSubmit);
  }
});

function handleLookupSubmit(e) {
  e.preventDefault();
  
  const wordInput = document.getElementById('word');
  const contextInput = document.getElementById('context');
  
  const word = wordInput.value.trim();
  const context = contextInput.value.trim();
  
  if (!word || !context) {
      showNotification('Please enter both a word and context', false);
      return;
  }
  
  fetch('/lookup', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify({ word: word, context: context })
  })
  .then(response => {
      if (!response.ok) {
          throw new Error('Network response was not ok');
      }
      return response.json();
  })
  .then(data => {
      if (data.success) {
          showNotification(
              `P(${data.result.word} | ${data.result.context}) = ${data.result.probability}`, 
              true
          );
      } else {
          showNotification(`Error: ${data.error}`, false);
      }
  })
  .catch(error => {
      showNotification(`Network error: ${error.message}`, false);
  });
}