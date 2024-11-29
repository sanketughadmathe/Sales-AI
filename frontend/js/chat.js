document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chatMessages');
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const logoutBtn = document.getElementById('logoutBtn');

    // Check authentication
    const token = localStorage.getItem('token');
    if (!token) {
        window.location.replace('/index.html');
        return;
    }

    // Auto-resize textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = chatInput.scrollHeight + 'px';
    });

    // Handle form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = chatInput.value.trim();
        if (!message) return;

        // Add user message to chat
        addMessage(message, 'user');
        
        // Clear input
        chatInput.value = '';
        chatInput.style.height = 'auto';

        // Show loading state
        const loadingMessage = addLoadingMessage();

        try {
            const response = await fetch('http://localhost:8000/chat/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ query: message })
            });
        
            const data = await response.json();
            
            // Remove loading message
            loadingMessage.remove();
        
            if (data.results && data.results.length > 0) {
                data.results.forEach(result => {
                    const messageText = `${result.text}\nConfidence: ${(result.confidence * 100).toFixed(2)}%`;
                    addMessage(messageText, 'assistant', result.image);
                });
                
                // Add query time information
                addMessage(`Query processed in ${data.query_time} seconds`, 'system');
            } else {
                addMessage('I couldn\'t find relevant information in the documents.', 'assistant');
            }
        
        } catch (error) {
            console.error('Error:', error);
            loadingMessage.remove();
            addMessage('Sorry, there was an error processing your request.', 'system');
        }
    });

    // Handle logout
    logoutBtn.addEventListener('click', () => {
        localStorage.removeItem('token');
        window.location.replace('/index.html');
    });

    // Utility functions
    function addMessage(text, type, image = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        messageDiv.textContent = text;

        if (image) {
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${image}`;
            img.className = 'result-image';
            messageDiv.appendChild(img);
        }

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return messageDiv;
    }

    function addLoadingMessage() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant loading-dots';
        loadingDiv.innerHTML = `
            <span></span>
            <span></span>
            <span></span>
        `;
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return loadingDiv;
    }
});