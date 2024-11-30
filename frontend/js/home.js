// // frontend/assets/js/home.js
// document.addEventListener('DOMContentLoaded', () => {
//     // Check if user is authenticated
//     const token = localStorage.getItem('token');
//     if (!token && window.location.pathname === '/home.html') {
//         window.location.replace('/index.html');
//         return;
//     }
    
//     // Setup logout functionality
//     const logoutBtn = document.getElementById('logoutBtn');
//     logoutBtn.addEventListener('click', () => {
//         localStorage.removeItem('token');
//         window.location.replace('/index.html');
//     });
    
//     // Add click handlers for option cards
//     const text1Card = document.getElementById('text1');
//     const text2Card = document.getElementById('text2');
    
//     text1Card.addEventListener('click', () => {
//         alert('Text 1 option selected');
//         // Add your navigation or action logic here
//     });
    
//     text2Card.addEventListener('click', () => {
//         alert('Text 2 option selected');
//         // Add your navigation or action logic here
//     });

//     // Add hover effects for cards
//     const cards = document.querySelectorAll('.option-card');
//     cards.forEach(card => {
//         card.addEventListener('mouseenter', () => {
//             card.style.transform = 'translateY(-5px)';
//         });
        
//         card.addEventListener('mouseleave', () => {
//             card.style.transform = 'translateY(0)';
//         });
//     });
// });

document.addEventListener('DOMContentLoaded', () => {
    // Check if user is authenticated
    const token = localStorage.getItem('token');
    if (!token && window.location.pathname === '/home.html') {
        window.location.replace('/index.html');
        return;
    }

    // Display user name if available
    try {
        const userName = localStorage.getItem('userName');
        if (userName) {
            document.getElementById('userName').textContent = userName;
        }
    } catch (error) {
        console.error('Error displaying user name:', error);
    }
    
    // Setup logout functionality
    const logoutBtn = document.getElementById('logoutBtn');
    logoutBtn.addEventListener('click', () => {
        localStorage.removeItem('token');
        localStorage.removeItem('userName');
        window.location.replace('/index.html');
    });
    
    // Add click handlers for option cards
    const text1Card = document.getElementById('text1');
    const text2Card = document.getElementById('text2');
    
    text1Card.addEventListener('click', () => {
        // alert('Text 1 option selected');
        window.location.href = 'chat.html';
        // Add your navigation or action logic here
    });
    
    text2Card.addEventListener('click', () => {
        alert('Text 2 option selected');
        // Add your navigation or action logic here
    });

    // Add hover effects for cards
    const cards = document.querySelectorAll('.option-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-5px)';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0)';
        });
    });
});