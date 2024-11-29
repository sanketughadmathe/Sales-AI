// // frontend/assets/js/auth.js
// document.addEventListener('DOMContentLoaded', () => {
//     // Check if already authenticated
//     const token = localStorage.getItem('token');
//     if (token && window.location.pathname === '/index.html') {
//         window.location.replace('/home.html');
//         return;
//     }

//     // Get DOM elements
//     const loginSection = document.getElementById('loginSection');
//     const signupSection = document.getElementById('signupSection');
//     const loginForm = document.getElementById('loginForm');
//     const signupForm = document.getElementById('signupForm');
//     const showSignupLink = document.getElementById('showSignup');
//     const showLoginLink = document.getElementById('showLogin');

//     // Create error message elements
//     const loginError = createMessageElement('error-message');
//     const signupError = createMessageElement('error-message');
//     const signupSuccess = createMessageElement('success-message');
    
//     loginForm.appendChild(loginError);
//     signupForm.appendChild(signupError);
//     signupForm.appendChild(signupSuccess);

//     // Toggle between login and signup forms
//     showSignupLink.addEventListener('click', (e) => {
//         e.preventDefault();
//         loginSection.style.display = 'none';
//         signupSection.style.display = 'block';
//         clearMessages();
//     });

//     showLoginLink.addEventListener('click', (e) => {
//         e.preventDefault();
//         signupSection.style.display = 'none';
//         loginSection.style.display = 'block';
//         clearMessages();
//     });

//     // Handle Login
//     loginForm.addEventListener('submit', async (e) => {
//         e.preventDefault();
//         clearMessages();
        
//         const email = document.getElementById('loginEmail').value;
//         const password = document.getElementById('loginPassword').value;
//         const submitButton = loginForm.querySelector('button');
        
//         setLoading(submitButton, true);
        
//         try {
//             const response = await fetch('http://localhost:8000/auth/login', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json',
//                 },
//                 body: JSON.stringify({ email, password }),
//             });
            
//             const data = await response.json();
            
//             if (response.ok) {
//                 localStorage.setItem('token', data.access_token);
//                 window.location.replace('/home.html');
//             } else {
//                 showMessage(loginError, data.detail || 'Login failed. Please try again.');
//             }
//         } catch (error) {
//             showMessage(loginError, 'An error occurred during login. Please try again.');
//         } finally {
//             setLoading(submitButton, false);
//         }
//     });

//     // Handle Signup
//     signupForm.addEventListener('submit', async (e) => {
//         e.preventDefault();
//         clearMessages();

//         const name = document.getElementById('signupName').value;
//         const email = document.getElementById('signupEmail').value;
//         const password = document.getElementById('signupPassword').value;
//         const confirmPassword = document.getElementById('confirmPassword').value;
//         const submitButton = signupForm.querySelector('button');

//         // Validate passwords match
//         if (password !== confirmPassword) {
//             showMessage(signupError, 'Passwords do not match');
//             return;
//         }

//         setLoading(submitButton, true);

//         try {
//             const response = await fetch('http://localhost:8000/auth/signup', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json',
//                 },
//                 body: JSON.stringify({ name, email, password }),
//             });
            
//             const data = await response.json();
            
//             if (response.ok) {
//                 showMessage(signupSuccess, 'Account created successfully! Please login.');
//                 setTimeout(() => {
//                     signupSection.style.display = 'none';
//                     loginSection.style.display = 'block';
//                     signupForm.reset();
//                 }, 2000);
//             } else {
//                 showMessage(signupError, data.detail || 'Signup failed. Please try again.');
//             }
//         } catch (error) {
//             showMessage(signupError, 'An error occurred during signup. Please try again.');
//         } finally {
//             setLoading(submitButton, false);
//         }
//     });

//     // Utility functions
//     function createMessageElement(className) {
//         const element = document.createElement('div');
//         element.className = className;
//         return element;
//     }

//     function showMessage(element, message) {
//         element.textContent = message;
//         element.style.display = 'block';
//     }

//     function clearMessages() {
//         const messages = document.querySelectorAll('.error-message, .success-message');
//         messages.forEach(msg => msg.style.display = 'none');
//     }

//     function setLoading(button, isLoading) {
//         if (isLoading) {
//             button.classList.add('loading');
//             button.disabled = true;
//         } else {
//             button.classList.remove('loading');
//             button.disabled = false;
//         }
//     }
// });


document.addEventListener('DOMContentLoaded', () => {
    // Dummy users database
    const dummyUsers = [
        {
            name: "John Doe",
            email: "john@example.com",
            password: "password123"
        },
        {
            name: "Test User",
            email: "test@example.com",
            password: "test123"
        },
        {
            name: "Test User",
            email: "user@ex.com",
            password: "123456"
        },
    ];

    // Check if already authenticated
    const token = localStorage.getItem('token');
    if (token && window.location.pathname === '/index.html') {
        window.location.replace('/home.html');
        return;
    }

    // Get DOM elements
    const loginSection = document.getElementById('loginSection');
    const signupSection = document.getElementById('signupSection');
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
    const showSignupLink = document.getElementById('showSignup');
    const showLoginLink = document.getElementById('showLogin');

    // Create error message elements
    const loginError = createMessageElement('error-message');
    const signupError = createMessageElement('error-message');
    const signupSuccess = createMessageElement('success-message');
    
    loginForm.appendChild(loginError);
    signupForm.appendChild(signupError);
    signupForm.appendChild(signupSuccess);

    // Toggle between login and signup forms
    showSignupLink.addEventListener('click', (e) => {
        e.preventDefault();
        loginSection.style.display = 'none';
        signupSection.style.display = 'block';
        clearMessages();
    });

    showLoginLink.addEventListener('click', (e) => {
        e.preventDefault();
        signupSection.style.display = 'none';
        loginSection.style.display = 'block';
        clearMessages();
    });

    // Handle Login
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        clearMessages();
        
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;
        const submitButton = loginForm.querySelector('button');
        
        setLoading(submitButton, true);
        
        try {
            // Simulate API delay
            await new Promise(resolve => setTimeout(resolve, 1000));

            // Check credentials against dummy users
            const user = dummyUsers.find(u => u.email === email && u.password === password);

            if (user) {
                // Generate a dummy token
                const dummyToken = btoa(JSON.stringify({ email: user.email, name: user.name }));
                localStorage.setItem('token', dummyToken);
                localStorage.setItem('userName', user.name); // Store user name for display
                window.location.replace('/home.html');
            } else {
                showMessage(loginError, 'Invalid email or password');
            }
        } catch (error) {
            showMessage(loginError, 'An error occurred during login. Please try again.');
        } finally {
            setLoading(submitButton, false);
        }
    });

    // Handle Signup
    signupForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        clearMessages();

        const name = document.getElementById('signupName').value;
        const email = document.getElementById('signupEmail').value;
        const password = document.getElementById('signupPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        const submitButton = signupForm.querySelector('button');

        // Validate passwords match
        if (password !== confirmPassword) {
            showMessage(signupError, 'Passwords do not match');
            return;
        }

        setLoading(submitButton, true);

        try {
            // Simulate API delay
            await new Promise(resolve => setTimeout(resolve, 1000));

            // Check if email already exists
            if (dummyUsers.some(user => user.email === email)) {
                showMessage(signupError, 'Email already registered');
                return;
            }

            // Simulate successful registration
            dummyUsers.push({ name, email, password });
            
            showMessage(signupSuccess, 'Account created successfully! Please login.');
            setTimeout(() => {
                signupSection.style.display = 'none';
                loginSection.style.display = 'block';
                signupForm.reset();
            }, 2000);
        } catch (error) {
            showMessage(signupError, 'An error occurred during signup. Please try again.');
        } finally {
            setLoading(submitButton, false);
        }
    });

    // Utility functions
    function createMessageElement(className) {
        const element = document.createElement('div');
        element.className = className;
        return element;
    }

    function showMessage(element, message) {
        element.textContent = message;
        element.style.display = 'block';
    }

    function clearMessages() {
        const messages = document.querySelectorAll('.error-message, .success-message');
        messages.forEach(msg => msg.style.display = 'none');
    }

    function setLoading(button, isLoading) {
        if (isLoading) {
            button.classList.add('loading');
            button.disabled = true;
        } else {
            button.classList.remove('loading');
            button.disabled = false;
        }
    }
});