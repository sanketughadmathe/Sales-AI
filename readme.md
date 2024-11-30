# Sales Assistant

A RAG-powered Sales Assistant tool that helps sales professionals quickly access and understand financial information from documents.

## Features

- Document analysis using ColQwen2
- Efficient document retrieval with Vespa vector database
- Intelligent response generation using Claude AI
- User-friendly web interface
- Secure authentication system

## Prerequisites

- Python 3.8 or higher
- Node.js and npm (optional, for development)
- Vespa Cloud account
- Anthropic API key

## Project Structure

```
Sales-AI/
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
├── backend/
│   ├── main.py          # FastAPI main application
│   ├── app/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── chat.py  # Chat endpoint implementation
├── frontend/
│   ├── index.html       # Login page
│   ├── home.html        # Home dashboard
│   ├── chat.html        # Chat interface
│   ├── css/
│   │   └── ...         # Stylesheets
│   └── js/
│       └── ...         # JavaScript files
└── keys/                # Vespa certificates
    ├── data-plane-public-cert.pem
    └── data-plane-private-key.pem
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Sales-AI.git
cd Sales-AI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file in the root directory with your credentials:
```env
VESPA_URL=your_vespa_url
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Running the Application

1. Start the backend server (from the backend directory):
```bash
cd backend
uvicorn main:app --reload
```
The backend will be available at `http://localhost:8000`

2. Start the frontend server (from the frontend directory in a new terminal):
```bash
cd frontend
python3 -m http.server 8001
```
The frontend will be available at `http://localhost:8001`

3. Access the application:
Open your browser and navigate to `http://localhost:8001`

## Test Credentials

For testing purposes, you can use these credentials:
- Email: test@example.com
- Password: test123

## API Endpoints

- `POST /api/chat`: Send queries to the RAG system
- `GET /api/test-vespa`: Test Vespa connection

## Technologies Used

- FastAPI
- ColQwen2
- Vespa Cloud
- Claude AI
- HTML/CSS/JavaScript

## Development

To run the application in development mode:

1. Backend (with auto-reload):
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Frontend (with live server):
```bash
cd frontend
python3 -m http.server 8001
```

## Troubleshooting

1. If you get certificate errors:
   - Verify your Vespa credentials
   - Check certificate paths in the `.env` file

2. If the frontend can't connect to the backend:
   - Ensure both servers are running
   - Check CORS settings in `main.py`
   - Verify API endpoint URLs in frontend code

3. Common issues:
   - Port already in use: Change the port numbers
   - Module not found: Verify all dependencies are installed
   - Authentication errors: Check your API keys

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Your Name - youremail@example.com
Project Link: https://github.com/yourusername/Sales-AI

## Acknowledgments

- ColQwen2 for document understanding
- Vespa Cloud for vector search
- Anthropic for Claude AI
- FastAPI community