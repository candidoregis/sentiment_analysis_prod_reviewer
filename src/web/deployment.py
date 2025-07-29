import os
import subprocess
from pathlib import Path

def create_dockerfile():
    """Create Dockerfile for deployment"""
    dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Copy model files
COPY models/* models/

# Expose port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    print("Dockerfile created successfully!")

def create_docker_compose():
    """Create docker-compose.yml for local development"""
    docker_compose_content = """
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - .:/app
      - models:/app/models
    environment:
      - PYTHONPATH=/app
    command: streamlit run app.py

volumes:
  models:
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose_content)
    print("docker-compose.yml created successfully!")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.vscode/
.idea/

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Models
models/*.pkl

# Logs
*.log

# Environment variables
.env

# Jupyter
.ipynb_checkpoints
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print(".gitignore created successfully!")

def create_readme():
    """Create README.md with deployment instructions"""
    readme_content = """# Amazon Product Sentiment Analyzer

This application analyzes Amazon product reviews using machine learning to provide sentiment analysis and shopping recommendations.

## Deployment Instructions

### Prerequisites
- Docker
- Docker Compose

### Local Development
1. Clone the repository
2. Build and run the application:
   ```bash
   docker-compose up --build
   ```
3. Access the application at http://localhost:8501

### Production Deployment
1. Build the Docker image:
   ```bash
   docker build -t amazon-sentiment-analyzer .
   ```

2. Run the container:
   ```bash
   docker run -d -p 8501:8501 amazon-sentiment-analyzer
   ```

3. Access the application at http://localhost:8501

## Project Structure
- `app.py`: Main Streamlit application
- `models/`: Contains trained ML models
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container configuration
- `docker-compose.yml`: Local development configuration

## Technologies Used
- Python
- Streamlit
- scikit-learn
- Docker
- Docker Compose
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    print("README.md created successfully!")

def main():
    print("\nSetting up deployment files...")
    
    # Create deployment files
    create_dockerfile()
    create_docker_compose()
    create_gitignore()
    create_readme()
    
    # Build and run the application
    print("\nBuilding and running the application...")
    try:
        subprocess.run(['docker-compose', 'up', '--build'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError during docker-compose: {str(e)}")
    
    print("\nDeployment setup completed!")
    print("Access the application at http://localhost:8501")

if __name__ == "__main__":
    main()
