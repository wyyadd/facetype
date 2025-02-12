FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy only requirements file first (for better caching)
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port FastAPI runs on
EXPOSE 80

# Run the FastAPI app using FastAPI CLI
CMD ["fastapi", "run", "main.py", "--port", "80"]
