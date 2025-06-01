# Use a lightweight Python base image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Ensure the static/uploads directory exists (if not created by your app)
RUN mkdir -p static/uploads

# Expose the port that Flask will run on (Hugging Face Spaces uses PORT env var)
EXPOSE 7860

# Command to run your Flask application
# Hugging Face Spaces will set the PORT environment variable.
# Your app.py is already set up to use it (os.environ.get('PORT', 7860)).
CMD ["python", "app.py"]
