# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bot script
COPY live_xrp_bot.py .

# Command to run the bot
CMD ["python", "live_xrp_bot.py"]