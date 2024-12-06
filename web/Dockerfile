# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
COPY resnet_model.h5 /app
# Install the required packages
RUN pip install -r requirements.txt

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh

# Ensure the entrypoint script is executable
RUN chmod +x /entrypoint.sh

# Command to run both FastAPI and Streamlit
CMD ["/entrypoint.sh"]
