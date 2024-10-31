# Dockerfile
# Use an official Python image as a base image
FROM python:3.12
# Set the working directory in the container
WORKDIR /code

# debugging: winpty docker exec -it 35a50ad44d68f2b8fed19614ff1bf36c4916b4562c3b203632a0a9110fc7eb91 //bin//sh  # or //bash?

# Copy the necessary files to the container
COPY . /code

# Install pipenv for managing the virtual environment
RUN python -m venv /env && \
    . /env/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install -e .

# Expose port for Jupyter
EXPOSE 8888

# Run the container in an interactive shell (optional)
CMD ["sleep", "infinity"]
