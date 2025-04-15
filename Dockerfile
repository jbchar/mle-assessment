# Use the official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# copy the provided take home code to the working directory
COPY src .

# Install the Python dependencies (not using conda as I was having issues getting it to work)
RUN pip install -r mle_project_challenge_2/requirements.txt

# Expose the port on which the application will run
EXPOSE 8000

# train the model upon image startup
RUN python mle_project_challenge_2/create_model.py

# Run the FastAPI application using uvicorn server
CMD ["uvicorn", "model_api:app", "--host", "0.0.0.0","--log-level", "debug", "--port", "8000"]