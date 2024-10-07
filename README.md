# Fetch-ML-Internship

This Flask application predicts monthly receipt counts for 2022 using a pre-trained LSTM model. The app displays actual vs. predicted receipt counts and provides a table of predicted values. The application is dockerized for easy deployment.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Clone the Repository](#clone-the-repository)
  - [Build the Docker Image](#build-the-docker-image)
  - [Run the Docker Container](#run-the-docker-container)
- [Accessing the Application](#accessing-the-application)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Prerequisites

- **Docker**: Make sure Docker is installed on your machine. You can download it from [Docker's official website](https://www.docker.com/get-started).

- **Python 3.9.6**: Ensure Python 3.9.6 is installed on your machine. You can download it from [Python's official website](https://www.python.org/downloads/release/python-396/).

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/htripathi12/Fetch-ML-Internship.git
cd Fetch-ML-Internship
```

### Running Docker
You need to have Docker Desktop downloaded and running on your machine. You can download it from Docker's official website [here](https://www.docker.com/products/docker-desktop).
```

In your terminal, navigate to the project directory and run:

```bash
docker build -t receipt-prediction-app .
```

This command builds a Docker image named `receipt-prediction-app` using the `Dockerfile` in the current directory.

## Run the Docker Container

To run the Docker container and map the container's port to your host machine, use:

```bash
docker run -p 3000:3000 receipt-prediction-app
```

- `-p 3000:3000`: Maps port `3000` inside the container to port `3000` on your host machine.
- `receipt-prediction-app`: The name of the Docker image to run.

If this port is not available due to another process, you can adjust it accordingly.

## Accessing the Application

Open your web browser and navigate to:

```
http://localhost:3000
```

As mentioned previously, if you change the port in your run command, you must adjust it here as well. You should see the application's homepage displaying the predicted vs. actual monthly receipt counts, along with the associated dataframe.

## Thank you for reviewing my submission!