# Face Blur

## Description

Project to blur the faces of minors in images. Uses two AI models to detect the faces and guess the age of the person in
the image. If the person is a minor, the face is blurred. Project developed using Python and built as a distributed
system using Docker. Developed by [JulianBSL](https://github.com/TehWeifu) for Programación de Inteligencia Artificial
at CIPFP Mislata.

## Project Structure

In this section, there is a brief explanation of the project structure.

- AiDetectMinorTraining: Source code to train the AI model. Also, some utility scripts to prepare the dataset.
- ApiAi: Source and deployment files for the API which the logic that detects the faces and the age of the person in the
  image.
- ApiGateway: Source and deployment files for the API Gateway which is the entry point for the system.
- frontend: Source for a simple frontend to test the system.
- http: Sample HTTP requests to test the system.
- Instructions: guidelines, requirements, and instructions to achieve in the project.
- nginx: Configuration files for the Nginx server.
- tests: Unit tests for the project.
- docker-compose.yml: File to deploy the system using Docker Compose.
- requirements.txt: Python dependencies for the project.

## Installation and Deployment

In this section, there is a brief explanation of how to deploy the system using Docker Compose.

1. Clone the repository.
2. Install Docker and Docker Compose.
3. Run the following command in the root of the project:

```bash
docker-compose up -d
```

4. The system is now running. You can access the frontend at `http://localhost:80`.
5. To stop the system, run the following command:

```bash
docker-compose down
```

## Training the AI Model

In this section, there is a brief explanation of how to train the AI model.

1. Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

2. Prepare the datasets. You can use the scripts in the `AiDetectMinorTraining` folder according to your needs.
3. Train the model by running the notebook `DetectMinorTraining.ipynb`.

## API Usage

### 1. Health Check (`/hc`)

- **Method**: GET
- **Description**: Returns the API Gateway server status, along with the uptime since the server.
- **Parameters**: None
- **Response Format**: JSON object with keys:
    - status: Server status
    - uptime: seconds since the server was launched

- **Examples**
    - Success response:
        ```json
        {
            "status": "ok",
            "uptime": "4928.41"
        }
        ```

### 2. Health Check (`/hc_model`)

- **Method**: GET
- **Description**: Returns the AI service server status, along with the uptime since the server.
- **Parameters**: None
- **Response Format**: JSON object with keys:
    - status: Server status
    - uptime: seconds since the server was launched

- **Examples**
    - Success response:
        ```json
        {
            "status": "ok",
            "uptime": "420.69"
        }
        ```

### 3. Blur (`/blur`)

- **Method**: POST
- **Description**: Blurs the face of people in a picture. According to the parameters given
- **Parameters**:
    - Query parameters
        - mode: Determines the targets of the blurring. Can be `all`, `minors`, or `random`. 'minors' is default
        - type: Determines the type of blurring. Can be `blur`, `pixelate` or `emoji`. 'pixelate' is default
        - debug: Adds debugging information to the image. Can be `on` or `off`. 'off' is default
    - Form data
        - image: Image with a .jpg or .jpeg, .png extension
- **Response Format**: Image in .jpeg format

- **Examples**
    - Sample request: `http://localhost:80/blur?mode=minors&type=pixelate&debug=on`
    - Success Response:
        - Image with the faces blurred
