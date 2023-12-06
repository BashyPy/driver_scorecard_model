#!/usr/bin/env bash

## Complete the following steps to get Docker running locally

# Step 1:
# Create dockerpath
dockerpath=herbehordeun/scorecard_system_fastapi

# step 2: pull the image
docker pull $dockerpath:latest

# Step 3: list the images
docker images

# Step 4: run the container
docker run -d -p 8000:8000 --name scorecard_system_fastapi $dockerpath:latest

# step 5: check the running container
docker ps -a

# step 7: exec into docker container
docker exec -it scorecard_system_fastapi /bin/sh

# Step 8: check the logs
docker logs scorecard_system_fastapi
