#!/usr/bin/env bash

## Complete the following steps to get Docker running locally

# Step 1:
# Create dockerpath
dockerpath=herbehordeun/driver_scorecard_model

# step 2: pull the image
docker pull $dockerpath:main

# Step 3: list the images
docker images

# Step 4: run the container
docker run -d -p 8021:8000 --name driver_scorecard_model $dockerpath:main

# step 5: check the running container
docker ps -a
