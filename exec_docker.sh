# Step 1:
# Create dockerpath
dockerpath=herbehordeun/driver_scorecard_model

# step 2: exec into docker container
docker exec -it $dockerpath /bin/sh

# Step 3: check the logs
docker logs driver_scorecard_model