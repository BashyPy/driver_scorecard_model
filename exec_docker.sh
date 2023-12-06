# Step 1:
# Create dockerpath
dockerpath=herbehordeun/scorecard_system_fastapi

# step 2: exec into docker container
docker exec -it $dockerpath /bin/sh

# Step 3: check the logs
docker logs scorecard_system_fastapi