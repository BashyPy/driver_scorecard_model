#!/usr/bin/env bash

# This tags and uploads an image to Docker Hub

#This is your Docker ID/path
dockerpath=herbehordeun/scorecard_system_fastapi


# Run the Docker Hub container with kubernetes

kubectl run scorecard_system_fastapi\
    --image=$dockerpath\
    --port=80 --labels app=scorecard_system_fastapi


# List kubernetes pods
kubectl get pods


# Forward the container port to a host
kubectl port-forward scorecard_system_fastapi-app 8000:80