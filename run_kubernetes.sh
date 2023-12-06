#!/usr/bin/env bash

# This tags and uploads an image to Docker Hub

#This is your Docker ID/path
dockerpath=herbehordeun/driver_scorecard_model


# Run the Docker Hub container with kubernetes

kubectl run driver_scorecard_model\
    --image=$dockerpath\
    --port=80 --labels app=driver_scorecard_model


# List kubernetes pods
kubectl get pods


# Forward the container port to a host
kubectl port-forward driver_scorecard_model-app 8000:80