#!/bin/bash
# This script stops and removes the previously running container
# (if one exists) before the new version is deployed.

if [ "$(docker ps -aq -f name=yt-sentiment-container)" ]; then
    echo "Stopping existing container..."
    docker stop yt-sentiment-container
    docker rm yt-sentiment-container
else
    echo "No existing container found, nothing to stop."
fi
