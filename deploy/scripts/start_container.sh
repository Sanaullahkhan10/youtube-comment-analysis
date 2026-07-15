#!/bin/bash
# This script logs into ECR, pulls the latest Docker image,
# and starts it as a new container. It runs every time a
# deployment happens.

# Log everything to a file so we can debug if something goes wrong
exec > /home/ec2-user/deploy.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 739022091079.dkr.ecr.ap-south-1.amazonaws.com

echo "Pulling latest image..."
docker pull 739022091079.dkr.ecr.ap-south-1.amazonaws.com/youtube-comment-analysis:latest

echo "Starting new container..."
docker run -d \
  -p 5000:5000 \
  --name yt-sentiment-container \
  --restart unless-stopped \
  739022091079.dkr.ecr.ap-south-1.amazonaws.com/youtube-comment-analysis:latest

echo "Deployment complete."
