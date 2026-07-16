#!/bin/bash
# This script runs automatically the first time the EC2 server boots up.
# It installs Docker (to run our containerized app) and the CodeDeploy
# agent (so AWS CodeDeploy can push new deployments to this server).

# Install Docker
dnf install -y docker
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user

# Install the CodeDeploy agent (needed for automated deployments later)
dnf install -y ruby wget
cd /home/ec2-user
wget https://aws-codedeploy-ap-south-1.s3.ap-south-1.amazonaws.com/latest/install
chmod +x ./install
./install auto

# Enable the agent so it survives reboots. On this AMI, "systemctl enable"
# alone does not work reliably because systemd-sysv-install is missing,
# so we create the multi-user.target symlink manually instead.
ln -sf /usr/lib/systemd/system/codedeploy-agent.service /etc/systemd/system/multi-user.target.wants/codedeploy-agent.service
systemctl daemon-reload

systemctl start codedeploy-agent
systemctl enable codedeploy-agent
