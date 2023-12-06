# Step 1:
# Login into ECR

aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 856488756663.dkr.ecr.us-east-2.amazonaws.com

# Step 2:
# Build the image
docker build -t scorecard .

# Step 3:
#  Tag the image
docker tag scorecard:latest 856488756663.dkr.ecr.us-east-2.amazonaws.com/scorecard:latest

# Step 4:
# Push the image to ECR
docker push 856488756663.dkr.ecr.us-east-2.amazonaws.com/scorecard:latest

