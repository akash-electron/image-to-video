# Image to Video API

Convert images to videos with smooth transitions and upload to S3.

## Features

- Convert 1-5 images to video with Fade Zoom In transition
- Automatic 5-second minimum duration for 1-2 images
- Upload directly to AWS S3
- Optimized for performance (720p max, 24 FPS)
- Fast processing with automatic image resizing

## Quick Start

### Using Docker

1. **Build the image:**
   ```bash
   docker build -t image-to-video-api .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     -p 8000:8000 \
     -e S3_REGION=us-east-1 \
     -e S3_ACCESS_KEY=your_access_key \
     -e S3_SECRET_KEY=your_secret_key \
     -e S3_BUCKET_NAME=your_bucket \
     --name video-api \
     image-to-video-api
   ```

### Using Docker Compose

1. **Configure environment variables in `.env`:**
   ```
   S3_REGION=us-east-1
   S3_ACCESS_KEY=your_access_key_here
   S3_SECRET_KEY=your_secret_key_here
   S3_BUCKET_NAME=your_bucket_name_here
   ```

2. **Start the service:**
   ```bash
   docker-compose up -d
   ```

3. **Check status:**
   ```bash
   docker-compose ps
   docker-compose logs -f
   ```

4. **Stop the service:**
   ```bash
   docker-compose down
   ```

## API Endpoints

### POST /create-video
Create video from image URLs.

**Request:**
```json
{
  "image_urls": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
  ],
  "propertyId": "EP78869746"
}
```

**Response:**
```json
{
  "success": true,
  "video_url": "https://bucket.s3.region.amazonaws.com/Property/Property_videos/generated-videos/EasyPost_EP78869746_0.mp4",
  "metadata": {
    "duration": 5.0,
    "image_count": 2,
    "transition": "Fade Zoom In",
    "fps": 24,
    "width": 1280,
    "height": 720
  }
}
```

### GET /transitions
List available transitions.

### GET /health
Health check endpoint.

### GET /docs
Interactive API documentation (Swagger UI).

## Deployment Options

### 1. Docker Hub
```bash
# Tag and push
docker tag image-to-video-api yourusername/image-to-video-api:latest
docker push yourusername/image-to-video-api:latest

# Pull and run on any server
docker pull yourusername/image-to-video-api:latest
docker run -d -p 8000:8000 --env-file .env yourusername/image-to-video-api:latest
```

### 2. AWS ECS/Fargate
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ECR_URL

# Tag and push
docker tag image-to-video-api:latest YOUR_ECR_URL/image-to-video-api:latest
docker push YOUR_ECR_URL/image-to-video-api:latest
```

### 3. Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/image-to-video-api
gcloud run deploy --image gcr.io/PROJECT_ID/image-to-video-api --platform managed
```

### 4. Fly.io
```bash
# Install flyctl and login
fly launch
fly deploy
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| S3_REGION | AWS region | us-east-1 |
| S3_ACCESS_KEY | AWS access key | - |
| S3_SECRET_KEY | AWS secret key | - |
| S3_BUCKET_NAME | S3 bucket name | - |

### Default Settings

- **FPS**: 24
- **Image Duration**: 1.5 seconds (adjusted to 5s minimum for 1-2 images)
- **Transition Duration**: 1.0 seconds
- **Max Resolution**: 1280x720
- **Transition**: Fade Zoom In

## Production Considerations

1. **Reverse Proxy**: Use Nginx or Traefik
2. **HTTPS**: Add SSL certificate
3. **Rate Limiting**: Implement API rate limits
4. **Monitoring**: Add logging and metrics
5. **Auto-scaling**: Configure based on load

## License

MIT