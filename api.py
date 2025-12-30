from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from typing import List
from pydantic import BaseModel, HttpUrl, field_validator
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from transitions import TRANSITIONS
import io
import shutil
import requests
import subprocess
from s3_upload import s3_uploader

app = FastAPI(
    title="Image to Video API",
    description="API to create videos from images with various transition effects",
    version="2.0.0"
)


class VideoRequest(BaseModel):
    image_urls: List[HttpUrl]
    propertyId: str
    transition: str = "Fade Zoom In"
    fps: int = 24
    image_duration: float = 1.5
    transition_duration: float = 1.0
    upload_to_s3: bool = True

    @field_validator('image_urls')
    @classmethod
    def validate_image_urls(cls, v):
        if len(v) < 1:
            raise ValueError('At least 1 image URL is required')
        if len(v) > 5:
            raise ValueError('Maximum 5 image URLs allowed')
        return v

    @field_validator('propertyId')
    @classmethod
    def validate_property_id(cls, v):
        if not v or not v.strip():
            raise ValueError('propertyId is required and cannot be empty')
        return v.strip()

    @field_validator('fps')
    @classmethod
    def validate_fps(cls, v):
        if v < 24 or v > 60:
            raise ValueError('FPS must be between 24 and 60')
        return v

    @field_validator('image_duration')
    @classmethod
    def validate_image_duration(cls, v):
        if v < 1.0 or v > 10.0:
            raise ValueError('Image duration must be between 1 and 10 seconds')
        return v

    @field_validator('transition_duration')
    @classmethod
    def validate_transition_duration(cls, v):
        if v < 0.5 or v > 3.0:
            raise ValueError('Transition duration must be between 0.5 and 3 seconds')
        return v


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Image to Video API",
        "version": "2.0.0",
        "endpoints": {
            "POST /create-video": "Create a video from image URLs (1-5 images)",
            "POST /create-video-upload": "Create a video from uploaded files (2+ images)",
            "GET /transitions": "List available transition effects",
            "GET /docs": "API documentation"
        }
    }


@app.get("/transitions")
def get_transitions():
    """Get list of available transition effects"""
    return {
        "transitions": list(TRANSITIONS.keys()),
        "count": len(TRANSITIONS)
    }


@app.post("/create-video")
async def create_video(request: VideoRequest):
    """
    Create a video from image URLs with transition effects

    - **image_urls**: List of 1-5 image URLs (JPG, PNG)
    - **transition**: Name of transition effect (use /transitions endpoint to see options)
    - **fps**: Frames per second (24-60)
    - **image_duration**: How long each image stays visible (1-10 seconds)
    - **transition_duration**: How long the transition lasts (0.5-3 seconds)

    Returns: MP4 video file
    """

    # Validate transition name
    if request.transition not in TRANSITIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid transition '{request.transition}'. Use /transitions endpoint to see available options."
        )

    # Calculate frame counts
    image_frames = int(request.fps * request.image_duration)
    transition_frames = int(request.fps * request.transition_duration)

    # Ensure minimum 5 second video for 1 or 2 images
    num_images = len(request.image_urls)
    if num_images <= 2:
        MIN_DURATION = 5.0  # 5 seconds minimum
        if num_images == 1:
            current_duration = request.image_duration
        else:  # 2 images
            current_duration = (2 * request.image_duration) + request.transition_duration

        if current_duration < MIN_DURATION:
            # Adjust image_duration to meet minimum duration
            if num_images == 1:
                adjusted_image_duration = MIN_DURATION
            else:  # 2 images
                adjusted_image_duration = (MIN_DURATION - request.transition_duration) / 2

            image_frames = int(request.fps * adjusted_image_duration)

    try:
        # Download and process images from URLs
        processed_images = []
        for idx, image_url in enumerate(request.image_urls):
            try:
                # Download image
                response = requests.get(str(image_url), timeout=10)
                response.raise_for_status()

                # Convert to PIL Image
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                processed_images.append(img)
            except requests.RequestException as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download image {idx + 1} from URL: {str(e)}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process image {idx + 1}: {str(e)}"
                )

        # Resize all images to same dimensions with max resolution for faster processing
        MAX_WIDTH = 1280
        MAX_HEIGHT = 720

        # Find minimum dimensions across all images
        min_h = min(img.shape[0] for img in processed_images)
        min_w = min(img.shape[1] for img in processed_images)

        # Cap to max resolution for performance
        if min_w > MAX_WIDTH or min_h > MAX_HEIGHT:
            aspect_ratio = min_w / min_h
            if aspect_ratio > MAX_WIDTH / MAX_HEIGHT:
                min_w = MAX_WIDTH
                min_h = int(MAX_WIDTH / aspect_ratio)
            else:
                min_h = MAX_HEIGHT
                min_w = int(MAX_HEIGHT * aspect_ratio)

        processed_images = [cv2.resize(img, (min_w, min_h)) for img in processed_images]

        h, w, _ = processed_images[0].shape

        # Create temporary file for output video
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video_path = temp_video.name
        temp_video.close()

        # Initialize video writer with codec fallback
        # Try multiple codecs in order of preference
        codecs_to_try = [
            ("avc1", "H.264 (avc1)"),      # Best quality, most compatible
            ("X264", "H.264 (X264)"),      # Alternative H.264
            ("mp4v", "MPEG-4"),            # Fallback codec
            ("MP4V", "MPEG-4 uppercase"),  # Case variation
        ]

        video = None
        successful_codec = None

        for codec, codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video = cv2.VideoWriter(temp_video_path, fourcc, request.fps, (w, h))

                if video.isOpened():
                    successful_codec = codec_name
                    print(f"✓ Video writer initialized successfully with {codec_name}")
                    break
                else:
                    video.release()
                    video = None
            except Exception as e:
                print(f"✗ Failed to initialize with {codec_name}: {str(e)}")
                if video:
                    video.release()
                    video = None
                continue

        if not video or not video.isOpened():
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize video writer with any codec. Please check FFmpeg installation."
            )

        # Get transition function
        transition_func = TRANSITIONS[request.transition]

        # Helper function for Ken Burns zoom effect
        def apply_ken_burns(img, frames, zoom_amount=0.15):
            """Apply Ken Burns slow zoom effect to an image"""
            h, w, _ = img.shape
            output = []
            for i in range(frames):
                progress = i / frames
                scale = 1.0 + zoom_amount * progress
                scaled_h, scaled_w = int(h * scale), int(w * scale)
                zoomed_img = cv2.resize(img, (scaled_w, scaled_h))
                crop_y = (scaled_h - h) // 2
                crop_x = (scaled_w - w) // 2
                cropped = zoomed_img[crop_y:crop_y+h, crop_x:crop_x+w]
                output.append(cropped)
            return output

        # Create video
        total_images = len(processed_images)

        # Check if using zoom effect
        use_zoom = request.transition == "Fade Zoom In"

        # If only 1 image, create a static video
        if total_images == 1:
            if use_zoom:
                # Apply Ken Burns zoom effect
                zoomed_frames = apply_ken_burns(processed_images[0], image_frames)
                for frame in zoomed_frames:
                    video.write(frame)
            else:
                for _ in range(image_frames):
                    video.write(processed_images[0])
        else:
            # Multiple images with transitions
            for i in range(total_images - 1):
                # Write static image frames with optional zoom
                if use_zoom:
                    # Apply Ken Burns zoom effect to each static image
                    zoomed_frames = apply_ken_burns(processed_images[i], image_frames)
                    for frame in zoomed_frames:
                        video.write(frame)
                else:
                    for _ in range(image_frames):
                        video.write(processed_images[i])

                # Generate and write transition frames
                # For zoom transition, pass original images - it will handle zoom itself
                frames = transition_func(
                    processed_images[i],      # Original image
                    processed_images[i + 1],  # Original next image
                    frames=transition_frames
                )

                for f in frames:
                    frame_uint8 = np.clip(f, 0, 255).astype(np.uint8)
                    video.write(frame_uint8)

            # Write final image frames
            if use_zoom:
                zoomed_frames = apply_ken_burns(processed_images[-1], image_frames)
                for frame in zoomed_frames:
                    video.write(frame)
            else:
                for _ in range(image_frames):
                    video.write(processed_images[-1])

        # Release video writer
        video.release()

        # Re-encode with FFmpeg to ensure proper metadata (fix for Docker MPEG-4 codec issue)
        # This ensures duration and other metadata are correctly written
        temp_reencoded = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_reencoded_path = temp_reencoded.name
        temp_reencoded.close()

        try:
            # Use FFmpeg to re-encode with H.264 and ensure proper metadata
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', temp_video_path,
                '-c:v', 'libx264',           # Use H.264 codec
                '-preset', 'fast',            # Fast encoding
                '-crf', '23',                 # Good quality
                '-movflags', '+faststart',    # Optimize for streaming
                '-y',                         # Overwrite output file
                temp_reencoded_path
            ]

            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120  # 2 minutes timeout
            )

            if result.returncode == 0:
                # Replace original with re-encoded version
                os.replace(temp_reencoded_path, temp_video_path)
                print("✓ Video re-encoded with FFmpeg for proper metadata")
            else:
                # If FFmpeg fails, log but continue with original file
                print(f"⚠ FFmpeg re-encoding failed (using original): {result.stderr.decode()}")
                if os.path.exists(temp_reencoded_path):
                    os.unlink(temp_reencoded_path)

        except subprocess.TimeoutExpired:
            print("⚠ FFmpeg re-encoding timed out (using original)")
            if os.path.exists(temp_reencoded_path):
                os.unlink(temp_reencoded_path)
        except Exception as e:
            print(f"⚠ FFmpeg re-encoding error (using original): {str(e)}")
            if os.path.exists(temp_reencoded_path):
                os.unlink(temp_reencoded_path)

        # Calculate video metadata (using actual frame counts for accurate duration)
        num_images = len(processed_images)
        actual_image_duration = image_frames / request.fps
        actual_transition_duration = transition_frames / request.fps

        if num_images == 1:
            total_duration = actual_image_duration
        else:
            total_duration = (num_images * actual_image_duration) + ((num_images - 1) * actual_transition_duration)

        # Upload to S3 if requested
        if request.upload_to_s3:
            try:
                # Generate filename using propertyId: EasyPost_{propertyId}_0.mp4
                filename = f"EasyPost_{request.propertyId}_0.mp4"
                s3_url = await s3_uploader.upload_video_to_s3(temp_video_path, filename)

                # Clean up temp file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)

                # Return S3 URL and metadata
                return JSONResponse(content={
                    "success": True,
                    "video_url": s3_url,
                    "metadata": {
                        "duration": total_duration,
                        "image_count": num_images,
                        "transition": request.transition,
                        "fps": request.fps,
                        "width": w,
                        "height": h
                    }
                })
            except Exception as e:
                # Clean up temp file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload video to S3: {str(e)}"
                )
        else:
            # Return the video file directly
            return FileResponse(
                path=temp_video_path,
                media_type="video/mp4",
                filename=f"video_{request.transition.lower().replace(' ', '_')}.mp4",
                headers={
                    "X-Video-Duration": str(total_duration),
                    "X-Image-Count": str(num_images),
                    "X-Transition": request.transition,
                    "X-FPS": str(request.fps)
                },
                background=lambda: os.unlink(temp_video_path) if os.path.exists(temp_video_path) else None
            )

    except HTTPException:
        raise
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

        raise HTTPException(
            status_code=500,
            detail=f"Error creating video: {str(e)}"
        )


@app.post("/create-video-upload")
async def create_video_upload(
    images: List[UploadFile] = File(..., description="Upload 2 or more images in order"),
    transition: str = Form(default="Fade Zoom In", description="Transition effect name"),
    fps: int = Form(default=30, ge=24, le=60, description="Frames per second (24-60)"),
    image_duration: float = Form(default=3.0, ge=1.0, le=10.0, description="Duration each image stays on screen (seconds)"),
    transition_duration: float = Form(default=1.0, ge=0.5, le=3.0, description="Duration of transition effect (seconds)"),
    upload_to_s3: bool = Form(default=True, description="Upload video to S3"),
    custom_filename: str = Form(default=None, description="Custom filename for S3 upload")
):
    """
    Create a video from uploaded image files with transition effects

    - **images**: Upload 2 or more images (JPG, PNG)
    - **transition**: Name of transition effect (use /transitions endpoint to see options)
    - **fps**: Frames per second (24-60)
    - **image_duration**: How long each image stays visible (1-10 seconds)
    - **transition_duration**: How long the transition lasts (0.5-3 seconds)

    Returns: MP4 video file
    """

    # Validate number of images
    if len(images) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 images are required to create a video"
        )

    # Validate transition name
    if transition not in TRANSITIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid transition '{transition}'. Use /transitions endpoint to see available options."
        )

    # Calculate frame counts
    image_frames = int(fps * image_duration)
    transition_frames = int(fps * transition_duration)

    try:
        # Load and process images
        processed_images = []
        for img_file in images:
            # Read image bytes
            contents = await img_file.read()

            # Convert to PIL Image
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            processed_images.append(img)

        # Resize all images to the same dimensions (minimum dimensions)
        min_h = min(img.shape[0] for img in processed_images)
        min_w = min(img.shape[1] for img in processed_images)
        processed_images = [cv2.resize(img, (min_w, min_h)) for img in processed_images]

        h, w, _ = processed_images[0].shape

        # Create temporary file for output video
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video_path = temp_video.name
        temp_video.close()

        # Initialize video writer with codec fallback
        # Try multiple codecs in order of preference
        codecs_to_try = [
            ("avc1", "H.264 (avc1)"),      # Best quality, most compatible
            ("X264", "H.264 (X264)"),      # Alternative H.264
            ("mp4v", "MPEG-4"),            # Fallback codec
            ("MP4V", "MPEG-4 uppercase"),  # Case variation
        ]

        video = None
        successful_codec = None

        for codec, codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video = cv2.VideoWriter(temp_video_path, fourcc, fps, (w, h))

                if video.isOpened():
                    successful_codec = codec_name
                    print(f"✓ Video writer initialized successfully with {codec_name}")
                    break
                else:
                    video.release()
                    video = None
            except Exception as e:
                print(f"✗ Failed to initialize with {codec_name}: {str(e)}")
                if video:
                    video.release()
                    video = None
                continue

        if not video or not video.isOpened():
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize video writer with any codec. Please check FFmpeg installation."
            )

        # Get transition function
        transition_func = TRANSITIONS[transition]

        # Create video
        total_images = len(processed_images)

        for i in range(total_images - 1):
            # Write static image frames
            for _ in range(image_frames):
                video.write(processed_images[i])

            # Generate and write transition frames
            frames = transition_func(
                processed_images[i],
                processed_images[i + 1],
                frames=transition_frames
            )

            for f in frames:
                frame_uint8 = np.clip(f, 0, 255).astype(np.uint8)
                video.write(frame_uint8)

        # Write final image frames
        for _ in range(image_frames):
            video.write(processed_images[-1])

        # Release video writer
        video.release()

        # Re-encode with FFmpeg to ensure proper metadata (fix for Docker MPEG-4 codec issue)
        temp_reencoded = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_reencoded_path = temp_reencoded.name
        temp_reencoded.close()

        try:
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', temp_video_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-movflags', '+faststart',
                '-y',
                temp_reencoded_path
            ]

            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120
            )

            if result.returncode == 0:
                os.replace(temp_reencoded_path, temp_video_path)
                print("✓ Video re-encoded with FFmpeg for proper metadata")
            else:
                print(f"⚠ FFmpeg re-encoding failed (using original): {result.stderr.decode()}")
                if os.path.exists(temp_reencoded_path):
                    os.unlink(temp_reencoded_path)

        except subprocess.TimeoutExpired:
            print("⚠ FFmpeg re-encoding timed out (using original)")
            if os.path.exists(temp_reencoded_path):
                os.unlink(temp_reencoded_path)
        except Exception as e:
            print(f"⚠ FFmpeg re-encoding error (using original): {str(e)}")
            if os.path.exists(temp_reencoded_path):
                os.unlink(temp_reencoded_path)

        # Calculate video metadata
        num_images = len(processed_images)
        total_duration = (num_images * image_duration) + ((num_images - 1) * transition_duration)

        # Upload to S3 if requested
        if upload_to_s3:
            try:
                # Generate custom filename or use default
                filename = custom_filename if custom_filename else None
                s3_url = await s3_uploader.upload_video_to_s3(temp_video_path, filename)

                # Clean up temp file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)

                # Return S3 URL and metadata
                return JSONResponse(content={
                    "success": True,
                    "video_url": s3_url,
                    "metadata": {
                        "duration": total_duration,
                        "image_count": num_images,
                        "transition": transition,
                        "fps": fps,
                        "width": w,
                        "height": h
                    }
                })
            except Exception as e:
                # Clean up temp file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload video to S3: {str(e)}"
                )
        else:
            # Return the video file directly
            return FileResponse(
                path=temp_video_path,
                media_type="video/mp4",
                filename=f"video_{transition.lower().replace(' ', '_')}.mp4",
                headers={
                    "X-Video-Duration": str(total_duration),
                    "X-Image-Count": str(num_images),
                    "X-Transition": transition,
                    "X-FPS": str(fps)
                },
                background=lambda: os.unlink(temp_video_path) if os.path.exists(temp_video_path) else None
            )

    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

        raise HTTPException(
            status_code=500,
            detail=f"Error creating video: {str(e)}"
        )


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
