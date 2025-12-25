import boto3
from botocore.exceptions import ClientError
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class S3Uploader:
    """AWS S3 uploader for video files"""

    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            region_name=os.getenv('S3_REGION'),
            aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('S3_SECRET_KEY')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.region = os.getenv('S3_REGION')

    async def upload_video_to_s3(self, file_path: str, custom_file_name: str = None) -> str:
        """
        Upload a video file to S3

        Args:
            file_path: Path to the video file
            custom_file_name: Optional custom filename for S3

        Returns:
            Public URL of the uploaded video

        Raises:
            Exception: If upload fails
        """
        try:
            # Generate filename
            if custom_file_name:
                file_name = custom_file_name
            else:
                timestamp = int(datetime.now().timestamp() * 1000)
                file_name = f"property-{timestamp}.mp4"

            s3_key = f"Property/Property_videos/generated-videos/{file_name}"

            # Read file content
            with open(file_path, 'rb') as file:
                file_content = file.read()

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_content,
                ContentType='video/mp4',
                ContentDisposition='inline'
            )

            # Generate public URL
            public_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"

            print(f"✅ Video uploaded to S3: {public_url}")
            return public_url

        except ClientError as e:
            error_message = f"S3 upload failed: {e.response['Error']['Message']}"
            print(f"❌ {error_message}")
            raise Exception(error_message)
        except Exception as e:
            error_message = f"S3 upload failed: {str(e)}"
            print(f"❌ {error_message}")
            raise Exception(error_message)

    async def upload_buffer_to_s3(self, buffer: bytes, file_name: str) -> str:
        """
        Upload a video buffer to S3

        Args:
            buffer: Video file content as bytes
            file_name: Filename for S3

        Returns:
            Public URL of the uploaded video

        Raises:
            Exception: If upload fails
        """
        try:
            s3_key = f"videos/{file_name}"

            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=buffer,
                ContentType='video/mp4',
                ContentDisposition='inline'
            )

            # Generate public URL
            public_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"

            print(f"✅ Video buffer uploaded to S3: {public_url}")
            return public_url

        except ClientError as e:
            error_message = f"S3 buffer upload failed: {e.response['Error']['Message']}"
            print(f"❌ {error_message}")
            raise Exception(error_message)
        except Exception as e:
            error_message = f"S3 buffer upload failed: {str(e)}"
            print(f"❌ {error_message}")
            raise Exception(error_message)


# Singleton instance
s3_uploader = S3Uploader()
