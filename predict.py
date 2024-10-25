import os
from cog import BasePredictor, Input, Path
import torch
from torchvision import transforms
import moviepy.editor as mp
from PIL import Image, ImageOps
import numpy as np
from transformers import AutoModelForImageSegmentation
from PIL import ImageFilter


torch.set_float32_matmul_precision("medium")

class Predictor(BasePredictor):
    def setup(self):
        self.birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        self.birefnet.to("cuda")
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Add background handling modes
        self.bg_modes = {
            "cover": self._bg_cover,      # Fill entire frame, crop excess
            "contain": self._bg_contain,  # Show entire bg, with letterbox/pillarbox
            "blur": self._bg_blur,        # Blur and stretch bg to fill
            "mirror": self._bg_mirror     # Mirror edges to fill gaps
        }

    def predict(
        self,
        input_video: Path = Input(description="Input video"),
        bg_type: str = Input(description="Background Type", choices=["Color", "Image", "Video"], default="Color"),
        bg_image: Path = Input(description="Background Image", default=None),
        bg_video: Path = Input(description="Background Video", default=None),
        bg_mode: str = Input(description="Background Mode", 
                           choices=["cover", "blur", "contain", "mirror"], 
                           default="cover"),
        color: str = Input(description="Background Color", default="#00FF00"),
        fps: int = Input(description="Output FPS", default=0),
        video_handling: str = Input(description="Video Handling", 
                                  choices=["loop", "slow_down", "freeze"], 
                                  default="loop"),
    ) -> Path:
        try:
            # Load the video using moviepy
            video = mp.VideoFileClip(str(input_video))

            # Load original fps if fps value is equal to 0
            if fps == 0:
                fps = video.fps

            # Extract audio from the video
            audio = video.audio

            # Extract frames at the specified FPS
            frames = video.iter_frames(fps=fps)

            # Get the original video dimensions
            original_width, original_height = video.w, video.h

            # Process each frame for background removal
            processed_frames = []

            if bg_type == "Video":
                background_video = mp.VideoFileClip(str(bg_video))
                background_video = self.bg_modes[bg_mode](
                    background_video, 
                    original_width, 
                    original_height,
                    None  # Removed blur_amount parameter
                )
                
                # Handle video duration
                if background_video.duration < video.duration:
                    if video_handling == "slow_down":
                        background_video = background_video.fx(mp.vfx.speedx, 
                            factor=video.duration / background_video.duration)
                    elif video_handling == "loop":
                        background_video = mp.concatenate_videoclips(
                            [background_video] * int(video.duration / background_video.duration + 1))
                    else:  # freeze
                        last_frame = background_video.frames[-1]
                        freeze_clip = mp.ImageClip(last_frame, 
                            duration=video.duration - background_video.duration)
                        background_video = mp.concatenate_videoclips([background_video, freeze_clip])
                
                background_frames = list(background_video.iter_frames(fps=fps))

            else:
                background_frames = None

            bg_frame_index = 0  # Initialize background frame index

            for frame in frames:
                pil_image = Image.fromarray(frame)
                if bg_type == "Color":
                    processed_image = self.process(pil_image, color, original_width, original_height)
                elif bg_type == "Image":
                    processed_image = self.process(pil_image, bg_image, original_width, original_height)
                elif bg_type == "Video":
                    background_frame = background_frames[bg_frame_index % len(background_frames)]
                    bg_frame_index += 1
                    background_image = Image.fromarray(background_frame)
                    processed_image = self.process(pil_image, background_image, original_width, original_height)
                else:
                    processed_image = pil_image  # Default to original image if no background is selected

                processed_frames.append(np.array(processed_image))

            # Create a new video from the processed frames
            processed_video = mp.ImageSequenceClip(processed_frames, fps=fps)

            # Add the original audio back to the processed video
            processed_video = processed_video.set_audio(audio)

            # Save the processed video to a temporary file
            output_path = "/tmp/output.mp4"
            processed_video.write_videofile(output_path, codec="libx264")

            return Path(output_path)

        except Exception as e:
            print(f"Error: {e}")
            return None

    def process(self, image, bg, target_width, target_height):
        # Calculate center padding for the input image
        image = ImageOps.contain(image, (target_width, target_height))
        # Convert to RGB to ensure consistency
        image = image.convert('RGB')
        
        # Create a background of target size
        padded_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        # Paste the resized image in the center
        paste_x = (target_width - image.size[0]) // 2
        paste_y = (target_height - image.size[1]) // 2
        padded_image.paste(image, (paste_x, paste_y))
        image = padded_image

        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        
        # Prediction
        with torch.no_grad():
            preds = self.birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)

        # Handle background based on type
        if isinstance(bg, str) and bg.startswith("#"):
            # For solid color, create RGB background
            background = Image.new("RGB", (target_width, target_height), 
                                 tuple(int(bg[i:i+2], 16) for i in (1, 3, 5)))
        else:
            # For image or video frame backgrounds
            if isinstance(bg, Image.Image):
                bg_image = bg.convert("RGB")
            else:
                bg_image = Image.open(bg).convert("RGB")
            
            # Calculate aspect ratios
            target_ratio = target_width / target_height
            bg_ratio = bg_image.width / bg_image.height

            if bg_ratio > target_ratio:
                # Background is wider, fit to height
                new_height = target_height
                new_width = int(new_height * bg_ratio)
            else:
                # Background is taller, fit to width
                new_width = target_width
                new_height = int(new_width / bg_ratio)

            # Resize background maintaining its aspect ratio
            bg_image = bg_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create a new image with target dimensions
            background = Image.new("RGB", (target_width, target_height), (0, 0, 0))
            
            # Calculate position to center the background
            x = (target_width - new_width) // 2
            y = (target_height - new_height) // 2
            
            # Paste the background image centered
            background.paste(bg_image, (x, y))

        # Composite the image onto the background using the mask
        result = Image.composite(image, background, mask)
        return result

    def _bg_cover(self, video, target_width, target_height, _):
        """Fill frame completely, cropping excess"""
        target_ratio = target_width / target_height
        bg_ratio = video.w / video.h
        
        if bg_ratio > target_ratio:
            new_height = target_height
            new_width = int(new_height * bg_ratio)
        else:
            new_width = target_width
            new_height = int(new_width / bg_ratio)
            
        video = video.resize(width=new_width, height=new_height)
        x_offset = (new_width - target_width) // 2
        y_offset = (new_height - target_height) // 2
        
        return video.crop(x1=x_offset, y1=y_offset, 
                         width=target_width, height=target_height)

    def _bg_contain(self, video, target_width, target_height, _):
        """Show entire background with letterbox/pillarbox"""
        target_ratio = target_width / target_height
        bg_ratio = video.w / video.h
        
        if bg_ratio > target_ratio:
            new_width = target_width
            new_height = int(target_width / bg_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * bg_ratio)
            
        video = video.resize(width=new_width, height=new_height)
        
        # Create black padding
        def add_padding(frame):
            padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            padded[y_offset:y_offset+new_height, 
                  x_offset:x_offset+new_width] = frame
            return padded
            
        return video.fl_image(add_padding)

    def _bg_blur(self, video, target_width, target_height, blur_amount):
        """Blur and stretch background to fill"""
        # First scale up to cover the frame
        video = self._bg_cover(video, target_width, target_height, None)
        
        # Create blurred edges
        def apply_blur(frame):
            # Convert to PIL for better blur quality
            pil_img = Image.fromarray(frame)
            blurred = pil_img.filter(ImageFilter.GaussianBlur(blur_amount))
            return np.array(blurred)
            
        return video.fl_image(apply_blur)

    def _bg_mirror(self, video, target_width, target_height, _):
        """Mirror edges to fill gaps"""
        # First contain the video
        video = self._bg_contain(video, target_width, target_height, None)
        
        def mirror_edges(frame):
            h, w = frame.shape[:2]
            if h < target_height:
                # Mirror top and bottom
                top_pad = (target_height - h) // 2
                bottom_pad = target_height - h - top_pad
                if top_pad > 0:
                    top = np.flip(frame[:top_pad], axis=0)
                    frame = np.vstack([top, frame])
                if bottom_pad > 0:
                    bottom = np.flip(frame[-bottom_pad:], axis=0)
                    frame = np.vstack([frame, bottom])
            
            if w < target_width:
                # Mirror left and right
                left_pad = (target_width - w) // 2
                right_pad = target_width - w - left_pad
                if left_pad > 0:
                    left = np.flip(frame[:, :left_pad], axis=1)
                    frame = np.hstack([left, frame])
                if right_pad > 0:
                    right = np.flip(frame[:, -right_pad:], axis=1)
                    frame = np.hstack([frame, right])
            
            return frame
            
        return video.fl_image(mirror_edges)
