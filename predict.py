import os
from cog import BasePredictor, Input, Path
import torch
from torchvision import transforms
import moviepy.editor as mp
from PIL import Image
import numpy as np
from transformers import AutoModelForImageSegmentation


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

    def predict(
        self,
        input_video: Path = Input(description="Input video"),
        bg_type: str = Input(description="Background Type", choices=["Color", "Image", "Video"], default="Color"),
        bg_image: Path = Input(description="Background Image", default=None),
        bg_video: Path = Input(description="Background Video", default=None),
        color: str = Input(description="Background Color", default="#00FF00"),
        fps: int = Input(description="Output FPS (0 will inherit the original fps value)", default=0),
        video_handling: str = Input(description="Video Handling", choices=["slow_down", "loop"], default="slow_down"),
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

            # Process each frame for background removal
            processed_frames = []

            if bg_type == "Video":
                background_video = mp.VideoFileClip(str(bg_video))
                if background_video.duration < video.duration:
                    if video_handling == "slow_down":
                        background_video = background_video.fx(mp.vfx.speedx, factor=video.duration / background_video.duration)
                    else:  # video_handling == "loop"
                        background_video = mp.concatenate_videoclips([background_video] * int(video.duration / background_video.duration + 1))
                background_frames = list(background_video.iter_frames(fps=fps))  # Convert to list
            else:
                background_frames = None

            bg_frame_index = 0  # Initialize background frame index

            for frame in frames:
                pil_image = Image.fromarray(frame)
                if bg_type == "Color":
                    processed_image = self.process(pil_image, color)
                elif bg_type == "Image":
                    processed_image = self.process(pil_image, bg_image)
                elif bg_type == "Video":
                    background_frame = background_frames[bg_frame_index % len(background_frames)]
                    bg_frame_index += 1
                    background_image = Image.fromarray(background_frame)
                    processed_image = self.process(pil_image, background_image)
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

    def process(self, image, bg):
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)

        if isinstance(bg, str) and bg.startswith("#"):
            color_rgb = tuple(int(bg[i:i+2], 16) for i in (1, 3, 5))
            background = Image.new("RGBA", image_size, color_rgb + (255,))
        elif isinstance(bg, Image.Image):
            background = bg.convert("RGBA").resize(image_size)
        else:
            background = Image.open(bg).convert("RGBA").resize(image_size)

        # Composite the image onto the background using the mask
        image = Image.composite(image, background, mask)

        return image
