# import os
# import json
# from pathlib import Path
# from typing import List, Dict, Tuple

# from moviepy import CompositeVideoClip, ImageClip, VideoFileClip
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont

# # Hardcoded configuration
# CAPTIONS_FONT_PATH = "Denk_One/DenkOne-Regular.ttf"
# CAPTIONS_TEXT_COLOR = (173, 216, 230, 255)  # Light blue color
# CAPTIONS_BORDER_COLOR = (0, 0, 0, 255)  # Black border


# class VideoCaptioner:
#     def __init__(self):
#         # Increased font size and thick border
#         self.fontsize = 120         # increased font size in pixels
#         self.padding_x = 20        # fixed horizontal padding in pixels
#         self.padding_y = 20        # fixed vertical padding in pixels
#         self.line_spacing = 50     # increased line spacing in pixels

#         self.caption_config = {
#             "font": CAPTIONS_FONT_PATH,
#             "text_color": CAPTIONS_TEXT_COLOR,
#             "border_color": CAPTIONS_BORDER_COLOR,
#             "border_px": 8  # thick border
#         }

#     def add_captions_to_video(self, video_path: str, subtitles: List[Dict], output_path: str) -> None:
#         video = VideoFileClip(video_path)
#         frame_width, frame_height = video.size
#         clips = [video]

#         for entry in subtitles:
#             caption_clip = self._create_caption_clip(entry, (frame_width, frame_height))
#             if caption_clip:
#                 clips.append(caption_clip)

#         final = CompositeVideoClip(clips).with_audio(video.audio)
#         final.write_videofile(output_path, fps=30, codec='libx264', audio_codec='aac')

#     def _create_caption_clip(self, subtitle_entry: Dict, framesize: Tuple[int, int]) -> ImageClip:
#         """Create a single caption clip from subtitle entry"""
#         frame_width, frame_height = framesize
#         fontsize = self.fontsize
        
#         pil_font = ImageFont.truetype(self.caption_config['font'], fontsize)
        
#         # Get the text from the subtitle entry
#         text = subtitle_entry['word']
#         start_time = subtitle_entry['start']
#         end_time = subtitle_entry['end']
        
#         # Clean the text
#         clean_text = text.strip()
#         if not clean_text:
#             return None
            
#         # Split text into words for line wrapping
#         words = clean_text.split()
#         if not words:
#             return None
        
#         # Calculate line wrapping
#         x_buffer = 0.1 * frame_width
#         max_line_width = frame_width - 2 * x_buffer
#         space_w = pil_font.getbbox(' ')[2] - pil_font.getbbox(' ')[0]
        
#         # Build lines with word wrapping
#         lines = []
#         current_line = []
#         current_width = 0
        
#         for word in words:
#             bbox = pil_font.getbbox(word)
#             word_width = bbox[2] - bbox[0]
            
#             # Check if word fits on current line
#             space_needed = space_w if current_line else 0
#             if current_width + space_needed + word_width <= max_line_width:
#                 current_line.append(word)
#                 current_width += space_needed + word_width
#             else:
#                 # Start new line
#                 if current_line:
#                     lines.append(' '.join(current_line))
#                 current_line = [word]
#                 current_width = word_width
        
#         # Add remaining words
#         if current_line:
#             lines.append(' '.join(current_line))
        
#         if not lines:
#             return None
            
#         # Create the caption image
#         return self._create_multiline_caption_image(
#             lines, pil_font, start_time, end_time, framesize
#         )

#     def _create_multiline_caption_image(self, lines: List[str], font: ImageFont.FreeTypeFont, 
#                                       start_time: float, end_time: float, 
#                                       framesize: Tuple[int, int]) -> ImageClip:
#         """Create a multi-line caption image with center bloom animation"""
#         frame_width, frame_height = framesize
        
#         # Calculate dimensions for all lines
#         line_heights = []
#         line_widths = []
#         max_width = 0
        
#         for line in lines:
#             bbox = font.getbbox(line)
#             width = bbox[2] - bbox[0]
#             height = bbox[3] - bbox[1]
#             line_widths.append(width)
#             line_heights.append(height)
#             max_width = max(max_width, width)
        
#         # Calculate total height
#         total_height = sum(line_heights) + (len(lines) - 1) * self.line_spacing
        
#         # Add padding for border
#         border_padding = self.caption_config.get('border_px', 8) * 2
#         img_width = int(max_width + 2 * border_padding)
#         img_height = int(total_height + 2 * border_padding)
        
#         # Ensure minimum dimensions
#         img_width = max(img_width, 10)
#         img_height = max(img_height, 10)
        
#         # Create image
#         img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
#         draw = ImageDraw.Draw(img)
        
#         # Draw each line
#         y_offset = border_padding
#         text_color = self.caption_config['text_color']
#         border_color = self.caption_config['border_color']
#         border_width = self.caption_config.get('border_px', 8)
        
#         for i, line in enumerate(lines):
#             # Center each line horizontally
#             line_width = line_widths[i]
#             x_offset = (img_width - line_width) // 2
            
#             # Get the bounding box for proper positioning
#             bbox = font.getbbox(line)
#             text_x = x_offset - bbox[0]
#             text_y = y_offset - bbox[1]
            
#             # Draw text with border
#             draw.text(
#                 (text_x, text_y),
#                 line,
#                 font=font,
#                 fill=text_color,
#                 stroke_width=border_width,
#                 stroke_fill=border_color
#             )
            
#             y_offset += line_heights[i] + self.line_spacing
        
#         # Create ImageClip with the complete styled text
#         clip = ImageClip(np.array(img))
#         clip = clip.with_start(start_time).with_duration(end_time - start_time)
        
#         # Position at bottom center of frame
#         y_position = int(frame_height * 0.55)
#         x_position = (frame_width - img_width) // 2
        
#         # Apply center bloom animation where EVERYTHING scales together
#         def center_bloom_transform(get_frame, t):
#             anim_duration = 0.3  # 0.3 seconds animation
#             if t < anim_duration:
#                 # Smooth easing function (ease-out cubic)
#                 progress = t / anim_duration
#                 # Cubic ease-out: 1 - (1 - t)^3
#                 eased_progress = 1 - pow(1 - progress, 3)
#                 # Scale from 0.1 to 1.0 with no overshoot
#                 scale = 0.1 + (eased_progress * 0.9)
#             else:
#                 scale = 1.0
            
#             # Get the original frame (complete caption with borders/effects)
#             frame = get_frame(t)
            
#             # If scale is 1.0, return original frame
#             if scale == 1.0:
#                 return frame
            
#             # Try using OpenCV first, fallback to PIL
#             try:
#                 import cv2
#                 # Get original dimensions
#                 height, width = frame.shape[:2]
                
#                 # Calculate new dimensions
#                 new_width = int(width * scale)
#                 new_height = int(height * scale)
                
#                 # Ensure minimum size
#                 new_width = max(new_width, 1)
#                 new_height = max(new_height, 1)
                
#                 # Resize the ENTIRE frame (text + borders + effects all together)
#                 scaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                
#                 # Create new frame with original dimensions, filled with transparent
#                 new_frame = np.zeros((height, width, 4), dtype=frame.dtype)
                
#                 # Calculate position to center the scaled frame
#                 x_offset = (width - new_width) // 2
#                 y_offset = (height - new_height) // 2
                
#                 # Place scaled frame in center
#                 new_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = scaled_frame
                
#                 return new_frame
                
#             except ImportError:
#                 # Fallback to PIL if OpenCV is not available
#                 from PIL import Image
#                 img = Image.fromarray(frame)
                
#                 # Calculate new dimensions
#                 new_width = int(img.width * scale)
#                 new_height = int(img.height * scale)
                
#                 # Ensure minimum size
#                 new_width = max(new_width, 1)
#                 new_height = max(new_height, 1)
                
#                 # Resize the ENTIRE image (text + borders + effects all together)
#                 scaled_img = img.resize((new_width, new_height), Image.LANCZOS)
                
#                 # Create new frame with original dimensions, centered
#                 new_frame = Image.new('RGBA', (img.width, img.height), (0, 0, 0, 0))
                
#                 # Calculate position to center the scaled image
#                 x_offset = (img.width - new_width) // 2
#                 y_offset = (img.height - new_height) // 2
                
#                 # Paste scaled image centered
#                 new_frame.paste(scaled_img, (x_offset, y_offset))
                
#                 return np.array(new_frame)
        
#         # Apply the transform to the complete caption (text + borders + effects)
#         clip = clip.transform(center_bloom_transform, keep_duration=True)
        
#         # Position the clip (MoviePy 2.0 syntax)
#         clip = clip.with_position((x_position, y_position))
        
#         return clip


import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

from moviepy import CompositeVideoClip, VideoFileClip
from moviepy.video.VideoClip import VideoClip
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from open_deep_research.configuration import Configuration
from langchain_core.runnables import RunnableConfig


def get_config(config: RunnableConfig) -> str:
    cfg = Configuration.from_runnable_config(config)
    return cfg

config_dict = {"configurable": {}}
configurable = get_config(RunnableConfig(config_dict))


# Hardcoded configuration
CAPTIONS_FONT_PATH = configurable.font_style_path
CAPTIONS_TEXT_COLOR = (153, 255, 204, 255) 
CAPTIONS_BORDER_COLOR = (0, 0, 0, 255)

class VideoCaptioner:
    def __init__(self):
        # Font and layout settings
        self.fontsize = 120          # Font size in pixels
        self.padding_x = 20          # Horizontal padding in pixels (per line)
        self.padding_y = 40          # Vertical padding in pixels (per line)
        self.line_spacing = 50       # Line spacing in pixels

        self.caption_config = {
            "font": CAPTIONS_FONT_PATH,
            "text_color": CAPTIONS_TEXT_COLOR,
            "border_color": CAPTIONS_BORDER_COLOR,
            "border_px": 10           # Border thickness
        }

    def add_captions_to_video(
        self,
        video_path: str,
        subtitles: List[Dict],
        output_path: str
    ) -> None:
        video = VideoFileClip(video_path)
        frame_width, frame_height = video.size
        clips: List[VideoClip] = [video]

        for entry in subtitles:
            caption_clip = self._create_caption_clip(entry, (frame_width, frame_height))
            if caption_clip:
                clips.append(caption_clip)

        final = CompositeVideoClip(clips).with_audio(video.audio)
        final.write_videofile(
            output_path,
            fps=30,
            codec='libx264',
            audio_codec='aac'
        )

    def _create_caption_clip(
        self,
        subtitle_entry: Dict,
        framesize: Tuple[int, int]
    ) -> VideoClip:
        frame_width, frame_height = framesize
        text = subtitle_entry.get('word', '').strip()
        start_time = subtitle_entry.get('start', 0)
        end_time = subtitle_entry.get('end', start_time)

        if not text or end_time <= start_time:
            return None

        # Prepare font
        pil_font = ImageFont.truetype(self.caption_config['font'], self.fontsize)

        # Word-wrap
        x_buffer = 0.1 * frame_width
        max_width = frame_width - 2 * x_buffer
        space_w = pil_font.getbbox(' ')[2] - pil_font.getbbox(' ')[0]

        words = text.split()
        lines, current_line, current_w = [], [], 0
        for word in words:
            w = pil_font.getbbox(word)[2] - pil_font.getbbox(word)[0]
            needed = space_w if current_line else 0
            if current_w + needed + w <= max_width:
                current_line.append(word)
                current_w += needed + w
            else:
                lines.append(' '.join(current_line))
                current_line, current_w = [word], w
        if current_line:
            lines.append(' '.join(current_line))

        if not lines:
            return None

        # Build caption image and frame function with scalable padding
        img_array, img_w, img_h = self._build_caption_image(lines, pil_font)

        def frame_fn(t: float) -> np.ndarray:
            # Ease-out cubic over 0.3s
            anim_dur = 0.3
            if t < anim_dur:
                p = t / anim_dur
                scale = 1 - (1 - p)**3
            else:
                scale = 1.0

            if scale <= 0:
                return np.zeros_like(img_array)
            if scale >= 1:
                return img_array

            # Scale using PIL
            img = Image.fromarray(img_array, 'RGBA')
            new_w = max(int(img_w * scale), 1)
            new_h = max(int(img_h * scale), 1)
            scaled = img.resize((new_w, new_h), Image.LANCZOS)
            frame = Image.new('RGBA', (img_w, img_h), (0,0,0,0))
            x_off = (img_w - new_w)//2
            y_off = (img_h - new_h)//2
            frame.paste(scaled, (x_off, y_off), scaled)
            return np.array(frame)

        duration = end_time - start_time
        # Instantiate VideoClip then set duration
        caption_clip = VideoClip(frame_fn)
        caption_clip = (
            caption_clip
            .with_duration(duration)
            .with_start(start_time)
            .with_position(('center', int(frame_height * 0.4)))
        )

        return caption_clip

    def _build_caption_image(
        self,
        lines: List[str],
        font: ImageFont.FreeTypeFont
    ) -> Tuple[np.ndarray, int, int]:
        # Measure lines
        widths, heights = [], []
        max_w = 0
        for line in lines:
            bbox = font.getbbox(line)
            w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
            widths.append(w); heights.append(h)
            max_w = max(max_w, w)

        total_h = sum(heights) + (len(lines)-1)*self.line_spacing
        border = self.caption_config['border_px']
        # Scalable padding: per-line vertical padding
        pad_x = self.padding_x * len(lines) + border
        pad_y = self.padding_y * len(lines) + border
        img_w = int(max_w + 2 * pad_x)
        img_h = int(total_h + 2 * pad_y)

        img = Image.new('RGBA', (img_w, img_h), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        y = pad_y

        for i, line in enumerate(lines):
            w, h = widths[i], heights[i]
            x = (img_w - w)//2
            draw.text(
                (x, y),
                line,
                font=font,
                fill=self.caption_config['text_color'],
                stroke_width=border,
                stroke_fill=self.caption_config['border_color']
            )
            y += h + self.line_spacing

        arr = np.array(img, dtype=np.uint8)
        return arr, img_w, img_h
