import os
import math
import random
import subprocess
import numpy as np
from PIL import Image, ImageFilter, ImageSequence
import imageio
from functools import partial

from typing import Optional

def ease_out_cubic(x):
    return 1 - (1 - x) ** 3
    
class Animations:
    def __init__(
        self,
        screen_width=1080,
        screen_height=1920,
        illustration_height_percent=0.45,
        padding_percent=0,
        duration=0.3,
        output_path="./"
    ):
        """
        :param output_path: directory where .webm files will be saved
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.top_area_height = int(screen_height * illustration_height_percent)
        self.padding_top = int(screen_height * padding_percent)
        self.default_duration = duration
        self.current_duration = duration
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.frame_generator = None
        self.last_effect = None
        self.sfx = None

    def load_image(self, image_path):
        self.image_path = image_path
        self.orig_img = Image.open(image_path).convert("RGBA")
        max_w, max_h = self.screen_width, self.top_area_height - self.padding_top
        iw, ih = self.orig_img.size
        scale = min(max_w / iw, max_h / ih, 1)
        self.target_width = int(iw * scale)
        self.target_height = int(ih * scale)

    def _prepare(self, image_path, duration, sfx):
        self.load_image(image_path)
        # clamp duration to avoid extremely long or unwanted lengths
        self.current_duration = duration if duration is not None else self.default_duration
        self.last_effect = None
        self.sfx = sfx

    # -- animation methods --
    def bubble_pop(self, image_path, duration=None, sfx=None):
        self._prepare(image_path, duration, sfx)
        self.last_effect = "bubble_pop"
        def frame(t):
            p = min(max(t / self.current_duration, 0), 1)
            w = max(1, int(self.target_width * p))
            h = max(1, int(self.target_height * p))
            img = self.orig_img.resize((w, h), Image.LANCZOS)
            bg = Image.new("RGBA", (self.screen_width, self.screen_height), (0,0,0,0))
            x = (self.screen_width - w)//2
            y = self.padding_top + (self.top_area_height - self.padding_top - h)//2
            bg.paste(img, (x,y), img)
            return np.array(bg)
        self.frame_generator = frame

    def slide_in(self, image_path, direction="left", duration=None, sfx=None):
        self._prepare(image_path, duration, sfx)
        self.last_effect = "slide_in"
        def frame(t):
            linear = min(max(t / self.current_duration, 0), 1)
            p = ease_out_cubic(linear)
            img = self.orig_img.resize((self.target_width, self.target_height), Image.LANCZOS)
            bg = Image.new("RGBA", (self.screen_width, self.screen_height), (0,0,0,0))
            y = self.padding_top + (self.top_area_height - self.padding_top - self.target_height)//2
            end_x = (self.screen_width - self.target_width)/2
            start_x = -self.target_width if direction == "left" else self.screen_width
            x = int(round(start_x + p * (end_x - start_x)))
            bg.paste(img, (x,y), img)
            return np.array(bg)
        self.frame_generator = frame

    def slide_in_overshoot(self, image_path, direction="left", duration=None, overshoot=100, overshoot_ratio=0.7, sfx=None):
        """
        Slides in past center by 'overshoot' pixels, then returns to center.
        :param overshoot: extra pixels past center before spring-back
        :param overshoot_ratio: portion of duration to reach overshoot
        """
        self._prepare(image_path, duration, sfx)
        def frame(t):
            p = min(max(t / self.current_duration, 0), 1)
            img = self.orig_img.resize((self.target_width, self.target_height), Image.LANCZOS)
            bg = Image.new("RGBA", (self.screen_width, self.screen_height), (0,0,0,0))
            y = self.padding_top + (self.top_area_height - self.padding_top - self.target_height)//2
            start_x = -self.target_width if direction == "left" else self.screen_width
            center_x = (self.screen_width - self.target_width)//2
            # overshoot position
            if direction == "left":
                over_x = center_x + overshoot
            else:
                over_x = center_x - overshoot
            if p < overshoot_ratio:
                # move from start to overshoot
                x = start_x + (p/overshoot_ratio) * (over_x - start_x)
            else:
                # spring back from overshoot to center
                q = (p - overshoot_ratio) / (1 - overshoot_ratio)
                x = over_x + q * (center_x - over_x)
            bg.paste(img, (int(x), y), img)
            return np.array(bg)
        self.frame_generator = frame

    def fade_in(self, image_path, duration=None, sfx=None):
        self._prepare(image_path, duration, sfx)
        def frame(t):
            p = min(max(t / self.current_duration, 0), 1)
            e = 1 - (1-p)**3
            img = self.orig_img.resize((self.target_width, self.target_height), Image.LANCZOS)
            alpha = img.split()[3].point(lambda p: p*e)
            img.putalpha(alpha)
            bg = Image.new("RGBA", (self.screen_width, self.screen_height), (0,0,0,0))
            x = (self.screen_width - self.target_width)//2
            y = self.padding_top + (self.top_area_height - self.padding_top - self.target_height)//2
            bg.paste(img, (x,y), img)
            return np.array(bg)
        self.frame_generator = frame

    def blur_in(self, image_path, duration=None, max_blur_radius=15, sfx=None):
        self._prepare(image_path, duration, sfx)
        def frame(t):
            p = min(max(t / self.current_duration, 0), 1)
            e = 1 - (1-p)**3
            r = max_blur_radius * (1-e)
            img = self.orig_img.resize((self.target_width, self.target_height), Image.LANCZOS)
            img = img.filter(ImageFilter.GaussianBlur(radius=r))
            bg = Image.new("RGBA", (self.screen_width, self.screen_height), (0,0,0,0))
            x = (self.screen_width - self.target_width)//2
            y = self.padding_top + (self.top_area_height - self.padding_top - self.target_height)//2
            bg.paste(img, (x,y), img)
            return np.array(bg)
        self.frame_generator = frame

    def drop(self, image_path, duration=None, sfx=None):
        self._prepare(image_path, duration, sfx)
        def frame(t):
            p = min(max(t / self.current_duration, 0), 1)
            e = 1 - (1-p)**3
            img = self.orig_img.resize((self.target_width, self.target_height), Image.LANCZOS)
            start_y = -self.target_height
            end_y = self.padding_top + (self.top_area_height - self.padding_top - self.target_height)//2
            y = int(round(start_y + e*(end_y - start_y)))
            x = (self.screen_width - self.target_width)//2
            bg = Image.new("RGBA", (self.screen_width, self.screen_height), (0,0,0,0))
            bg.paste(img, (x,y), img)
            return np.array(bg)
        self.frame_generator = frame

    def shake(self, image_path, intensity=15, frequency=25, duration=None, sfx=None):
        self._prepare(image_path, duration, sfx)
        def frame(t):
            p = min(max(t/self.current_duration,0),1)
            center_x = (self.screen_width - self.target_width)//2
            y = self.padding_top + (self.top_area_height - self.padding_top - self.target_height)//2
            offset = int(intensity*math.sin(2*math.pi*frequency*t) + random.uniform(-intensity/3, intensity/3))
            img = self.orig_img.resize((self.target_width, self.target_height), Image.LANCZOS)
            bg = Image.new("RGBA", (self.screen_width, self.screen_height), (0,0,0,0))
            bg.paste(img, (center_x+offset, y), img)
            return np.array(bg)
        self.frame_generator = frame

    def bounce(self, image_path, bounce_height=150, duration=None, sfx=None):
        self._prepare(image_path, duration, sfx)
        def frame(t):
            p = min(max(t/self.current_duration,0),1)
            bounce = 1 - (math.cos(4.5*math.pi*p) * math.exp(-6*p))
            img = self.orig_img.resize((self.target_width, self.target_height), Image.LANCZOS)
            x = (self.screen_width - self.target_width)//2
            target_y = self.padding_top + (self.top_area_height - self.padding_top - self.target_height)//2
            y = int(target_y - bounce_height*(1 - bounce))
            bg = Image.new("RGBA", (self.screen_width, self.screen_height), (0,0,0,0))
            bg.paste(img, (x,y), img)
            return np.array(bg)
        self.frame_generator = frame

    def blur_in_shake(self, image_path, max_blur_radius=15, intensity=15, frequency=25, duration=None, sfx=None):
        self._prepare(image_path, duration, sfx)
        def frame(t):
            p = min(max(t/self.current_duration,0),1)
            e = 1 - (1-p)**3
            r = max_blur_radius*(1-e)
            offset = int(intensity*math.sin(2*math.pi*frequency*t) + random.uniform(-intensity/3, intensity/3))
            img = self.orig_img.resize((self.target_width, self.target_height), Image.LANCZOS)
            img = img.filter(ImageFilter.GaussianBlur(radius=r))
            x = (self.screen_width - self.target_width)//2 + offset
            y = self.padding_top + (self.top_area_height - self.padding_top - self.target_height)//2
            bg = Image.new("RGBA", (self.screen_width, self.screen_height), (0,0,0,0))
            bg.paste(img, (x,y), img)
            return np.array(bg)
        self.frame_generator = frame

    def rotate_3d_page_flip(self, image_path, duration=None, sfx=None):
        self._prepare(image_path, duration, sfx)
        base = self.orig_img.resize((self.target_width, self.target_height), Image.LANCZOS)
        w, h = self.target_width, self.target_height
        def frame(t):
            def ease(t): return t*t*(3-2*t)
            p = ease(min(max(t/self.current_duration,0),1))
            ang = p * math.pi
            scale = abs(math.cos(ang))
            img = base if ang<=math.pi else base.transpose(Image.FLIP_LEFT_RIGHT)
            img = img.resize((max(1,int(w*scale)),h), Image.LANCZOS)
            x = (self.screen_width - max(1,int(w*scale)))//2
            y = self.padding_top + (self.top_area_height - self.padding_top - h)//2
            bg = Image.new("RGBA", (self.screen_width, self.screen_height), (0,0,0,0))
            bg.paste(img, (x,y), img)
            return np.array(bg)
        self.frame_generator = frame

    def bounce_pop_animation(self, image_path, duration=None, sfx=None):
        self._prepare(image_path, duration, sfx)
        base = self.orig_img.resize((self.target_width, self.target_height), Image.LANCZOS)
        w, h = self.target_width, self.target_height
        def osc(t): return 1 + (math.exp(-5*t) * math.cos(8*math.pi*t)) * 0.2
        def frame(t):
            p = min(max(t/self.current_duration,0),1)
            scale = osc(p)
            cw = max(1,int(w*scale)); ch = max(1,int(h*scale))
            img = base.resize((cw,ch), Image.LANCZOS)
            x = (self.screen_width-cw)//2
            y = self.padding_top + (self.top_area_height - self.padding_top - ch)//2
            bg = Image.new("RGBA", (self.screen_width, self.screen_height), (0,0,0,0))
            bg.paste(img, (x,y), img)
            return np.array(bg)
        self.frame_generator = frame

    def bubble_bounce_pop(self, image_path, duration=None, sfx=None):
        self._prepare(image_path, duration, sfx)
        def bounce_ease(t):
            n1,d1=7.5625,2.75
            if t<1/d1: return n1*t*t
            elif t<2/d1: t-=1.5/d1; return n1*t*t+0.75
            elif t<2.5/d1: t-=2.25/d1; return n1*t*t+0.9375
            else: t-=2.625/d1; return n1*t*t+0.984375
        def frame(t):
            p = min(max(t/self.current_duration,0),1)
            bv = bounce_ease(p)
            scale = bv/1.05*1.2
            if p>0.95:
                ep=(p-0.95)/0.05
                scale=scale*(1-ep)+1*ep
            w,h=self.target_width,self.target_height
            cw=max(1,int(w*scale)); ch=max(1,int(h*scale))
            img = self.orig_img.resize((cw,ch), Image.LANCZOS)
            x = (self.screen_width-cw)//2
            y = self.padding_top + (self.top_area_height - self.padding_top - ch)//2
            bg = Image.new("RGBA", (self.screen_width, self.screen_height), (0,0,0,0))
            bg.paste(img, (x,y), img)
            return np.array(bg)
        self.frame_generator = frame

    def transparent_in(self, image_path, duration=None, sfx=None):
        self.load_image(image_path)
        self.current_duration = duration or self.default_duration
        self.last_effect = "transparent_in"
        self.sfx = sfx

        def frame(t):
            t = min(t, self.current_duration)
            p = min(max(t / self.current_duration, 0), 1)
            alpha = int(p * 255)
            faded_img = self.orig_img.copy()
            faded_img.putalpha(alpha)
            bg = Image.new("RGBA", (self.screen_width, self.screen_height), (0, 0, 0, 0))
            x = (self.screen_width - self.target_width) // 2
            y = self.padding_top + (self.top_area_height - self.padding_top - self.target_height) // 2
            faded_img = faded_img.resize((self.target_width, self.target_height), Image.LANCZOS)
            bg.paste(faded_img, (x, y), faded_img)
            return np.array(bg)

        self.frame_generator = frame

    def place_in_y(self, image_path, duration=None, sfx=None):
        self.load_image(image_path)
        self.current_duration = duration or self.default_duration
        self.last_effect = "place_in_y"
        self.sfx = sfx

        # Starting scale larger than normal to simulate "closer to viewer"
        start_scale = 1.5  # 150% size at start, can tweak this
        end_scale = 1.0    # normal size at end

        # Starting position outside screen (e.g., above or in front, here top)
        # We'll move from above screen down to center.
        start_y = -int(self.target_height * start_scale)  # start fully above visible area
        end_y = self.padding_top + (self.top_area_height - self.padding_top - self.target_height) // 2

        def frame(t):
            t = min(t, self.current_duration)
            p = min(max(t / self.current_duration, 0), 1)  # progress 0->1

            # Interpolate scale and alpha
            scale = start_scale + (end_scale - start_scale) * p
            alpha = int(255 * p)

            # Calculate image size and position
            w = int(self.target_width * scale)
            h = int(self.target_height * scale)

            # Linear interpolate y position from start_y to end_y
            y = int(start_y + (end_y - start_y) * p)

            # Center horizontally
            x = (self.screen_width - w) // 2

            # Resize and set alpha
            img = self.orig_img.copy().resize((w, h), Image.LANCZOS)
            img.putalpha(alpha)

            # Create transparent background and paste image on it
            bg = Image.new("RGBA", (self.screen_width, self.screen_height), (0, 0, 0, 0))
            bg.paste(img, (x, y), img)

            return np.array(bg)

        self.frame_generator = frame

    def place_in_z(self, image_path, duration=None, sfx=None):
        self.load_image(image_path)
        self.current_duration = duration or self.default_duration
        self.last_effect = "place_in_z"
        self.sfx = sfx

        start_scale = 1.5
        end_scale = 1.0

        def frame(t):
            t = min(t, self.current_duration)
            linear_p = min(max(t / self.current_duration, 0), 1)
            p = ease_out_cubic(linear_p)  # easing progress

            scale = start_scale + (end_scale - start_scale) * p
            alpha = int(255 * p)

            w = int(self.target_width * scale)
            h = int(self.target_height * scale)

            x = (self.screen_width - w) // 2
            y = self.padding_top + (self.top_area_height - self.padding_top - h) // 2

            img = self.orig_img.copy().resize((w, h), Image.LANCZOS)
            img.putalpha(alpha)

            bg = Image.new("RGBA", (self.screen_width, self.screen_height), (0, 0, 0, 0))
            bg.paste(img, (x, y), img)

            return np.array(bg)

        self.frame_generator = frame


    # def render_gif(self, filename, fps=60):
    #     if not self.frame_generator:
    #         raise RuntimeError("No animation selected. Call an effect method first.")

    #     temp_gif = os.path.join(self.output_path, "temp.gif")
    #     frames = int(self.current_duration * fps)

    #     with imageio.get_writer(temp_gif, mode='I', duration=1 / fps, loop=0, disposal=2) as writer:
    #         for i in range(frames):
    #             t = (i + 0.5) / fps
    #             writer.append_data(self.frame_generator(t))

    #     output_webm = os.path.join(self.output_path, filename)
    #     cmd = [
    #         'ffmpeg', '-y',
    #         '-r', str(fps),
    #         '-i', temp_gif
    #     ]

    #     has_audio = False
    #     if self.sfx and os.path.isfile(self.sfx):
    #         cmd += ['-i', self.sfx]
    #         has_audio = True

    #     cmd += [
    #         '-c:v', 'libvpx-vp9', '-auto-alt-ref', '0',
    #         '-pix_fmt', 'yuva420p',
    #         '-r', str(fps)
    #     ]

    #     if has_audio:
    #         # Apply 20% volume and 0.1s fade-out
    #         audio_filter = f"volume=0.4,afade=t=out:st={self.current_duration - 0.1:.2f}:d=0.2"
    #         cmd += ['-filter:a', audio_filter, '-c:a', 'libopus', '-shortest']

    #     cmd += [output_webm]

    #     try:
    #         subprocess.run(cmd, check=True)
    #     finally:
    #         os.remove(temp_gif)




    def render_mov(self, filename, fps=30):
        if not self.frame_generator:
            raise RuntimeError("No animation selected. Call an effect method first.")

        # 1) make a tmp folder for PNGs
        png_dir = os.path.join(self.output_path, "temp_pngs")
        os.makedirs(png_dir, exist_ok=True)

        # 2) render each frame as a RGBA PNG
        total_frames = int(self.current_duration * fps)
        for i in range(total_frames):
            t = (i + 0.5) / fps
            arr = self.frame_generator(t)
            img = Image.fromarray(arr, mode="RGBA")
            img.save(os.path.join(png_dir, f"frame_{i:05d}.png"))

        # 3) build ffmpeg cmd using the PNG codec (which supports alpha)
        output_mov = os.path.join(
            self.output_path,
            filename if filename.lower().endswith(".mov") else filename + ".mov"
        )
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),          # input framerate
            '-i', os.path.join(png_dir, 'frame_%05d.png'),
        ]

        if self.sfx and os.path.isfile(self.sfx):
            cmd += ['-i', self.sfx]

        cmd += [
            '-c:v', 'prores_ks',
            '-profile:v', '4',
            '-pix_fmt', 'yuva444p10le',
            '-r', str(fps),                  # output framerate 60fps
            '-fps_mode', 'cfr',             # force constant framerate
        ]

        if self.sfx and os.path.isfile(self.sfx):
            cmd += [
                '-c:a', 'aac',
                '-b:a', '96k',
                '-filter:a', f"volume=0.9,afade=t=out:st={self.current_duration - 0.1:.2f}:d=0.2",
                '-shortest'
            ]

        cmd.append(output_mov)

        # 4) run & clean up
        try:
            subprocess.run(cmd, check=True)
        finally:
            # remove PNGs
            for f in os.listdir(png_dir):
                os.remove(os.path.join(png_dir, f))
            os.rmdir(png_dir)




# --- Function to process a directory of images with random animations ---
def process_images_with_random_fx(
        images_dir: str,
        sfx_dir: str,
        output_dir: str = 'webms_test',
        extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    ):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(extensions)]
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        entry = random.choice(animations_simple_fx)
        base_sfx = entry['sfx']
        sfx_path = None
        for ext in ('.wav', '.mp3'):
            candidate = os.path.join(sfx_dir, base_sfx + ext)
            if os.path.isfile(candidate):
                sfx_path = candidate
                break
        func = entry['function']
        duration = entry['duration']
        name, _ = os.path.splitext(img_file)
        output_name = f"{name}_{entry['animation']}.mov"
        func(img_path, duration=duration, sfx=sfx_path)
        anim.render_mov(output_name)
        print(f"Processed {img_file} -> {output_name} using {entry['animation']}")


if __name__=="__main__":
    # Instantiate Animations, directing output to 'webms_test'
    anim = Animations(output_path='./webms_test')

    # -- Mapping simple fx entries to their corresponding functions --
    animations_simple_fx = [
        {"animation": "bubble_pop",          "duration": 0.6, "sfx": "pop1",        "function": anim.bubble_pop},
        {"animation": "slide_in_left",       "duration": 0.6,  "sfx": "swish",       "function": partial(anim.slide_in, direction="left")},
        {"animation": "slide_in_right",      "duration": 0.6,  "sfx": "swish",       "function": partial(anim.slide_in, direction="right")},
        {"animation": "slide_in_overshoot_left",  "duration": 0.6, "sfx": "swoosh",     "function": partial(anim.slide_in_overshoot, direction="left")},
        {"animation": "slide_in_overshoot_right", "duration": 0.6, "sfx": "swoosh",     "function": partial(anim.slide_in_overshoot, direction="right")},
        {"animation": "fade_in",             "duration": 0.6, "sfx": "darkwoosh",   "function": anim.fade_in},
        {"animation": "blur_in",             "duration": 0.6, "sfx": "shimmer",     "function": anim.blur_in},
        {"animation": "drop",                "duration": 0.6, "sfx": "thud",        "function": anim.drop},
        {"animation": "shake",               "duration": 0.6, "sfx": "chime",       "function": anim.shake},
        {"animation": "bounce",              "duration": 0.6, "sfx": "bpop2",       "function": anim.bounce},
        {"animation": "blur_in_shake",       "duration": 0.6, "sfx": "whooshnormal","function": anim.blur_in_shake},
        {"animation": "rotate_3d_page_flip", "duration": 0.6, "sfx": "flip",        "function": anim.rotate_3d_page_flip},
        {"animation": "bounce_pop_animation","duration": 0.6, "sfx": "pop1",        "function": anim.bounce_pop_animation},
        {"animation": "bubble_bounce_pop",   "duration": 0.6, "sfx": "bpop2",       "function": anim.bubble_bounce_pop},
        {"animation": "transparent_in",      "duration": 0.6, "sfx": "shimmer",     "function": anim.transparent_in},
        {"animation": "place_in_y",          "duration": 0.6, "sfx": "swoosh",     "function": anim.place_in_y},
        {"animation": "place_in_z",          "duration": 0.6, "sfx": "swoosh",     "function": anim.place_in_z},
    ]


    # Run processing
    process_images_with_random_fx(
        'test_imgs',
        'src/open_deep_research/RESOURCES/SFX'
    )