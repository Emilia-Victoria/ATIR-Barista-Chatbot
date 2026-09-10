import pygame
import sys
import time
import random
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

# Animation state enumeration
class RobotState(Enum):
    IDLE = "idle"
    SPEAKING = "speaking"
    BLINKING = "blinking"
    LISTENING = "listening"

@dataclass
class AnimationSequence:
    name: str
    sprite_indices: List[int]
    frame_duration: float  # seconds per frame
    loop: bool = True
    priority: int = 1  # Higher = more important

class RobotFace:
    def __init__(self, sprites_folder="sprites", sprite_count=60):
        """Initialize the robot face system"""
        pygame.init()
        
        # Display setup
        self.screen_width = 1400
        self.screen_height = 1000
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Robot Face")
        
        # Face scaling and brightness
        self.face_scale = 2.5
        self.brightness_boost = 4.2
        self.glow_intensity = 0.9
        
        # Sprite loading setup
        self.sprites_folder = sprites_folder
        self.sprite_count = sprite_count
        self.eye_sprites = []
        self.mouth_sprites = []
        self.load_split_sprites()
        
        # Animation system
        self.current_state = RobotState.IDLE
        self.current_animation = None
        self.animation_frame = 0
        self.last_frame_time = time.time()
        self.animations = self.setup_animations()
        
        # Animation stuck detection
        self.animation_start_time = time.time()
        self.max_animation_duration = 20.0  # Force reset after 10 seconds
        
        # Eye tracking system
        self.eye_x = 0  # Offset from center (-1.0 to 1.0, left to right)
        self.eye_y = 0  # Offset from center (-1.0 to 1.0, up to down)
        self.eye_movement_range = 0.12  # Maximum movement as fraction of screen size (reduced from 0.3)
        
        # Timing and behavior
        self.blink_timer = time.time()
        self.next_blink = random.uniform(3.0, 6.0)
        self.blink_count = 0
        
        # Clock for smooth animation
        self.clock = pygame.time.Clock()
        self._needs_redraw = True
        
    def load_split_sprites(self):
        """Load all split sprites (eyes and mouth) from separate folders"""
        import os
        from pathlib import Path
        
        eyes_folder = f"{self.sprites_folder}_eyes"
        mouth_folder = f"{self.sprites_folder}_mouth"
        
        print(f"Loading split sprites from {eyes_folder}/ and {mouth_folder}/ folders...")
        
        for i in range(1, self.sprite_count + 1):
            eye_filename = f"{eyes_folder}/sprite_eyes_{i:02d}.png"
            mouth_filename = f"{mouth_folder}/sprite_mouth_{i:02d}.png"
            
            try:
                # Load eye sprite
                if os.path.exists(eye_filename):
                    original_eye = pygame.image.load(eye_filename).convert_alpha()
                    enhanced_eye = self.enhance_sprite(original_eye)
                    self.eye_sprites.append(enhanced_eye)
                else:
                    print(f"  Missing: {eye_filename}")
                    sys.exit()
                
                # Load mouth sprite
                if os.path.exists(mouth_filename):
                    original_mouth = pygame.image.load(mouth_filename).convert_alpha()
                    enhanced_mouth = self.enhance_sprite(original_mouth)
                    self.mouth_sprites.append(enhanced_mouth)
                else:
                    print(f"  Missing: {mouth_filename}")
                    sys.exit()
                        
            except pygame.error as e:
                print(f"  Error loading sprite {i:02d}: {e}")
                sys.exit()
        print(f"Loaded sprites")
        
    def enhance_sprite(self, sprite):
        """Enhance sprite brightness and scale it up"""
        original_size = sprite.get_size()
        new_size = (int(original_size[0] * self.face_scale), int(original_size[1] * self.face_scale))
        scaled_sprite = pygame.transform.scale(sprite, new_size)
        enhanced_sprite = scaled_sprite.copy()
        
        try:
            import numpy as np
            pixels = pygame.surfarray.array3d(enhanced_sprite)
            pixels = pixels.astype(float)
            pixels *= self.brightness_boost
            
            # Enhance neon colors
            pixels[:, :, 0] = np.minimum(pixels[:, :, 0] * 1.1, 255)  # Red
            pixels[:, :, 1] = np.minimum(pixels[:, :, 1] * 1.3, 255)  # Green  
            pixels[:, :, 2] = np.minimum(pixels[:, :, 2] * 1.2, 255)  # Blue
            
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)
            pygame.surfarray.blit_array(enhanced_sprite, pixels)
            
        except Exception as e:
            print(f"Warning: Could not enhance sprite brightness: {e}")
        
        return enhanced_sprite
    
    def setup_animations(self) -> Dict[str, AnimationSequence]:
        """Define simple animations"""
        # Sprites 1-4: Blink animation (0-3 in zero-indexed)
        # Sprites 5-48: Talking animations (4-47 in zero-indexed)
        # Sprites 49-60: Listening animation (48-59 in zero-indexed form)
        
        # Create smooth sequential talking animation
        talking_sprites = list(range(4, 32))  # Sequential: 5, 6, 7, 8... 32
        
        animations = {
            # Idle - static frame that doesn't advance
            "idle": AnimationSequence("idle", [1], float('inf'), False, 1),
            
            # Single blink
            "blink": AnimationSequence("blink", [4, 1, 2, 3, 2, 1, 4], 0.08, False, 10),
            
            # Double blink  
            "double_blink": AnimationSequence("double_blink", [4, 1, 2, 1, 4, 4, 1, 2, 1, 4], 0.08, False, 12),
            
            # Speaking - smooth sequential animation, slower speed
            "speaking": AnimationSequence("speaking", talking_sprites, 0.1, True, 5),

            # Speaking - smooth sequential animation, slower speed
            "listening": AnimationSequence("listening", [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 59], 0.2, True, 5),
        }
        
        return animations
    
    def start_animation(self, animation_name: str, force: bool = False):
        """Start a new animation sequence"""
        if animation_name not in self.animations:
            print(f"Animation '{animation_name}' not found")
            return
            
        new_anim = self.animations[animation_name]
        
        # Priority check only applies to interrupting ONGOING animations
        # Never block transitions when current animation is finished
        if (self.current_animation and not force and
            self.current_animation.priority >= new_anim.priority):
            return
        
        self.current_animation = new_anim
        self.animation_frame = 0
        self.last_frame_time = time.time()
        self.animation_start_time = time.time()  # Track when animation started
        self._needs_redraw = True
    
    def update_animation(self):
        """Update current animation frame"""
        if not self.current_animation:
            self.start_animation("idle")
            return
            
        current_time = time.time()
        
        
        # Skip frame updates for idle animation (infinite duration)
        if self.current_animation.name == "idle":
            return
            
        # Update frame timing for all non-idle animations
        if current_time - self.last_frame_time >= self.current_animation.frame_duration:
            self.animation_frame += 1
            self.last_frame_time = current_time
            self._needs_redraw = True
            
            if self.animation_frame >= len(self.current_animation.sprite_indices):
                if self.current_animation.loop:
                    # For looping animations, check if we should transition to idle
                    if (self.current_state == RobotState.IDLE and 
                        self.current_animation.name == "speaking") or (self.current_state == RobotState.IDLE and 
                        self.current_animation.name == "listening"):
                        # We were speaking but state changed to idle, transition
                        self.start_animation("idle")
                    else:
                        # Continue looping
                        self.animation_frame = 0
                else:
                    # Non-looping animation finished
                    self.animation_finished()
    
    def animation_finished(self):
        """Handle when a non-looping animation finishes"""  
        # Clear current animation first to prevent priority blocking
        self.current_animation = None
        
        # Then transition based on current state
        if self.current_state == RobotState.IDLE:            
            self.start_animation("idle")
        elif self.current_state == RobotState.SPEAKING:
            self.start_animation("speaking")
        elif self.current_state == RobotState.LISTENING:
            self.start_animation("listening")
        else:
            # Fallback to idle
            self.start_animation("idle")
    
    def get_current_sprite_index(self) -> int:
        """Get the current sprite index to display"""
        if not self.current_animation:
            return 4  # Default to first talking sprite
            
        frame_idx = min(self.animation_frame, len(self.current_animation.sprite_indices) - 1)
        return self.current_animation.sprite_indices[frame_idx]
    
    def set_eye_position(self, x: float = 0.0, y: float = 0.0):
        """Set target eye position. x,y range from -1.0 to 1.0 (left/right, up/down)"""
        eye_x = max(-1.0, min(1.0, x))
        eye_y = max(-1.0, min(1.0, y))
        if eye_x != self.eye_x or eye_y != self.eye_y:
            self.eye_x = eye_x
            self.eye_y = eye_y
            self._needs_redraw = True
    
    def get_eye_offset_pixels(self):
        """Convert eye position to pixel offsets"""
        offset_x = int(self.eye_x * self.screen_width * self.eye_movement_range)
        offset_y = int(self.eye_y * self.screen_height * self.eye_movement_range)
        return offset_x, offset_y
    
    def update_behavior(self):
        """Update automatic blinking"""
        current_time = time.time()
        
        time_ready = current_time - self.blink_timer >= self.next_blink
        state_ready = self.current_state == RobotState.IDLE
        anim_ready = self.current_animation and self.current_animation.name == "idle"
        
        # Handle automatic blinking (only when idle and not already blinking)
        if time_ready and state_ready and anim_ready:
            
            # 1 in 10 chance for double blink
            should_double_blink = (self.blink_count > 0 and random.random() < 0.1)
            
            if should_double_blink:
                self.start_animation("double_blink", force=True)
                self.blink_count = 0
            else:
                self.start_animation("blink", force=True)
                self.blink_count += 1
            
            # Reset timer only after actually starting the animation
            self.blink_timer = current_time
            self.next_blink = random.uniform(3.0, 8.0)
    
    def set_state(self, new_state: RobotState):
        """Change robot's state"""
        if new_state == self.current_state:
            return
            
        self.current_state = new_state
        
        # Handle state transitions
        if new_state == RobotState.IDLE:
            print("setting idle state")
            # Interrupt any current animation and go to idle
            # Unless it's a high-priority blink animation
            if (self.current_animation and 
                "blink" in self.current_animation.name and 
                not self.current_animation.loop):
                # Let blink finish, it will transition to idle automatically
                pass
            else:
                self.start_animation("idle", force=True)
                
        elif new_state == RobotState.SPEAKING:
            # Start speaking animation immediately
            # Blinks will be interrupted
            self.start_animation("speaking", force=True)
            print("started speaking ani")
        elif new_state == RobotState.LISTENING:
            # Start listening animation immediately
            # Blinks will be interrupted
            self.start_animation("listening", force=False)
            print("started listening ani")
    
    def draw(self):
        """Draw the current robot face with independent eye movement"""
        self.screen.fill((0, 0, 0))
        
        sprite_idx = self.get_current_sprite_index()
        if sprite_idx < len(self.eye_sprites) and sprite_idx < len(self.mouth_sprites):
            current_eye_sprite = self.eye_sprites[sprite_idx]
            current_mouth_sprite = self.mouth_sprites[sprite_idx]
            
            # Calculate base position (centered)
            base_x = self.screen_width // 2
            base_y = self.screen_height // 2
            
            # Get eye movement offset
            eye_offset_x, eye_offset_y = self.get_eye_offset_pixels()
            
            # Position mouth sprite (always centered)
            mouth_rect = current_mouth_sprite.get_rect()
            mouth_rect.centerx = base_x
            # Position mouth below where eyes would be
            mouth_rect.centery = base_y + (current_eye_sprite.get_height() // 2)
            
            # Position eye sprite (can move)
            eye_rect = current_eye_sprite.get_rect()
            eye_rect.centerx = base_x + eye_offset_x
            eye_rect.centery = base_y - (current_mouth_sprite.get_height() // 2) + eye_offset_y
            
            # Add glow effect
            if self.glow_intensity > 0:
                glow_offsets = [(-3, -3), (-2, -2), (-1, -1), (0, -2), (2, -2), (3, -3),
                               (-2, 0), (2, 0), (-3, 3), (-2, 2), (-1, 1), (0, 2), (2, 2), (3, 3)]
                
                # Glow for mouth
                mouth_alpha = current_mouth_sprite.get_alpha()
                current_mouth_sprite.set_alpha(int(255 * self.glow_intensity))
                for dx, dy in glow_offsets:
                    glow_rect = mouth_rect.copy()
                    glow_rect.x += dx
                    glow_rect.y += dy
                    self.screen.blit(current_mouth_sprite, glow_rect)
                current_mouth_sprite.set_alpha(mouth_alpha)
                
                # Glow for eyes
                eye_alpha = current_eye_sprite.get_alpha()
                current_eye_sprite.set_alpha(int(255 * self.glow_intensity))
                for dx, dy in glow_offsets:
                    glow_rect = eye_rect.copy()
                    glow_rect.x += dx
                    glow_rect.y += dy
                    self.screen.blit(current_eye_sprite, glow_rect)
                current_eye_sprite.set_alpha(eye_alpha)
            
            # Draw mouth first, then eyes on top
            self.screen.blit(current_mouth_sprite, mouth_rect)
            self.screen.blit(current_eye_sprite, eye_rect)
        
        pygame.display.flip()
        self._needs_redraw = False
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
    
    def loop(self):    
        self.update_behavior()
        self.update_animation()
        if self._needs_redraw:
            self.draw()
        self.clock.tick(30)

    def shutdown(self):
        """Release pygame resources."""
        pygame.quit()

    
    def run(self):
        """Main game loop - standalone mode for testing"""
        try:
            while self.handle_events():
                self.loop()
        finally:
            self.shutdown()
        

def main():
    """Main entry point for standalone mode"""
    print("Initializing Robot Face with Sprites...")
    
    robot = RobotFace()  
    robot.run()
if __name__ == "__main__":
    main()
