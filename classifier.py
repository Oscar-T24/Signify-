import pygame
import sys
import threading
import subprocess
from main import LoadCV, HandTrackingDynamic, analyze  # assuming these exist

# Global flag for video feed visibility
show_video = False
HEIGHT = 720
WIDTH = 1280


# ---------------- Scene Base & Manager ----------------

class SceneBase:
    """ Base class for all scenes in the game. """

    def __init__(self, scene_manager):
        self.scene_manager = scene_manager  # Reference to the scene manager

    def ProcessInput(self, events, pressed_keys):
        """ Process all input events """
        pass

    def Update(self):
        """ Update scene logic """
        pass

    def Render(self, screen):
        """ Render the scene to the screen """
        pass

    def SwitchToScene(self, next_scene):
        """ Switch to the next scene """
        self.scene_manager.SwitchToScene(next_scene)

    def Terminate(self):
        """ Terminate the current scene """
        self.SwitchToScene(None)


class SceneManager:
    """ Manages the scenes and transitions between them. """

    def __init__(self, initial_scene):
        self.current_scene = initial_scene

    def SwitchToScene(self, next_scene):
        """ Switch to a new scene. """
        self.current_scene = next_scene

    def Run(self, screen):
        """ Main loop for handling the current scene. """
        clock = pygame.time.Clock()
        while self.current_scene is not None:
            events = pygame.event.get()
            pressed_keys = pygame.key.get_pressed()

            for event in events:
                if event.type == pygame.QUIT:
                    self.current_scene.Terminate()

            self.current_scene.ProcessInput(events, pressed_keys)
            self.current_scene.Update()
            self.current_scene.Render(screen)

            pygame.display.flip()  # Update display
            clock.tick(30)  # Limit to 30 FPS


# ---------------- Output Frame ----------------

class OutputFrame:
    '''Frame for text output from the model'''

    def __init__(self, x, y, width, height, color=(100, 100, 100)):
        self.text = ""
        self.font = pygame.font.Font(None, 36)
        self.color = (211, 199, 185)
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def process(self):
        '''Renders the text output frame with the text.'''
        # Draw the background rectangle for the output frame
        frame_surface = pygame.Surface((self.width, self.height))
        frame_surface.fill(self.color)

        # Create a surface with the rendered text
        text_surface = self.font.render(self.text, True, (255, 255, 255))  # white text color

        # Center the text within the frame
        text_rect = text_surface.get_rect(center=(self.width // 2, self.height // 2))
        frame_surface.blit(text_surface, text_rect)

        # Blit the frame to the screen
        pygame.display.get_surface().blit(frame_surface, (self.x, self.y))

    def update_text(self, new_text):
        '''Updates the text to be displayed in the frame.'''
        self.text = new_text


# ---------------- Button ----------------

class Button:
    def __init__(self, x, y, width, height, text='', onclickFunction=None, font_size=40):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.onclickFunction = onclickFunction if onclickFunction else self.defaultFunction

        # Using RGB tuples for colors
        self.fillColors = {
            'normal': (220, 220, 220),
            'hover': (200, 200, 200),
            'pressed': (180, 180, 180),
        }
        self.textColor = (20, 20, 20)
        self.font = pygame.font.SysFont('Arial', font_size)

        self.buttonRect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.alreadyPressed = False
        self.onePress = False

    def process(self):
        mousePos = pygame.mouse.get_pos()
        mousePressed = pygame.mouse.get_pressed(num_buttons=3)[0]

        # Determine color based on mouse state
        if self.buttonRect.collidepoint(mousePos):
            if mousePressed:
                current_color = self.fillColors['pressed']
                if not self.alreadyPressed:
                    self.onclickFunction()
                    self.alreadyPressed = True
            else:
                current_color = self.fillColors['hover']
                self.alreadyPressed = False
        else:
            current_color = self.fillColors['normal']
            self.alreadyPressed = False

        # Draw a rounded rectangle button
        pygame.draw.rect(pygame.display.get_surface(), current_color, self.buttonRect, border_radius=8)

        # Render text and center it
        text_surf = self.font.render(self.text, True, self.textColor)
        text_rect = text_surf.get_rect(center=self.buttonRect.center)
        pygame.display.get_surface().blit(text_surf, text_rect)

    def defaultFunction(self):
        pass

    def is_clicked(self):
        return self.alreadyPressed, self.onePress


# ---------------- TextBox ----------------

class TextBox:
    def __init__(self, x, y, w, h, font_size=40):
        self.rect = pygame.Rect(x, y, w, h)
        self.base_color = (200, 200, 200)
        self.active_color = (255, 255, 255)
        self.color = self.base_color
        self.text = ""
        self.font = pygame.font.SysFont('Arial', font_size)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Toggle active status if the textbox is clicked
            if self.rect.collidepoint(event.pos):
                self.active = True
                self.color = self.active_color
            else:
                self.active = False
                self.color = self.base_color
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                # Optionally do something on Enter; here we just deactivate.
                self.active = False
                self.color = self.base_color
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode

    def draw(self, screen):
        # Draw the rectangle (with a border)
        pygame.draw.rect(screen, self.color, self.rect, 0, border_radius=5)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2, border_radius=5)
        # Render and draw the text inside the box with some padding
        text_surface = self.font.render(self.text, True, (0, 0, 0))
        screen.blit(text_surface, (self.rect.x + 10, self.rect.y + 10))


# ---------------- Scenes ----------------

class Homepage(SceneBase):
    def __init__(self, scene_manager, video_feed):
        super().__init__(scene_manager)
        self.video_feed = video_feed
        self.media_button = Button(30, 140, 400, 100, "Train from media")
        self.camera_button = Button(30, 300, 400, 100, "Open Camera")
        self.recognition_button = Button(30, 500, 400, 100, "Face recognition")
        # Load the image (make sure "a.png" is in the same directory)
        try:
            self.image = pygame.image.load("a.png")
            # Optionally scale the image to a desired size (example: 250x250)
            self.image = pygame.transform.scale(self.image, (250, 250))
        except Exception as e:
            print("Error loading image a.png:", e)
            self.image = None

    def ProcessInput(self, events, pressed_keys):
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                # Switch to video recording scene on Enter key (if desired)
                self.SwitchToScene(VideoRecord(self.scene_manager, self.video_feed))

    def Render(self, screen):
        screen.fill((211, 199, 185))  # A light gray background for a clean look
        self.media_button.process()
        self.camera_button.process()
        self.recognition_button.process()

        # Draw the image if it loaded successfully
        if self.image:
            screen.blit(self.image, (WIDTH - self.image.get_width() - 30, 30))

        # Check if any button was clicked to switch scenes
        if self.media_button.is_clicked()[0]:
            self.SwitchToScene(VideoRecord(self.scene_manager, self.video_feed))
        if self.camera_button.is_clicked()[0]:
            self.SwitchToScene(LiveDemo(self.scene_manager, self.video_feed))
        if self.recognition_button.is_clicked()[0]:
            self.SwitchToScene(FaceRecognition(self.scene_manager, self.video_feed))


class FaceRecognition(SceneBase):
    def __init__(self, scene_manager, video_feed):
        super().__init__(scene_manager)
        self.video_feed = video_feed
        self.home = Button(90, 30, 400, 100, "Home")
        self.toggle = Button(490, 30, 400, 100, "Take a snapshot")
        self.textbox = TextBox(890, 40, 300, 50)  # Textbox for entering name
        self.output = OutputFrame(50, HEIGHT - 100, 500, 50)
        self.detected_name = ""
        self.recognize()

    def ProcessInput(self, events, pressed_keys):
        for event in events:
            self.textbox.handle_event(event)

    def Render(self, screen):
        screen.fill((211, 199, 185))  # White background
        self.home.process()
        self.toggle.process()

        if self.home.is_clicked()[0]:
            self.SwitchToScene(Homepage(self.scene_manager, self.video_feed))

        # Draw the video feed (if available)
        frame_surface = self.video_feed.get_frame()
        if frame_surface:
            screen.blit(frame_surface, (320, 120))

        self.textbox.draw(screen)  # Draw the textbox

        if self.toggle.is_clicked()[0]:
            filename = f"img/{self.textbox.text}" if self.textbox.text else "snapshot"
            self.video_feed.save_snapshot(filename)  # Save snapshot with entered name

        self.output.process()
        self.output.update_text(self.detected_name)

    def clear_message(self):
        self.output.update_text("")

    def recognize(self):
        """Run face recognition in a separate thread."""
        thread = threading.Thread(target=self._run_recognition, daemon=True)
        thread.start()

    def _run_recognition(self):
        """Actual face recognition logic running in a thread."""
        result = subprocess.run(
            ["python", "Face_Recognition.py"],  # Replace with the actual script filename
            text=True,
            capture_output=True
        )
        detected_name = result.stdout.strip()
        if detected_name:
            self.detected_name = detected_name
        else:
            self.detected_name = "No face recognized"


class LiveDemo(SceneBase):
    def __init__(self, scene_manager, video_feed):
        super().__init__(scene_manager)
        self.video_feed = video_feed
        self.home = Button(90, 30, 400, 100, "Home")
        self.message = ""
        self.textbox = OutputFrame(50, HEIGHT - 100, 500, 50)

    def ProcessInput(self, events, pressed_keys):
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self.SwitchToScene(Homepage(self.scene_manager, self.video_feed))

    def Render(self, screen):
        screen.fill((211, 199, 185))
        frame_surface = self.video_feed.get_frame()

        if frame_surface:
            screen.blit(frame_surface, (320, 120))
            text_surface = pygame.font.SysFont('Arial', 40).render(self.message, True, (100, 100, 100))
            screen.blit(text_surface, (50, HEIGHT - 50))

        self.home.process()
        if self.home.is_clicked()[0]:
            self.SwitchToScene(Homepage(self.scene_manager, self.video_feed))

        self.textbox.update_text(self.video_feed.get_text())
        self.textbox.process()
        threading.Timer(1, lambda: self.clear_message()).start()

    def clear_message(self):
        self.textbox.update_text("")


class VideoRecord(SceneBase):
    def __init__(self, scene_manager, video_feed):
        super().__init__(scene_manager)
        self.video_feed = video_feed
        self.record_button = Button(400, 30, 400, 100, "Record")
        self.snapshot = Button(800, 30, 400, 100, "Take a snapshot")
        self.home = Button(90, 30, 400, 100, "Home")
        self.counter = 0
        self.recording_started = False
        self.message = ""
        self.textbox = OutputFrame(50, HEIGHT - 100, 500, 50)

    def ProcessInput(self, events, pressed_keys):
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self.SwitchToScene(Homepage(self.scene_manager, self.video_feed))

    def Update(self):
        pass

    def clear_message(self):
        self.message = ""

    def Render(self, screen):
        screen.fill((211, 199, 185))
        frame_surface = self.video_feed.get_frame()
        self.record_button.process()
        self.home.process()
        self.snapshot.process()
        self.textbox.process()

        if self.snapshot.is_clicked()[0]:
            res = self.video_feed.record(None)
            if res is None:  # nothing detected
                self.SwitchToScene(Homepage(self.scene_manager, self.video_feed))
                self.message = "No hand detected! Please try again"
                threading.Timer(5, lambda: self.clear_message()).start()

        if self.record_button.is_clicked()[0]:
            res = self.video_feed.record(self.counter)
            if self.recording_started:
                self.recording_started = False
                print("STOPPED RECORDING")
            else:
                self.recording_started = True
                print("BEGIN RECORDING")
                self.counter = 0

        if self.recording_started:
            res = self.video_feed.record(self.counter)
            if res is None:
                self.video_feed.record(self.counter)
                self.message = "No hand detected! Please try again"
                threading.Timer(5, lambda: self.clear_message()).start()
                self.recording_started = False
            self.counter += 1

        if frame_surface:
            screen.blit(frame_surface, (320, 120))
            text_surface = pygame.font.SysFont('Arial', 40).render(self.message, True, (100, 100, 100))
            screen.blit(text_surface, (50, HEIGHT - 50))

        if self.home.is_clicked()[0]:
            self.SwitchToScene(Homepage(self.scene_manager, self.video_feed))


# ---------------- Main Game Runner ----------------

def run_game():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Signify GUI")
    video_feed = LoadCV()  # Video feed for the scene
    scene_manager = SceneManager(None)
    homepage_scene = Homepage(scene_manager, video_feed)
    scene_manager.SwitchToScene(homepage_scene)
    scene_manager.Run(screen)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run_game()
