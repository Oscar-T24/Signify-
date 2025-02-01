import pygame
import sys
from main import LoadCV, HandTrackingDynamic

# Global flag for video feed visibility
show_video = False
import time


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
        while self.current_scene is not None:
            events = pygame.event.get()
            pressed_keys = pygame.key.get_pressed()

            # Handle input, update and render the current scene
            self.current_scene.ProcessInput(events, pressed_keys)
            self.current_scene.Update()
            self.current_scene.Render(screen)

            pygame.display.flip()  # Update display


class Button:
    def __init__(self, x, y, width, height, color, text='', onclickFunction=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.text = text
        self.onclickFunction = onclickFunction if onclickFunction else self.defaultFunction

        self.fillColors = {
            'normal': '#ffffff',
            'hover': '#666666',
            'pressed': '#333333',
        }

        self.buttonSurface = pygame.Surface((self.width, self.height))
        self.buttonRect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.buttonSurf = pygame.font.SysFont('Arial', 40).render(text, True, (20, 20, 20))

        self.alreadyPressed = False
        self.onePress = False

    def process(self):
        mousePos = pygame.mouse.get_pos()

        self.buttonSurface.fill(self.fillColors['normal'])
        if self.buttonRect.collidepoint(mousePos):
            self.buttonSurface.fill(self.fillColors['hover'])

            if pygame.mouse.get_pressed(num_buttons=3)[0]:
                self.buttonSurface.fill(self.fillColors['pressed'])

                if self.onePress:
                    self.onclickFunction()
                elif not self.alreadyPressed:
                    self.onclickFunction()
                    self.alreadyPressed = True
            else:
                self.alreadyPressed = False

        self.buttonSurface.blit(self.buttonSurf, [
            self.buttonRect.width / 2 - self.buttonSurf.get_rect().width / 2,
            self.buttonRect.height / 2 - self.buttonSurf.get_rect().height / 2
        ])
        pygame.display.get_surface().blit(self.buttonSurface, self.buttonRect)


    def defaultFunction(self):
        pass

    def is_clicked(self):
        return self.alreadyPressed, self.onePress


class Homepage(SceneBase):
    def __init__(self, scene_manager, video_feed):
        super().__init__(scene_manager)
        self.video_feed = video_feed
        self.media_button = Button(30, 140, 400, 100, (255, 100, 255), "Train from media")
        self.camera_button = Button(30, 300,400, 100, (255, 100, 255), "Open Camera")

    def ProcessInput(self, events, pressed_keys):
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                # Switch to video recording scene
                self.SwitchToScene(VideoRecord(self.scene_manager, self.video_feed))

    def Render(self, screen):
        screen.fill((255, 0, 0))  # Red background
        self.media_button.process()
        self.camera_button.process()

        if self.camera_button.is_clicked()[0]:
            self.SwitchToScene(VideoRecord(self.scene_manager, self.video_feed))


    def on_open_camera(self):
        """ Open camera feed. """
        global show_video
        show_video = not show_video


class VideoRecord(SceneBase):
    def __init__(self, scene_manager, video_feed):
        super().__init__(scene_manager)
        self.video_feed = video_feed
        self.record_button = Button(400, 30, 400, 100, (120, 100, 255), "Record")
        self.snapshot = Button(800, 30, 400, 100, (120, 100, 255), "Take a snapshot")
        self.home = Button(90, 30, 400, 100, (120, 100, 255), "Home")
        self.counter = 0
        self.recording_started = False

    def ProcessInput(self, events, pressed_keys):
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                # Switch to video recording scene
                self.SwitchToScene(Homepage(self.scene_manager, self.video_feed))

    def Update(self):
        pass

    def Render(self, screen):
        screen.fill((255, 255, 255))  # White background
        frame_surface = self.video_feed.get_frame()
        self.record_button.process()
        self.home.process()
        self.snapshot.process()

        if self.snapshot.is_clicked()[0]:
            res = self.video_feed.record(None)

            if res is None: # nothing detected
                self.SwitchToScene(Homepage(self.scene_manager, self.video_feed))
                text_surface = pygame.font.SysFont('Arial', 40).render("No hand was detected ! Try again", True,
                                                                   (100, 100, 100))
                screen.blit(text_surface, (100, 100))

        if self.record_button.is_clicked()[0]:
            time.sleep(1)
            res = self.video_feed.record(self.counter)

            if self.recording_started:
                self.recording_started = False
                print("STOPED RECORDING")
            else:
                self.recording_started = True
                print("BEGIN RECORDING")
                self.counter = 0

        if self.recording_started:
            res = self.video_feed.record(self.counter)
            if res is None:
                self.video_feed.record(self.counter)
                text_surface = pygame.font.SysFont('Arial', 40).render("No hand was detected ! Try again", True,
                                                                       (100, 100, 100))
                screen.blit(text_surface, (100, 100))
                self.recording_started = False
            self.counter +=1

        if frame_surface:
            screen.blit(frame_surface, (320, 120))  # Position the video inside Pygame window

       # text_surface = pygame.font.SysFont('Arial', 40).render(str(self.record_button.is_clicked()), True, (100, 100, 100))
       # screen.blit(text_surface, (100, 100))

        if self.home.is_clicked()[0]:
            self.SwitchToScene(Homepage(self.scene_manager, self.video_feed))



def run_game():
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))

    video_feed = LoadCV()  # Video feed for the scene
    scene_manager = SceneManager(None)  # Initialize scene manager without a scene

    # Now that the scene manager is initialized, create the Homepage scene
    homepage_scene = Homepage(scene_manager, video_feed)

    # Set the homepage as the initial scene
    scene_manager.SwitchToScene(homepage_scene)

    # Run the game loop
    scene_manager.Run(screen)

    pygame.quit()
    sys.exit()



if __name__ == "__main__":
    run_game()
