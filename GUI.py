import pygame
import subprocess

# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pygame UI with Buttons")

# Define colors
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLUE = (100, 149, 237)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Define fonts
font = pygame.font.Font(None, 36)

# Define button properties
buttons = [
    {"label": "Run Script 1", "rect": pygame.Rect(200, 100, 200, 50), "color": BLUE, "script": "Landmark_Extraction.py"},
    {"label": "Run Script 2", "rect": pygame.Rect(200, 180, 200, 50), "color": GREEN, "script": "Face_Recognition.py"},
    {"label": "Run Script 3", "rect": pygame.Rect(200, 260, 200, 50), "color": RED, "script": "Visual_test.py"},
]

# Function to run an external Python script
def run_script(script_name):
    try:
        subprocess.run(["python", script_name], check=True)
    except Exception as e:
        print(f"Error running {script_name}: {e}")

# Main loop
running = True
while running:
    screen.fill(WHITE)

    # Draw buttons
    for button in buttons:
        pygame.draw.rect(screen, button["color"], button["rect"])
        text = font.render(button["label"], True, BLACK)
        text_rect = text.get_rect(center=button["rect"].center)
        screen.blit(text, text_rect)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            for button in buttons:
                if button["rect"].collidepoint(mouse_pos):
                    print(f"Running {button['script']}...")
                    run_script(button["script"])

    pygame.display.flip()

pygame.quit()
