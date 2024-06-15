import pygame

class Ball:
    
    def __init__(self, x, y, radius, velocity_x, velocity_y, color, screen):
        self.x = x
        self.y = y
        self.radius = radius
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.color = color
        self.screen = screen

        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()

        self.ball = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)

    def display(self):
        pygame.draw.ellipse(self.screen, self.color, self.ball)
