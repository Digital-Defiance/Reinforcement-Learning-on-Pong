import pygame
class Striker:
    
    def __init__(self, x, y, width, height, color, screen):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.velocity = 5
        self.color = color
        self.screen = screen
        self.screen_height = screen.get_height()

        self.paddle= pygame.Rect(x, y, width, height)
        self.paddle_rect = pygame.draw.rect(self.screen, self.color, self.paddle)

    def display(self):
        self.paddle_rect = pygame.draw.rect(self.screen, self.color, self.paddle)

    def move(self, direction):
        if direction == 'up':
            self.y -= self.velocity
        elif direction == 'down':
            self.y += self.velocity
        
        # defining boundaries!
        if self.y < 0:
            self.y = 0
        elif self.y + self.height > self.screen_height:
            self.y = self.screen_height - self.height
        # so after moving we have to update original striker's position
        self.paddle.y = self.y


