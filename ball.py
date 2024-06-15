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

    def move(self):
        self.x += self.velocity_x
        self.y += self.velocity_y

        # updating ball position
        self.ball.x = self.x - self.radius
        self.ball.y = self.y - self.radius

        # now collisions for TOP and BOTTOM
        if self.ball.top <= 0 or self.ball.bottom >= self.screen_height:
            self.velocity_y = -self.velocity_y

        if self.ball.left <= 0 or self.ball.right >= self.screen_width:
            self.velocity_x = -self.velocity_x

    def check_collision(self, striker):
        # so here striker is a object

        if self.ball.colliderect(striker.striker):
            # it's logical we have to only reverse x velocity
            self.velocity_x = -self.velocity_x

    def collision_with_wall(self):
        if self.ball.left <= 0:
            return 'left'
        elif self.ball.right >= self.screen_width:
            return 'right'

        else:
            return None
