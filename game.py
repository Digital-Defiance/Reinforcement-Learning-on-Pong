import pygame
from striker import Striker


WIDTH = 800
HEIGHT = 400

pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Pong')


# now making strikers
left_striker = Striker(0, 200, 15, 60, (0, 0, 255), screen)
right_striker = Striker(785, 200, 15, 60, (255, 0, 0), screen)


while True:
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    
    # handling user input for keys on both the players!
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        left_striker.move('up')
    if keys[pygame.K_d]:
        left_striker.move('down')
    if keys[pygame.K_LEFT]:
        right_striker.move('down')
    if keys[pygame.K_RIGHT]:
        right_striker.move('up')




    left_striker.display()
    right_striker.display()
    pygame.display.update()
    clock.tick(120)

