# PyGame, Keras Neural Network Game
# Author: TrackLab (https://github.com/TrackLab)

####################################################

import pygame
import os
import time
import random
import pandas as pd
# Comment these imports if you don`t have keras, tensorflow or numpy installed. 
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.optimizers import RMSprop
import numpy as np

# Settings (These are values you can change without breaking the game immediately)
RecordData = True      # Change to False if you want the network to play. Keep on True to play yourself
LoadNN = False         # Change to True if you want to load a existing network.h5 file (You need to record a dataset atleast once!)
GameFPS = 60           # GameFPS (will affect game speed!)
ObsSpeed = 10          # Barrel movement speed
BoatSpeed = 10         # Boat movement speed
BoatScale = 256        # Boat scale
BarrelScale = 128      # Barrel scale
CrashedScale = 768     # Crashed sign scale
ObstacleRange = 100    # The range in which the barrels spawn (orientated from the boat)
ScrW, ScrH = 1280, 720 # Game screen size


# Do not change anything below unless you know what you are doing!
# Initiate Pygame
pygame.init()

# Images
boatImgForward = pygame.image.load('textures/BoatForward.png')
boatImgLeft = pygame.image.load('textures/BoatLeft.png')
boatImgRight = pygame.image.load('textures/BoatRight.png')
barrelImg = pygame.image.load('textures/Barrel.png')
crashedImg = pygame.image.load('textures/Crashed.png')

# Resize Images
boatImgForward = pygame.transform.scale(boatImgForward, (BoatScale ,BoatScale))
boatImgLeft = pygame.transform.scale(boatImgLeft, (BoatScale ,BoatScale))
boatImgRight = pygame.transform.scale(boatImgRight, (BoatScale ,BoatScale))
barrelImg = pygame.transform.scale(barrelImg, (BarrelScale, BarrelScale))
crashedImg = pygame.transform.scale(crashedImg, (CrashedScale, CrashedScale))

# Colors
black = (0,0,0)
white = (255,255,255)
water_blue = (40, 40, 255)

# Basic Game Initializers
gameDisplay = pygame.display.set_mode((ScrW, ScrH))
pygame.display.set_caption('PyNN')
clock = pygame.time.Clock()

# Takes care of turning the Network off in case the player plays
if RecordData:
    UseNN = False
else:
    UseNN = True
NNExists = False

# Neural Network
def Neural_Network(LoadModel=False):

    # Create Model
    if LoadModel:
        network = load_model('saved_models/network.h5')
        Exists = True
    else:
        if os.path.exists('training_data.csv'):
            # Read Training Data
            data = pd.read_csv('training_data.csv')
            boatX = data.boatX.tolist()
            obsX = data.obsX.tolist()
            obsY = data.obsY.tolist()
            disX = data.disX.tolist()
            disY = data.disY.tolist()
            data_distances = np.array([boatX,obsX,obsY,disX,disY])
            data_size = len(data)
            data_distances = np.reshape(data_distances, (data_size, 5))
            data_labels = np.array(np.expand_dims(data.direction, 1))

            # Optimizer
            rmsprop = RMSprop(lr=0.0001, decay=0.2)

            network = Sequential()
            network.add(Dense(units=32, activation='softsign', input_dim=5))
            network.add(Dense(units=3, activation='softplus'))
            network.compile(loss='sparse_categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
            network.summary()
            network.fit(data_distances, data_labels, epochs=5, batch_size=128, shuffle=False)
            network.save('saved_models/network.h5')
            Exists = True
        else:
            raise ValueError("Could not find training_data.csv for the Network. Did you record a dataset?")
    if Exists:
        return network, Exists
    else:
        return False

# Barrel Place Function
def Barrel(obs_x,obs_y, obs_w, obs_h):
    gameDisplay.blit(barrelImg,(obs_x,obs_y))

# Collision Calculation
def col_check(x,y,w,h,x2,y2,w2,h2):
    if (x < (x2 + w2) and (x + w) > x2 and y < (y2 + h2) and (h + y) > y2):                     
        return True
    else:
        return False

# Boat Placement Function
def boat(x,y, direction=1):
    directs = [boatImgLeft, boatImgForward, boatImgRight]
    gameDisplay.blit(directs[direction],(x,y))
    
# Display a message on the screen
def message_display(text, x=0, y=0):
    font = pygame.font.Font('freesansbold.ttf', 25)
    TextSurface = font.render(text, True, white)
    TextSurface, TextRect = TextSurface, TextSurface.get_rect()
    gameDisplay.blit(TextSurface, (x,y))

# Returns the Control Updates
def controls(x_pos, BoatSpeed, decision):
    if decision == 0:
        x = -BoatSpeed
        direction = decision
    if decision == 1:
        x = 0
        direction = decision
    if decision == 2:
        x = BoatSpeed
        direction = decision
    return x, direction

# Game Loop
def game_loop(RecordData):
    
    if UseNN:
        model, NNExists = Neural_Network(LoadNN)
    else:
        model, NNExists = None, False

    # Changing any of these will mostly likely break the game!
    global x_pos
    x_pos = 0
    counter = 0
    obsY = -600
    direction = 1
    gameExit = False
    ObsW = BarrelScale
    ObsH = BarrelScale
    y = (ScrH / 2 + 70)
    obsX = random.randrange(0, ScrW)
    x = ((ScrW / 2) - (BoatScale / 2))

    while not gameExit:

        # Iterate through every Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print('Game closed')
                gameExit = True
                if RecordData:
                    print('Converting data.txt into training_data.csv')
                    txt = pd.read_csv('data.txt')
                    txt.to_csv('training_data.csv', index=False)
                    os.remove('data.txt')
                return
            if not UseNN:
                if event.type == pygame.KEYDOWN:
                    # Left
                    if (event.key == pygame.K_LEFT or event.key == pygame.K_a):
                        x_pos, direction = controls(x_pos, BoatSpeed, 0)
                    # Right
                    if (event.key == pygame.K_RIGHT or event.key == pygame.K_d):
                        x_pos, direction = controls(x_pos, BoatSpeed, 2)
                # If Key Released
                if event.type == pygame.KEYUP:
                    if (event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT or event.key == pygame.K_d or event.key == pygame.K_a):
                        x_pos, direction = controls(x_pos, BoatSpeed, 1)

        # The Network and Recorder needs this
        x_distance = x-obsX 
        y_distance = y-obsY
        
        # Get Neural Network Decision
        if model and UseNN:

            nn_input = [[x, obsX, obsY, x_distance, y_distance]]
            nn_input = np.array(nn_input)

            decision = model.predict(nn_input)
            decision = decision[0]

            index = list(decision)
            index = index.index(max(index))
            decision = index

            DecStr = ['Left', 'None', 'Right']
            print(f"BoatX: {x}, BoatY: {y}, BarrelX: {obsX}, BarrelY: {obsY}, NN Decision: {DecStr[decision]}")

            # Neural Network Controls
            x_pos, direction = controls(x_pos, BoatSpeed, decision)

        # Updating x
        x += x_pos
        # Water Background
        gameDisplay.fill(water_blue)

        # Spawn Obstacle
        Barrel(obsX, obsY, ObsW, ObsH)
        obsY += ObsSpeed

        # Reset Obstacle when out of Screen
        if obsY > ScrH:
            obsY = 0 - ObsH
            if random.randint(0,1) == 0:   
                obsX = random.randint(x-ObstacleRange, x)
            else:
                obsX = random.randint(x, x+ObstacleRange)

        # Boat Boundaries Left/Right
        if x <= 1045 and x >= -20:
            boat(x,y, direction=direction)
            # Make Boat look forward on Boundaries
        else:
            # Boat reached Boundaries
            if x <= -20:
                x = -20
            if x >= 1045:
                x = 1045
            boat(x,y, direction=1)


        # Colission Check
        if col_check(x,y,BoatScale,BoatScale,obsX,obsY,BarrelScale,BarrelScale):
            # Show Crashed Image
            gameDisplay.blit(crashedImg,(ScrW/2 - CrashedScale/2, ScrH / 2 - CrashedScale/2))      
        

        # Print Obstacle Distances in Corner and NN Decision
        message_display(str(y_distance) + " " + str(x_distance), 0, 0)
        if direction == 0:
            if NNExists:
                message_display("NN Decision: Left", 0, 20)
            else:
                message_display("User Decision: Left", 0, 20)
        if direction == 2:
            if NNExists:
                message_display("NN Decision: Right", 0, 20)
            else:
                message_display("User Decision: Right", 0, 20)
        if direction == 1:
            if NNExists:
                message_display("NN Decision: None", 0, 20)
            else:
                message_display("User Decision: None", 0, 20)

        pygame.display.update()
        clock.tick(GameFPS)

        # Record Training data if RecordData is set to True
        if RecordData:
            if not os.path.exists('data.txt'):
                with open('data.txt', 'a') as f:
                    f.write("boatX,obsX,obsY,disX,disY,direction\n")
            else:
                with open('data.txt', 'a') as f:
                    f.write(str(x)+","+str(obsX)+","+str(obsY)+","+str(x_distance)+","+str(x_distance)+","+str(direction)+"\n")
            counter += 1
            if counter % 100 == 0:
                print("Collected", counter, "Samples")


if __name__ == "__main__":
    game_loop(RecordData=RecordData)
    pygame.quit()
    quit()
