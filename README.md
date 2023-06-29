# Valorant-Enemy-Detection
Using object detection to locate enemies in Valorant and aim at them automatically. 

There is a video game called Valorant. The creators, Riot Games, describe it as: "VALORANT: a 5v5 character-based tactical FPS where precise gunplay meets unique agent abilities". The goal of the game is to kill all your opponents.
It is good to shoot your enemy. It is not good to shoot your teammate. It is great to shoot your enemy in the head.

Idea: What if we could create a program that could shoot your enemies precisely, every time? (Note: we are not going to "hack" the game at all for this project. The input will just be a screen grab, the output will just be either mouse or keyboard inputs)

Steps need:
1. "Vision"- See where the enemies are
2. "Aim"- Point your mouse/gun at the enemy--preferably their head
3. "Shoot"- Fire at the enemy to kill them

**Step One** is both the hardest and most interesting aspect of this project. 
So, what should "Vision" look like?
My ideas were either to create a mask or bounding box where the enemies are. I prefer using a mask, because to me it feels more precise and looks cooler. Unfortunately, due to model and data limitations, I used bounding boxes as labels.
So, if we are using bounding boxes, what does "Vision Look Like"?
(INSERT IMAGE)


To collect data, I went into Valorant with a friend and took screenshots of them doing different actions, with different backgrounds, as different characters. I captured and labeled 200 images, before realizing that to create a "sufficiently" good dataset would take a lot of time.
Luckily, I found two "good" datasets consisting of about ~9000 and ~4000 images, with bounding box labels for enemies and teammates, and enemies and heads, respectively.
(EXAMPLE IMAGES)
