# Valorant-Enemy-Detection
Using object detection to locate enemies in Valorant and aim at them automatically. 

# Demo of Finished Project

![hippo](https://github.com/matt-wats/Valorant-Enemy-Detection/blob/main/Analysis%20Images/Data%20Images/valorant-vision-demo.gif)

This is a video/gif I took using the final model and program I wrote. I took a screen recording of me scrolling through screenshots I previously took in Valorant, and periodically hitting a key to activate the detection (twice per image).
After the key is pressed, it first centers your mouse (like how in Valorant your mouse is always in the center), then uses the trained model to detect enemies, then moves the mouse cursor onto the enemy (hopefully their head).

# Abstract

There is a video game called Valorant. The creators, Riot Games, describe it as: "VALORANT: a 5v5 character-based tactical FPS where precise gunplay meets unique agent abilities". The goal of the game is to kill all your opponents.
It is good to shoot your enemy. It is not good to shoot your teammate. It is great to shoot your enemy in the head.

Idea: What if we could create a program* that could shoot your enemies precisely, every time? (*Note: we are not going to "hack" the game at all for this project. The input will just be a screen grab, the output will just be either mouse or keyboard inputs)

Steps need:
1. "Vision"- See where the enemies are
2. "Aim"- Point your mouse/gun at the enemy--preferably their head
3. "Shoot"- Fire at the enemy to kill them

**Step One** is both the hardest and most interesting aspect of this project. 
So, what should "Vision" look like?
My ideas were either to create a mask or bounding box where the enemies are. I prefer using a mask, because to me it feels more precise and looks cooler. Unfortunately, due to model and data limitations, I used bounding boxes as labels.
So, if we are using bounding boxes, what does "Vision" Look Like?
![Valorant Enemy Vision](https://github.com/matt-wats/Valorant-Enemy-Detection/blob/main/Analysis%20Images/Data%20Images/enemy_detection.png "Valorant Enemy Vision")

# Data Collection

To collect data, I went into Valorant with a friend and took screenshots of them doing different actions, with different backgrounds, as different characters. I captured and labeled 200 images, before realizing that to create a "sufficiently" good dataset would take a lot of time, so I am maintaining these images as a test set, but nothing more.
Luckily, I found two "good" datasets consisting of about ~9000 and ~4000 images, with bounding box labels for enemies and teammates, and enemies and heads, respectively.

Potential Issues with the datasets:
1. Both datasets appear to be images taken in sequence, possibly from a video, which means that separating the data into a train and validation split may not work as one would hope: If we have a sequence of three images and split the middle into the validation split, while the model wouldn't have "seen" the image before, it would be a sort of interpolation between images it has seen, decreasing its performance and generalization on new and realistic datasets i.e. it could overfit and be difficult to judge when it occurs
2. In Valorant, enemies have an outline color to differentiate them from teammates. There are multiple options for what this color could be: red, yellow, etc. The large dataset only has enemies outlined in red, which could both make it too easy for the model to detect them by only checking for red, as well as decrease its performance and generalization on new and realistic datasets i.e. it could overfit and be difficult to judge when it occurs
3. There are a lot of duplicate images in both datasets (I wrote a script that would loop through all of the images and remove duplicates).

# Data Use

For this project, we are interested in finding enemies and their heads. So this leaves the question: What do we do with the larger dataset that likely contains relevant data, but is not exactly what we want?
My solution: We can use the larger dataset as a sort of "pre-training", to improve the model weights before training on the "real" dataset of 4000 images. To experiment with what training method is best, I employed 3 different pre-training experiments:
1. No extra pre-training (SPT- Sans Pre-Training)
2. Pre-train on the large dataset (WPT- With Pre-Training)**
3. "Pre-train" on a combination of the large dataset on the small dataset (CPT- Combined Pre-Training)**

**NOTE: For pre-training, I trained the model as though there were only a single class within the data, hoping it would learn what features of the images could constitue on object of relevence, and not worry about what classes exist within the image (which as we learn later, may have been a dumb idea)

We could discuss whether the third method of CPT is actually pre-training, but I found it to be an interesting enough idea to try. Before training, I hypthesized that their performances would rank from best to worst: CPT, WPT, SPT. My thought being that training on more data is better.

# Model

I looked at various models for this task, eventually settling on YOLOv8 (You Only Look Once), because it is supposed to be fast and works well with the data we found. I used YOLOv8n (the smallest model) because it is the fastest, and I thought that the data was simple enough with only two classes that we wouldn't need any larger model. The YOLOv8n detection model has 3.2 million parameters and a speed of 80.4ms on a CPU ONXX (https://github.com/ultralytics/ultralytics). The model is composed of three "parts": the backbone, middle, and detection head. For training, if pre-training was applied, the whole model was updated. For fine-tuning, I employed three different model training techniques where I would only update a given section of the model:
1. All model weights would be updated (FM- Full Model)
2. The backbone weights would be frozen (PM- Partial Model)
3. The detection head would be the only unfrozen weights (MM- Minimal Model)

This was to test how performance and training speed would be impacted with different weight updating techniques.

# Training

Using the pre-trained YOLOv8n weights given by Ultralytics, we "pre-trained" two models following the teqchniques outlined above: CPT and WPT.
For each of the three base-models--SPT, WPT, and CPT--we then fine-tuned them using the three model training techniques outline above--FM, PM, MM--giving us a total of nine final models:

|                   | SPT     | WPT    |  CPT   |
|      -            |:-------:|:------:|:------:|
| **Full Model**    | FM-SPT  | FM-WPT | FM-CPT |
| **Partial Model** | PM-SPT  | PM-WPT | PM-CPT |
| **Minimal Model** | MM-SPT  | MM-WPT | MM-CPT |

All training runs were done for 50 epochs, with learning rate schedules using linear warmup and decay. After training, the best model was chosen. A batch size of 16 was used. Images were combined into a 2x2 mosaic for training. During validation, batches of 16 images were processed separately, which looked like:
![Validation Batch Images](https://github.com/matt-wats/Valorant-Enemy-Detection/blob/main/Analysis%20Images/Data%20Images/val_batch1_labels.jpg "Validation Batch Images")

For fine-tuning, the average training times were:
| Model         | Time        |
| ------------- |:-----------:|
| Full Model    | 12.292 mins |
| Partial Model | 10.233 mins |
| Minimal Model | 9.617 mins  |

The validation losses for class and box during training for the FMs and MM-CPT are as follows:
![Validation Box Losses](https://github.com/matt-wats/Valorant-Enemy-Detection/blob/main/Analysis%20Images/Top%204/val_box_loss%20VS%20epoch.jpeg "Validation Box Losses")
![Validation Class Losses](https://github.com/matt-wats/Valorant-Enemy-Detection/blob/main/Analysis%20Images/Top%204/val_cls_loss%20VS%20epoch.jpeg "Validation Class Losses")

# Results

The MMs improve quickly, but also plateau quickly, always being worse than the FMs and typically worse than the PMs, with the CPTs being a notable exception.
The PMs sit between the MMs and FMs in terms of both speed and performance, unless a rigorous enough pre-training scheme is applied, in which case they are even worse than MMs.
The FMs take longer, but their performance is better, significantly enough for our data for them to be the clear winners.

When it comes to our pre-training techniques, something interesting has occured: I was wrong. In actuality, the models performance by pre-training is SPT is best. When we look at the Recall scores of the FMs on the validation set, we find:
|                   | SPT     | WPT    |  CPT   |
|      -            |:-------:|:------:|:------:|
| **Enemy Body**    | 0.88    | 0.87   | 0.87   |
| **Enemy Head**    | 0.51    | 0.49   | 0.51   |

Now I know what you're thinking-- Oh ok, they're all pretty close, but I see what you're talking about. The SPT is a little bit better.

**BUT WAIT!** What issues did we realize might occur during our data collect? Overfitting.
I know... I know... It absolutely couldn't happen to us. But just for fun, let's double check our models against that small test set I made.
|                   | SPT     | WPT    |  CPT   |
|      -            |:-------:|:------:|:------:|
| **Enemy Body**    | 0.75    | 0.73   | 0.74   |
| **Enemy Head**    | 0.35    | 0.31   | 0.26   |

Not only have all of the scores decreased (which isn't necessarily terrible as my test data has its quirks), the head recall gets worse with more pre-training. When we think about why, it may make sense that pre-training worsens the performance as it had to look at lots of images of both enemies and teammates that didn't have their heads labelled, which could make the model not consider the heads to be relevant, which would be very difficult to train itself out of. I would say that the Enemy Body Recall is nominally better with SPT, so we shouldn't pay too much attention to that.

In short, our pre-training data was both not suitable for our desired end goal, and we overfit our model on its data. A bad combination.

# Bonus Model Improvement

We, as experts, know that enemies have an outline color. So what we can do is look at the pixel values of the image data, and if there is no outline color in an area, then we know there isn't an enemy, and if there is an outline color within an area, then there might be an enemy. We could use this to either mask bad predictions of the model or to focus where the model "looks" for enemies. This is actually extremely beneficial when teammates are within an image, so that we don't accidentally shoot them.

My implemented process follows 4 steps:
1. Get image
2. Run image through YOLO detection model to get candidate objects
3. Get mask of enemy outline color (in this case yellow)
4. Remove objects not near color mask for final predictions

What does all this look like?:

Get image:

![Valorant Enemy Image](https://github.com/matt-wats/Valorant-Enemy-Detection/blob/main/Analysis%20Images/Data%20Images/enemy_image.png "Valorant Enemy Image")

Detect objects:

![Valorant Enemy Error Vision](https://github.com/matt-wats/Valorant-Enemy-Detection/blob/main/Analysis%20Images/Data%20Images/error_enemy_detection.png "Valorant Enemy Error Vision")

Get color mask:

![Valorant Enemy Mask Vision](https://github.com/matt-wats/Valorant-Enemy-Detection/blob/main/Analysis%20Images/Data%20Images/enemy_color_mask.png "Valorant Enemy Mask Vision")

Only detect objects near color mask:

![Valorant Enemy Vision](https://github.com/matt-wats/Valorant-Enemy-Detection/blob/main/Analysis%20Images/Data%20Images/enemy_detection.png "Valorant Enemy Vision")

# Step 2 (and 3)
Step 2 was very simple. Write a script that does the following:
```python
image = TAKE SCREENSHOT
potential_targets = model(image)
real_targets = color_mask(potential_targets)
if found "head" in real_targets:
  aim at "head"
else if found "body" in real_targets:
  aim at TOP OF "body"

if AIM:
  shoot
```
Which I did, but then I realized something: This is a cheat.

I know what you're thinking: You JUST realized that??

I don't know. I was just doing a fun computer vision project. Now that it's done, I've realized that 1. this would be really difficult for Valorant to detect and ban people for using and 2. It works way better than I anticipated. Testing this against images of the game made me realize that this would be supremely unfair to use, and I can't actually release it on github for just anyone to download. I am sorry.

So this is as far as my project will go. If we ignore potential misuse of this project, it was lots of fun and it's really cool!

# Future Improvements

1. I noticed that zooming in on images with enemies far away, and then processing them with the detection model made it possible to detect them. So, instead of just processing an image, we could either: process the original and 5 (or something) zoomed in images to increase recall, or we could use the color mask before prediction to focus/zoom in on sections of interest. Both of these ideas would increase accuracy, while decreasing speed of prediction.
2. We could dedicate time to creating larger and better datasets for our needs.
