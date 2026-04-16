Hackathon Understanding each line of code - technical questions will be on the code
why did you choose value (e.g high parameter )
Minimum score 97.9% accuracy for classifications

Minimum score
0.03 (For Regressions, r squared → KPI) 
MSE (error)

3 Domains:
Machine Learning 
Computer Vision 
NLP 


[ Raw Images ]
↓
[ Data Strategy ] ← What you load, how you split, how you balance
↓
[ Preprocessing ] ← Resize, normalize, augment
↓
[ Architecture ] ← Pretrained CNN + custom classification head
↓
[ Training Strategy ] ← Learning rate, epochs, fine-tuning schedule
↓
[ Evaluation ] ← Accuracy, confusion matrix, per-class breakdown


Layer 1: Data Strategy

You don't use all 289K images. You scope strategically:

* Take the top 15–20 categories by image count (ensures balance)
* Target ~3,000–5,000 images per class
* Split 80% train / 10% validation / 10% test — always split before augmentation so your test set reflects real-world conditions

Why this matters: A balanced 15-class problem with 4K images per class is dramatically easier to hit 99% on than an imbalanced 50-class problem with wildly unequal samples.

Layer 2: Preprocessing

Every image needs to be transformed into a tensor a neural network can consume:

* Resize to 224×224 (standard input size for most pretrained CNNs)
* Normalize pixel values using ImageNet's mean and std — this is non-negotiable when using pretrained weights, because the model was trained expecting that exact distribution
* Augment training images only: random horizontal flips, small rotations, color jitter — this artificially multiplies your dataset and forces the model to generalize

Layer 3: Architecture — Transfer Learning

This is the conceptual heart of why 99% is achievable fast:


ImageNet Pretrained ResNet50 or EfficientNet-B3│├── Frozen early layers (detect edges, textures, basic shapes)│     These weights are already perfect — don't touch them│├── Partially unfrozen middle layers (detect patterns, parts)│     Fine-tune these gently with a low learning rate│└── Replaced final layer (was 1000 ImageNet classes)      → Your new layer: N fashion classes (e.g. 15)      → Train this from scratch with normal learning rate

PROBLEM: "Classify fashion images into N categories" DATASET CHOICE: DeepFashion → volume, label quality, diversity DATA STRATEGY: Scope to top 15 balanced classes (~60K images) MODEL CHOICE: EfficientNet-B3 pretrained on ImageNet WHY IT WORKS: Domain similarity → transfer learning is highly effective TRAINING: Phase 1 (head only) → Phase 2 (fine-tune backbone) EXPECTED RESULT: 98–99%+ on validation/test set TIME TO BUILD: ~90 minutes with clean pipeline




7. Check image width and height

Number of images checked: 600
Min width: 288 Max width: 512
Min height: 260 Max height: 512

EDA SUMMARY:

“The Food-101 dataset contains 101 balanced classes with 1000 images each.
 Images are mostly uniform in size (~512x512) with minor variation, making preprocessing straightforward.
 The dataset is clean, well-structured, and suitable for training a classification model without major adjustments.”


# -----------------------------------
# 11. Quick summary
# -----------------------------------

print("Dataset summary")
print("-" * 40)
print("Images directory:", images_dir)
print("Number of classes:", len(class_names))
print("Total images:", class_df["image_count"].sum())
print("Min images in a class:", class_df["image_count"].min())
print("Max images in a class:", class_df["image_count"].max())
print("Average images per class:", round(class_df["image_count"].mean(), 2))

Dataset summary
----------------------------------------
Images directory: /kaggle/input/datasets/dansbecker/food-101/food-101/food-101/images
Number of classes: 101
Total images: 101000
Min images in a class: 1000
Max images in a class: 1000
Average images per class: 1000.0

The roadmap

We’ll build it in this order:

1.  Prepare labels and filepaths 

What this does

This creates a table like:

*  one row = one image 
* filepath = where the image lives 
* label = the food category 

You should get about:

* 101000 rows
* 2 columns


2. Create train/validation split : Step 3: split the data into training and validation sets.

Now we divide them into:

* training set: the model learns from this 
* validation set: the model is tested on this while training 

A common split is:

* 80% train
* 20% validation

Because your dataset is balanced, we also want each split to keep the same class proportions. That is what stratify does.


What your output means

Train shape: (80800, 3)
Validation shape: (20200, 3)



This means:

* 80,800 images will be used to teach the model 
* 20,200 will be used to test how well it is learning during training 

The 3 means your dataframe has 3 columns, likely:

* filepath
* label
* label_idx

We split the dataset so the model can learn on one part and be tested on unseen but similar data to check if it actually understands — not just memorizes.



1.  Build TensorFlow datasets: 

A TensorFlow dataset is a data pipeline that:

*  reads each image from disk 
*  resizes it 
*  converts it into numbers 
*  groups images into batches 
*  feeds them efficiently to the model 

So instead of the model seeing:

“here’s a filename”

it sees:

“here’s a 224×224 image tensor and its label”



A) Pipeline is working:

Image batch shape: (32, 224, 224, 3)
Label batch shape: (32,)

Meaning:

* 32 → batch size (you process 32 images at once) 
* 224, 224 → every image has been resized correctly 
* 3 → RGB channels (color images) 

Translation:

“The model will receive batches of 32 color images, each sized 224×224”

B) Images & Labels Match

*  actual food images
*  correct labels like falafel, foie_gras, etc. 

This confirms:

*  images loaded correctly
*  labels are aligned correctly
*  preprocessing worked


Step 5: Build the transfer learning model

Uses a pretrained vision brain (EfficientNet)
 Keeps its knowledge frozen 
 Adds a new layer to classify your 101 foods 



*  Load pretrained model 
*  Add classification head 
*  Train frozen model 
*  Evaluate 
*  Optionally fine-tune a little more 


Questions:
Why did you choose this model? already trained. CNN using transfer learning. 
If we change one line of code, eg..... what would happen if you changed your epochs 
how to make training your data faster
