import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import tensorflow.keras
import random

label = ["scissors", "rock", "paper"]
model = tensorflow.keras.models.load_model("keras_model.h5")

you_score = 0
computer_score = 0

cap = cv2.VideoCapture(0)

image = Image.open("usefull images/ready_img.jpg")
image = image.resize((224, 224)) # resizing the image

# converting RGB image to BGR image
r, g, b = image.split() 
image = Image.merge('RGB', (b, g, r))
starting_window = Image.new('RGB', (448,224))

while True:
	while True:
		ret, frame = cap.read()	
		if ret:
			frame = cv2.flip(frame, 1)
			row, col, _ = frame.shape
			height = int(row/2)-114
			width = int(col/2)#-114

			cropped_frame = frame[height : height+224, width : width+224]

			#cv2.imshow("frame", frame)
			#cv2.imshow("cropped_frame", cropped_frame)
			cropped_frame_image = Image.fromarray(cropped_frame)
			starting_window.paste(cropped_frame_image,(0,0))
			starting_window.paste(image,(224,0))
			cv2.imshow('new',np.array(starting_window))

		key = cv2.waitKey(1) & 0xFF
		if(key == ord('q') or key == 27):
			break

	if key == 27:
		cap.release()
		break
	cv2.destroyAllWindows()


	data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

	user_image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

	# Normalize the image
	normalized_image_array = (user_image.astype(np.float32) / 127.0) - 1

	# Load the image into the array
	data[0] = normalized_image_array

	# run the inference
	prediction = np.argmax(model.predict(data))
	print("predected as " + label[prediction])

	#final window
	you_image = Image.open("usefull images/you.jpg")
	computer_image = Image.open("usefull images/com.jpg")

	random_no = random.randint(0, 2)
	comp_Choice = Image.open("computer\\{}.jpg".format(random_no))
	comp_Choice = comp_Choice.resize((224, 224)) 
	r, g, b = comp_Choice.split() 
	comp_Choice = Image.merge('RGB', (b, g, r))


	label = ["scissors", "rock", "paper"]
	if((prediction == 0 and random_no == 1) or (prediction == 1 and random_no == 2) or (prediction == 2 and random_no == 0)):
		computer_score += 1;
	elif prediction == random_no:
		pass
	else:
		you_score += 1;

	final_window = Image.new('RGB', (448,264))
	final_window.paste(you_image,(0,0))
	final_window.paste(computer_image,(224,0))
	final_window.paste(cropped_frame_image,(0,41))
	final_window.paste(comp_Choice,(224,41))

	final_window_array = np.array(final_window)
	cv2.rectangle(final_window_array, (0, 0), (447, 263), (0, 0, 0), 2)
	cv2.line(final_window_array, (0, 40), (448, 40), (0, 0, 0), 2)
	cv2.line(final_window_array, (224, 0), (224, 264), (0, 0, 0), 2)

	cv2.putText(final_window_array, "Score : {}".format(you_score), (5, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
	cv2.putText(final_window_array, "Score : {}".format(you_score), (5, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
	cv2.putText(final_window_array, "Score : {}".format(computer_score), (229, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 4)
	cv2.putText(final_window_array, "Score : {}".format(computer_score), (229, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

	cv2.imshow('final_window', final_window_array)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

print(" --------------------------------------------")
print("|   Your Score : {}   |  Computer Score : {} |".format(you_score,computer_score))
print(" --------------------------------------------")
if you_score > computer_score:
	print("----YOU WON----")
elif you_score < computer_score:
	print("----YOU LOST----")
else:
	print("----DRAW----")

