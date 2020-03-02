import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2) 
cap3 = cv2.VideoCapture(4)

i = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    _, frame2 = cap2.read()
    _, frame3 = cap3.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    cv2.imshow('frame2',gray2)
    cv2.imshow('frame3',gray3)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        print('hi')
        cv2.imwrite('out_{}.png'.format(i), gray2);
        i += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
