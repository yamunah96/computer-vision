import cv2

cap=cv2.VideoCapture(0) # 0 means default camera,  1 means external camera
shape_count={
        'Triangle': 0,
        'Rectangle': 0,
        "Square": 0,
        'Pentagon': 0,
        'Hexagon': 0,
        'Circle': 0
}
while True:
    dummy, image=cap.read()
    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
    blur= cv2.GaussianBlur(gray, (5,5), 0)
    edges= cv2.Canny(blur, 50, 150)
   
    contours,_= cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) <300:
            continue
    #approximate the contour
        approx = cv2.approxPolyDP(contour, 0.02*cv2.arcLength(contour, True), True)
        vertices = len(approx)
        print(vertices)
        if vertices == 3:
            shape_name = 'Triangle'
            shape_count['Triangle'] += 1

        elif vertices == 4:
            x,y,w,h = cv2.boundingRect(approx)
            aspect_ratio = w/h
            if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                shape_name = 'Square'
                shape_count['Square'] += 1
            else:
                shape_name = 'Rectangle'
                shape_count['Rectangle'] += 1

        elif vertices == 5:
            shape_name = 'Pentagon'
            shape_count['Pentagon'] += 1

        elif vertices == 6:
            shape_name = 'Hexagon'
            shape_count['Hexagon'] += 1

        else:
            shape_name = 'Circle'
            shape_count['Circle'] += 1


        cv2.drawContours(image, [contour], -1, (0,255,0), 2)
        cv2.putText(image, shape_name, (approx.ravel()[0], approx.ravel()[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        cv2.imshow("shapes",image)
        for shape, count in shape_count.items():
            print(f"{shape}: {count}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

