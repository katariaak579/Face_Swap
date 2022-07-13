import cv2
import numpy as np
import dlib

def extract_indices(nparray):
    index=None
    for i in nparray[0]:
        index=i
        break
    return i

img1=cv2.imread("Face_Swapping/Face_Swap_Images/Emma_Watson.jpg")
gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2=cv2.imread("Face_Swapping/Face_Swap_Images/Gigi_Hadid.jpg")
gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
mask=np.zeros_like(gray1)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Face_Swapping/shape_predictor_68_face_landmarks.dat")
faces1=detector(gray1)
img2_new_face=np.zeros_like(img2)

for face in faces1:
    landmarks=predictor(gray1,face)
    landmarks_points1=[]
    for i in range(0,68):
        x=landmarks.part(i).x
        y=landmarks.part(i).y
        landmarks_points1.append((x,y))
        # cv2.circle(img1,(x,y),5,(0,255,255),-1)
    
    points=np.array(landmarks_points1,dtype=np.int32)
    convexhull=cv2.convexHull(points)
    cv2.fillConvexPoly(mask,convexhull,255)
    # cv2.imshow("Face_Isfmage",mask)
    face_image=cv2.bitwise_and(img1,img1,mask=mask)
    # cv2.imshow("Face_Image",face_image)
    # cv2.polylines(img1,[convexhull],True,(255,0,0),3)

    # Make Triangles
    rect = cv2.boundingRect(convexhull)
    (x,y,w,h)=rect
    # cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),3)
    subdivis=cv2.Subdiv2D(rect)
    subdivis.insert(landmarks_points1)
    triangles = subdivis.getTriangleList()
    trinagles=np.array(triangles,dtype=np.int32)
    # print(triangles)
    triangle_index=[]
    for t in trinagles:
        pt1=(t[0],t[1])
        # print(pt1)
        pt2=(t[2],t[3])
        pt3=(t[4],t[5])
        # cv2.line(img1,pt1,pt2,(0,0,255),2)
        # cv2.line(img1,pt3,pt2,(0,0,255),2)
        # cv2.line(img1,pt1,pt3,(0,0,255),2)
        index_pt1=np.where((pt1==points).all(axis=1))
        index_pt1= extract_indices(index_pt1)
        # print(index1)
        index_pt2=np.where((pt2==points).all(axis=1))
        index_pt2=extract_indices(index_pt2)
        index_pt3=np.where((pt3==points).all(axis=1))
        index_pt3=extract_indices(index_pt3)
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle_pts = [index_pt1, index_pt2, index_pt3]
            triangle_index.append(triangle_pts)
        # print(triangle_index)


######   FOR FACE 2

faces2=detector(gray2)
for face in faces2:
    landmarks=predictor(gray2,face)
    landmarks_points2=[]
    for i in range(0,68):
        x=landmarks.part(i).x
        y=landmarks.part(i).y
        landmarks_points2.append((x,y))
        # cv2.circle(img2,(x,y),3,(0,255,255),-1)
    points_convul=np.array(landmarks_points2,dtype=np.int32)
    convexhull2=cv2.convexHull(points_convul)
    mask_for_convul=np.zeros_like(img2) 
    mask_for_convul=cv2.cvtColor(mask_for_convul,cv2.COLOR_BGR2GRAY)  
    cv2.fillConvexPoly(mask_for_convul,convexhull2,255)
    # cv2.imshow("hd",mask_for_convul)



for index_trinagle in triangle_index:

    #Face 1

    t1_pt1=landmarks_points1[index_trinagle[0]]
    t1_pt2=landmarks_points1[index_trinagle[1]]
    t1_pt3=landmarks_points1[index_trinagle[2]]

    # cv2.line(img1,t1_pt1,t1_pt2,(0,0,255),2)
    # cv2.line(img1,t1_pt3,t1_pt2,(0,0,255),2)
    # cv2.line(img1,t1_pt1,t1_pt3,(0,0,255),2)
    triangle1=np.array((t1_pt1,t1_pt2,t1_pt3),dtype=np.int32)
    rect1=cv2.boundingRect(triangle1)
    (x,y,w,h)=rect1
    # cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,255),2)
    crop_img1=img1[y:y+h,x:x+w]
    crop_img1_mask=np.zeros((h,w),np.uint8)
    # cv2.imshow("gs",crop_img1_mask)
    points1=np.array([[t1_pt1[0]-x,t1_pt1[1]-y],[t1_pt2[0]-x,t1_pt2[1]-y],[t1_pt3[0]-x,t1_pt3[1]-y]])
    cv2.fillConvexPoly(crop_img1_mask,points1,255)
    crop_img1=cv2.bitwise_and(crop_img1,crop_img1,mask=crop_img1_mask)

    # Face 2

    t2_pt1=landmarks_points2[index_trinagle[0]]
    t2_pt2=landmarks_points2[index_trinagle[1]]
    t2_pt3=landmarks_points2[index_trinagle[2]]

    # cv2.line(img2,t2_pt1,t2_pt2,(0,0,255),2)
    # cv2.line(img2,t2_pt3,t2_pt2,(0,0,255),2)
    # cv2.line(img2,t2_pt1,t2_pt3,(0,0,255),2)
    triangle2=np.array((t2_pt1,t2_pt2,t2_pt3),dtype=np.int32)

    rect2=cv2.boundingRect(triangle2)
    # cv2.imshow("fw",rect2)
    (x,y,w,h)=rect2
    # cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,255),2)
    crop_img2=img2[y:y+h,x:x+w]
    crop_img2_mask=np.zeros((h,w),np.uint8)
    points2=np.array([[t2_pt1[0]-x,t2_pt1[1]-y],[t2_pt2[0]-x,t2_pt2[1]-y],[t2_pt3[0]-x,t2_pt3[1]-y]],dtype=np.int32)
    cv2.fillConvexPoly(crop_img2_mask,points2,255)
    crop_img2=cv2.bitwise_and(crop_img2,crop_img2,mask=crop_img2_mask)
    

    # Warp
    points1 = np.float32(points1)
    points2 = np.float32(points2)
    M = cv2.getAffineTransform(points1, points2)
    warped_triangle = cv2.warpAffine(crop_img1, M, (w, h),flags=cv2.INTER_NEAREST)
    # cv2.imshow("fahd",warped_triangle)


    # Reconstructing new face
    img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
    img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
    img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
    img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
    
    
img2_face_mask = np.zeros_like(gray2)
img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
img2_face_mask = cv2.bitwise_not(img2_head_mask)
img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
final_face_swap= cv2.add(img2_head_noface, img2_new_face)
widht,height,_=img2_new_face.shape
(x,y,w,h)=cv2.boundingRect(convexhull2)
centre=(int((x+x+w)/2),int((y+y+h)/2))
final_face=cv2.seamlessClone(final_face_swap,img2,img2_head_mask,centre,cv2.MIXED_CLONE)
    
cv2.imshow("Image 1",img1)
# cv2.imshow("Crop Image 1",img2_new_face)
# cv2.imshow("Mask1",crop_img1_mask)
# cv2.imshow("Mask",mask)
cv2.imshow("Image 2",img2)
cv2.imshow("FINAL",final_face)
# cv2.imshow("BackGround",final_face_swap)
# cv2.imshow("Crop Image 2",crop_img2)
# cv2.imshow("Mask2",crop_img2_mask)
# cv2.imshow("WarpTriangle",warped_triangle)
cv2.waitKey(0)
cv2.destroyAllWindows()