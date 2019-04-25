import cv2
import numpy as np
import tensorflow as tf
from object_detection import utils
from utils import label_map_util



def detection() :   
     """Load a Frozen Graph"""
     cap=cv2.VideoCapture(r"C:\Users\AEMIE\Downloads\footage2.mp4")
     MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17' 
     PATH_TO_FROZEN_GRAPH = r'C:\Users\AEMIE\Downloads\models\models-master\research\object_detection\ssd_mobilenet_v1_coco_2017_11_17\frozen_inference_graph.pb'
     PATH_TO_LABELS = r'C:\Users\AEMIE\Downloads\models\models-master\research\object_detection\data\mscoco_complete_label_map.pbtxt'

     detection_graph = tf.Graph()
     with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            

     """Create the category index"""
     category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

     """ Prediction"""
     with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
             _ , image_np = cap.read() 
             image_np_expanded = np.expand_dims(image_np, axis=0)
             image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
             scores = detection_graph.get_tensor_by_name('detection_scores:0')
             classes = detection_graph.get_tensor_by_name('detection_classes:0')
             num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      
             (scores, classes, num_detections) = sess.run([scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
    
             classes=np.squeeze(classes).astype(np.int32)
             scores=np.squeeze(scores)
             num_detections = np.squeeze(num_detections)      
             
             

                            
def main(font,frame_no):
 
   cap=cv2.VideoCapture(r"C:\Users\AEMIE\Downloads\footage2.mp4")
   
   if cap.isOpened():
      ret,frame = cap.read()
   else:
       ret =False
   ret,frame1 = cap.read()
   ret,frame2 = cap.read()
   
   while ret:
      ret,frame = cap.read()
      frame_no+=1
      d=cv2.absdiff(frame1,frame2)

      grey=cv2.cvtColor(d,cv2.COLOR_BGR2GRAY)

      blur =cv2.GaussianBlur(grey,(21,21),0)
      ret,th=cv2.threshold(blur,35,255,cv2.THRESH_BINARY)
      dilated=cv2.dilate(th,np.ones((3,3),np.uint8),iterations=3)
      _,c,_=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
      edge = cv2.Canny(frame1,300,400)
      '''Perspective transform'''
      tl = [220, 160]
      tr = [448, 160]
      bl = [64, 359]
      br = [560, 359]
      pts1 = np.float32([tl, tr, bl, br])
      pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
      matrix = cv2.getPerspectiveTransform(pts1, pts2)
      res1= cv2.warpPerspective(frame1, matrix, (500,600))
      res2= cv2.warpPerspective(frame2, matrix, (500,600))
      
      d1=cv2.absdiff(res1,res2)

      grey1=cv2.cvtColor(d1,cv2.COLOR_BGR2GRAY)

      blur1 =cv2.GaussianBlur(grey1,(59,59),0)
      ret1,th1=cv2.threshold(blur1,20,255,cv2.THRESH_BINARY)
      dilated1=cv2.dilate(th1,np.ones((3,3),np.uint8),iterations=3)
      _,c1,_=cv2.findContours(dilated1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
      edge1 = cv2.Canny(res1 ,150,200) 
      for i in range(len(c1)) : 
           M1= cv2.moments(c1[i])
           cX1 = int(M1["m10"] / M1["m00"])
           cY1 = int(M1["m01"] / M1["m00"])
           cv2.circle(res1, (cX1, cY1), 5, (255,0,0), -1)
      '''----------------------------------------------'''
      contours_poly = [None]*len(c)
      boundRect = [None]*len(c) 
      for i, cvt in enumerate(c):
        contours_poly[i] = cv2.approxPolyDP(cvt, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        
      for i in range(len(c)):
        cv2.rectangle(frame1, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])),(255,0,0), 2)
        M = cv2.moments(c[i]) 
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(frame1, (cX, cY), 5, (0,0, 255), -1)
        
         
      cv2.imshow("Frame1",frame1)
      cv2.imshow("Edges",edge)
      cv2.imshow("Threshold",dilated)
      cv2.imshow("Perspective",res1)
      
      if cv2.waitKey(20) == 27:
         break
      frame1 = frame2
      ret,frame2= cap.read()
   cv2.destroyAllWindows()
   cap.release()
main(font = cv2.FONT_HERSHEY_SIMPLEX,frame_no=0) 

